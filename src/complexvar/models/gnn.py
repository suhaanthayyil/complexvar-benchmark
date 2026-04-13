"""Graph neural network models."""

from __future__ import annotations

try:
    import torch
    from torch import nn
except ImportError:  # pragma: no cover
    torch = None
    nn = None

try:
    from torch_geometric.nn import GATv2Conv
except ImportError:  # pragma: no cover
    GATv2Conv = None


if nn is not None:

    class ComplexVarGAT(nn.Module):
        def __init__(
            self,
            node_dim: int,
            edge_dim: int,
            perturbation_dim: int,
            hidden_dim: int = 256,
            heads: int = 4,
            num_layers: int = 4,
            dropout: float = 0.3,
            use_cross_chain: bool = False,
        ) -> None:
            if GATv2Conv is None:
                raise RuntimeError("torch-geometric is required for ComplexVarGAT")
            super().__init__()
            self.node_dim = node_dim
            self.edge_dim = edge_dim
            self.perturbation_dim = perturbation_dim
            self.hidden_dim = hidden_dim
            self.node_proj = nn.Linear(node_dim, hidden_dim)
            self.edge_proj = nn.Linear(edge_dim, hidden_dim)
            self.num_layers = num_layers
            self.use_cross_chain = use_cross_chain
            self.layers = nn.ModuleList(
                [
                    GATv2Conv(
                        hidden_dim,
                        hidden_dim // heads,
                        heads=heads,
                        edge_dim=hidden_dim,
                        dropout=dropout,
                    )
                    for _ in range(num_layers)
                ]
            )
            self.norms = nn.ModuleList(
                [nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)]
            )
            self.dropout = nn.Dropout(dropout)

            # Cross-chain attention: attends from mutant node to partner
            # chain nodes, producing a partner-context vector
            readout_input = (hidden_dim * 4) + perturbation_dim
            if use_cross_chain:
                self.cross_chain_query = nn.Linear(hidden_dim, hidden_dim)
                self.cross_chain_key = nn.Linear(hidden_dim, hidden_dim)
                self.cross_chain_value = nn.Linear(hidden_dim, hidden_dim)
                self.cross_chain_norm = nn.LayerNorm(hidden_dim)
                readout_input += hidden_dim

            self.readout = nn.Sequential(
                nn.Linear(readout_input, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(512, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.ReLU(),
            )
            self.classification_head = nn.Linear(128, 1)
            self.regression_head = nn.Linear(128, 1)

        def _get_mutant_embedding(self, x, data):
            if not hasattr(data, "mutant_index"):
                if hasattr(data, "batch"):
                    from torch_geometric.nn import global_mean_pool

                    return global_mean_pool(x, data.batch)
                return x.mean(dim=0, keepdim=True)

            # PyTorch Geometric automatically shifts attributes ending in '_index'
            # during batching if they are correctly defined in the Data object.
            return x[data.mutant_index]

        def _graph_slices(self, data):
            if hasattr(data, "ptr"):
                ptr = data.ptr.cpu()
                return [
                    (int(ptr[i].item()), int(ptr[i + 1].item()))
                    for i in range(len(ptr) - 1)
                ]
            return [(0, int(data.x.shape[0]))]

        def _chain_groups(self, data):
            chain_ids = getattr(data, "chain_ids", [])
            if not chain_ids:
                return [[]]
            if isinstance(chain_ids[0], list):
                return [[str(value) for value in group] for group in chain_ids]
            return [[str(value) for value in chain_ids]]

        def _context_embeddings(self, x, data, mutant_embeddings):
            graph_slices = self._graph_slices(data)
            chain_groups = self._chain_groups(data)
            mutant_index = getattr(data, "mutant_index", None)
            if isinstance(mutant_index, torch.Tensor):
                if mutant_index.ndim == 0:
                    mutant_index = mutant_index.unsqueeze(0)
                mutant_index_list = [int(value.item()) for value in mutant_index]
            elif mutant_index is None:
                mutant_index_list = [0 for _ in graph_slices]
            else:
                mutant_index_list = [int(mutant_index)]

            neighborhood_embeddings = []
            same_chain_embeddings = []
            partner_chain_embeddings = []
            for graph_index, (start, end) in enumerate(graph_slices):
                local_x = x[start:end]
                if local_x.shape[0] == 0:
                    zero = torch.zeros(self.hidden_dim, device=x.device)
                    neighborhood_embeddings.append(zero)
                    same_chain_embeddings.append(zero)
                    partner_chain_embeddings.append(zero)
                    continue

                neighborhood_embeddings.append(local_x.mean(dim=0))

                chains = (
                    chain_groups[graph_index]
                    if graph_index < len(chain_groups)
                    else []
                )
                local_mutant_index = (
                    mutant_index_list[graph_index]
                    if graph_index < len(mutant_index_list)
                    else 0
                )
                local_mutant_index = max(
                    0,
                    min(local_mutant_index, local_x.shape[0] - 1),
                )
                mutant_chain = (
                    chains[local_mutant_index]
                    if chains and local_mutant_index < len(chains)
                    else None
                )
                if mutant_chain is None:
                    same_chain_embeddings.append(mutant_embeddings[graph_index])
                    partner_chain_embeddings.append(
                        torch.zeros(self.hidden_dim, device=x.device)
                    )
                    continue

                same_mask = torch.tensor(
                    [chain_id == mutant_chain for chain_id in chains],
                    dtype=torch.bool,
                    device=x.device,
                )
                partner_mask = ~same_mask
                same_chain_embeddings.append(local_x[same_mask].mean(dim=0))
                if partner_mask.any():
                    partner_chain_embeddings.append(local_x[partner_mask].mean(dim=0))
                else:
                    partner_chain_embeddings.append(
                        torch.zeros(self.hidden_dim, device=x.device)
                    )

            return (
                torch.stack(neighborhood_embeddings, dim=0),
                torch.stack(same_chain_embeddings, dim=0),
                torch.stack(partner_chain_embeddings, dim=0),
            )

        def _cross_chain_attention(self, x, data, mutant_embeddings):
            """Attend from mutant node embeddings to partner-chain nodes per graph."""
            graph_slices = self._graph_slices(data)
            chain_groups = self._chain_groups(data)
            mutant_index = getattr(data, "mutant_index", None)
            if isinstance(mutant_index, torch.Tensor):
                if mutant_index.ndim == 0:
                    mutant_index = mutant_index.unsqueeze(0)
                mutant_index_list = [int(value.item()) for value in mutant_index]
            elif mutant_index is None:
                mutant_index_list = [0 for _ in graph_slices]
            else:
                mutant_index_list = [int(mutant_index)]

            contexts = []
            for graph_index, (start, end) in enumerate(graph_slices):
                local_x = x[start:end]
                chains = (
                    chain_groups[graph_index]
                    if graph_index < len(chain_groups)
                    else []
                )
                if local_x.shape[0] == 0 or not chains:
                    contexts.append(torch.zeros(self.hidden_dim, device=x.device))
                    continue
                local_mutant_index = (
                    mutant_index_list[graph_index]
                    if graph_index < len(mutant_index_list)
                    else 0
                )
                local_mutant_index = max(
                    0,
                    min(local_mutant_index, local_x.shape[0] - 1),
                )
                mutant_chain = chains[local_mutant_index]
                partner_mask = torch.tensor(
                    [chain_id != mutant_chain for chain_id in chains],
                    dtype=torch.bool,
                    device=x.device,
                )
                if not partner_mask.any():
                    contexts.append(torch.zeros(self.hidden_dim, device=x.device))
                    continue
                q = self.cross_chain_query(
                    mutant_embeddings[graph_index : graph_index + 1]
                )
                k = self.cross_chain_key(local_x[partner_mask])
                v = self.cross_chain_value(local_x[partner_mask])
                scale = q.shape[-1] ** 0.5
                attn = torch.matmul(q, k.T) / scale
                attn = torch.softmax(attn, dim=-1)
                context = self.cross_chain_norm((attn @ v).squeeze(0))
                contexts.append(context)
            return torch.stack(contexts, dim=0)

        def forward(self, data):
            x = self.node_proj(data.x)
            edge_attr = self.edge_proj(data.edge_attr)
            for layer, norm in zip(self.layers, self.norms, strict=False):
                residual = x
                x = layer(x, data.edge_index, edge_attr=edge_attr)
                x = norm(x)
                x = torch.relu(x)
                x = self.dropout(x)
                x = x + residual

            mutant_embeddings = self._get_mutant_embedding(x, data)
            neighborhood_embeddings, same_chain_embeddings, partner_chain_embeddings = (
                self._context_embeddings(x, data, mutant_embeddings)
            )

            perturbation = data.perturbation
            if perturbation.ndim == 1:
                perturbation = perturbation.unsqueeze(0)

            if self.use_cross_chain:
                cross_context = self._cross_chain_attention(
                    x, data, mutant_embeddings
                )
                combined = torch.cat(
                    [
                        mutant_embeddings,
                        neighborhood_embeddings,
                        same_chain_embeddings,
                        partner_chain_embeddings,
                        cross_context,
                        perturbation,
                    ],
                    dim=-1,
                )
            else:
                combined = torch.cat(
                    [
                        mutant_embeddings,
                        neighborhood_embeddings,
                        same_chain_embeddings,
                        partner_chain_embeddings,
                        perturbation,
                    ],
                    dim=-1,
                )

            hidden = self.readout(combined)
            return {
                "classification": self.classification_head(hidden).squeeze(-1),
                "regression": self.regression_head(hidden).squeeze(-1),
            }
