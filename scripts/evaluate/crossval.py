#!/usr/bin/env python3
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch_geometric.nn import GATv2Conv
from sklearn.model_selection import GroupKFold
from torch_geometric.loader import DataLoader as GraphDataLoader

from complexvar.metrics.classification import compute_classification_metrics
from complexvar.models.sequence import SequenceMLP, build_sequence_feature_vector
from complexvar.models.train import _load_graph_samples, _evaluate_graph_model, _evaluate_tabular_model, _device
from complexvar.structure.mapping import load_structure_residues

warnings.filterwarnings("ignore")

class LegacyComplexVarGAT(nn.Module):
    def __init__(self, node_dim=28, edge_dim=3, perturbation_dim=9, hidden_dim=128, heads=4, num_layers=3, dropout=0.3):
        super().__init__()
        self.node_proj = nn.Linear(node_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim, hidden_dim)
        self.layers = nn.ModuleList([
            GATv2Conv(hidden_dim, hidden_dim//heads, heads=heads, edge_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        
        readout_input = hidden_dim + perturbation_dim
        self.readout = nn.Sequential(
            nn.Linear(readout_input, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classification_head = nn.Linear(64, 1)
        self.regression_head = nn.Linear(64, 1)

    def _get_mutant_embedding(self, x, data):
        if hasattr(data, "mutant_index"):
            mutant_idx = data.mutant_index
            if not isinstance(mutant_idx, torch.Tensor):
                mutant_idx = torch.as_tensor(mutant_idx, dtype=torch.long, device=x.device)
            if mutant_idx.ndim == 0:
                mutant_idx = mutant_idx.unsqueeze(0)
            
            # Shifted automatically by torch_geometric during batching
            global_indices = mutant_idx
            return x[global_indices]
        return x.mean(dim=0, keepdim=True)
        
    def forward(self, data):
        x = self.node_proj(data.x[:, :self.node_proj.in_features])
        edge_attr = self.edge_proj(data.edge_attr[:, :self.edge_proj.in_features])
        for layer, norm in zip(self.layers, self.norms):
            res = x
            x = layer(x, data.edge_index, edge_attr=edge_attr)
            x = norm(x)
            x = torch.relu(x)
            x = self.dropout(x)
            x = x + res
            
        mutant_embeddings = self._get_mutant_embedding(x, data)
        perturb = data.perturbation[:, :9]
        if perturb.ndim == 1:
            perturb = perturb.unsqueeze(0)
            
        combined = torch.cat([mutant_embeddings, perturb], dim=-1)
        hidden = self.readout(combined)
        return {
            "classification": self.classification_head(hidden).squeeze(-1),
            "regression": self.regression_head(hidden).squeeze(-1),
        }

def main():
    manifest_path = "data/processed/skempi_graph_split_manifest.tsv"
    df = pd.read_csv(manifest_path, sep="\t")
    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, groups=df["split_group"]))
    
    print("Loading graph data...")
    df_complex = df.copy()
    df_monomer = df.copy()
    if "monomer_graph_path" in df_monomer.columns:
        df_monomer["graph_path"] = df_monomer["monomer_graph_path"]
        
    complex_graphs = _load_graph_samples(df_complex)
    monomer_graphs = _load_graph_samples(df_monomer)
    
    print("Loading models...")
    device = torch.device(_device())
    
    complex_model = LegacyComplexVarGAT(node_dim=28, edge_dim=3)
    complex_model.load_state_dict(torch.load("results/skempi/complex_gnn/best_model.pt", map_location="cpu"))
    complex_model.to(device)
    complex_model.eval()
    
    monomer_model = LegacyComplexVarGAT(node_dim=28, edge_dim=3)
    monomer_model.load_state_dict(torch.load("results/skempi/monomer_gnn/best_model.pt", map_location="cpu"))
    monomer_model.to(device)
    monomer_model.eval()
    
    print("Building sequence features...")
    seq_features = []
    _CHAIN_SEQUENCE_CACHE = {}
    for row in df.itertuples(index=False):
        chain_id = row.mutated_chain_id if hasattr(row, "mutated_chain_id") and pd.notna(row.mutated_chain_id) else "A"
        cache_key = (row.structure_path, chain_id)
        if cache_key not in _CHAIN_SEQUENCE_CACHE:
            residues = load_structure_residues(row.structure_path)
            seq = "".join(r.residue_code for r in residues if r.chain_id == cache_key[1] and r.residue_code != "X")
            _CHAIN_SEQUENCE_CACHE[cache_key] = seq
        else:
            seq = _CHAIN_SEQUENCE_CACHE[cache_key]
            
        pos = int(row.residue_number) - 1
        if pos < 0 or pos >= len(seq):
            seq_features.append(np.array([]))
            continue
            
        encoding = build_sequence_feature_vector(sequence=seq, position=pos, wildtype=row.wildtype, mutant=row.mutant)
        seq_features.append(encoding.vector)
        
    valid_lens = [len(f) for f in seq_features if len(f) > 0]
    seq_dim = valid_lens[0] if valid_lens else 10
    for i, f in enumerate(seq_features):
        if len(f) == 0:
            seq_features[i] = np.zeros(seq_dim)
            
    seq_features_tensor = torch.tensor(np.array(seq_features), dtype=torch.float32)
    seq_model = SequenceMLP(input_dim=seq_dim)
    seq_model.load_state_dict(torch.load("results/skempi/sequence_baseline/best_model.pt", map_location="cpu"))
    seq_model.to(device)
    seq_model.eval()
    
    cls_labels = torch.tensor(df["binary_label"].fillna(0.0).to_numpy(), dtype=torch.float32)
    reg_labels = torch.tensor(df["ddg"].fillna(0.0).to_numpy(), dtype=torch.float32)
    cls_mask = torch.tensor(df["binary_label"].notna().astype(int).to_numpy(), dtype=torch.float32)
    reg_mask = torch.tensor(df["ddg"].notna().astype(int).to_numpy(), dtype=torch.float32)
    
    results = {"sequence": [], "monomer_gnn": [], "complex_gnn": []}
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"Evaluating fold {fold_idx + 1}...")
        
        test_f = seq_features_tensor[test_idx].to(device)
        test_c = cls_labels[test_idx].to(device)
        test_r = reg_labels[test_idx].to(device)
        test_cm = cls_mask[test_idx].to(device)
        test_rm = reg_mask[test_idx].to(device)
        
        from torch.utils.data import TensorDataset
        from torch.utils.data import DataLoader as TorchDataLoader
        
        seq_loader = TorchDataLoader(TensorDataset(test_f, test_c, test_r, test_cm, test_rm), batch_size=64)
        seq_metrics, _ = _evaluate_tabular_model(seq_model, seq_loader, "test")
        results["sequence"].append(seq_metrics.get("auroc", np.nan))
        
        fold_monomer = [monomer_graphs[i] for i in test_idx]
        monomer_loader = GraphDataLoader(fold_monomer, batch_size=64)
        monomer_metrics, _ = _evaluate_graph_model(monomer_model, monomer_loader)
        results["monomer_gnn"].append(monomer_metrics.get("auroc", np.nan))
        
        fold_complex = [complex_graphs[i] for i in test_idx]
        complex_loader = GraphDataLoader(fold_complex, batch_size=64)
        complex_metrics, _ = _evaluate_graph_model(complex_model, complex_loader)
        results["complex_gnn"].append(complex_metrics.get("auroc", np.nan))
        
    summary = {}
    for model_name, aurocs in results.items():
        summary[model_name] = {
            "mean_auroc": float(np.mean(aurocs)),
            "std_auroc": float(np.std(aurocs)),
            "folds": [float(x) for x in aurocs]
        }
        
    Path("results/crossval").mkdir(parents=True, exist_ok=True)
    with open("results/crossval/crossval_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
        
    print("Cross-validation summary saved.")

if __name__ == "__main__":
    main()
