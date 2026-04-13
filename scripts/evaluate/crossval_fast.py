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
from tqdm import tqdm

from complexvar.metrics.classification import compute_classification_metrics
from complexvar.models.sequence import SequenceMLP, build_sequence_feature_vector, _load_esm2_components
from complexvar.features import mutation_descriptor
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
            # PyG handles '*_index' attributes automatically during batching.
            return x[mutant_idx]
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
        if perturb.ndim == 1: perturb = perturb.unsqueeze(0)
        combined = torch.cat([mutant_embeddings, perturb], dim=-1)
        hidden = self.readout(combined)
        return {
            "classification": self.classification_head(hidden).squeeze(-1),
            "regression": self.regression_head(hidden).squeeze(-1),
        }

def get_esm_embeddings_cached(df, device="cpu"):
    tokenizer, model = _load_esm2_components()
    model = model.to(device)
    
    wt_cache = {}
    mt_cache = {}
    
    _SEQ_CACHE = {}
    
    def get_seq(struct_path, chain_id):
        key = (struct_path, chain_id)
        if key in _SEQ_CACHE: return _SEQ_CACHE[key]
        residues = load_structure_residues(struct_path)
        seq = "".join(r.residue_code for r in residues if r.chain_id == chain_id and r.residue_code != "X")
        _SEQ_CACHE[key] = seq
        return seq

    all_vectors = []
    print("Precomputing ESM embeddings...")
    for row in tqdm(df.itertuples(index=False), total=len(df)):
        chain_id = row.mutated_chain_id if hasattr(row, "mutated_chain_id") and pd.notna(row.mutated_chain_id) else "A"
        seq = get_seq(row.structure_path, chain_id)
        pos = int(row.residue_number) - 1
        
        if pos < 0 or pos >= len(seq):
            all_vectors.append(np.array([]))
            continue

        if seq not in wt_cache:
            with torch.no_grad():
                tokens = {k: v.to(device) for k, v in tokenizer(seq, return_tensors="pt").items()}
                outputs = model(**tokens).last_hidden_state[0].cpu().numpy()
                wt_cache[seq] = outputs
        
        wt_vec = wt_cache[seq][pos + 1]
        
        mt_key = (seq, pos, row.mutant)
        if mt_key not in mt_cache:
            mutant_sequence = seq[:pos] + row.mutant + seq[pos + 1 :]
            with torch.no_grad():
                tokens = {k: v.to(device) for k, v in tokenizer(mutant_sequence, return_tensors="pt").items()}
                outputs = model(**tokens).last_hidden_state[0].cpu().numpy()
                mt_cache[mt_key] = outputs[pos + 1]
        
        mt_vec = mt_cache[mt_key]
        delta = mutation_descriptor(wildtype=row.wildtype, mutant=row.mutant)
        vector = np.concatenate([wt_vec, mt_vec, np.asarray(list(delta.values()), dtype=float)])
        all_vectors.append(vector)
        
    
    valid_dim = next(v.shape[0] for v in all_vectors if len(v) > 0)
    for i in range(len(all_vectors)):
        if len(all_vectors[i]) == 0:
            all_vectors[i] = np.zeros(valid_dim)
    return np.array(all_vectors)

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
    
    device = _device()
    print(f"Using device: {device}")
    
    seq_vectors = get_esm_embeddings_cached(df, device=device)
    seq_features_tensor = torch.tensor(seq_vectors, dtype=torch.float32)
    
    print("Loading models...")
    complex_model = LegacyComplexVarGAT(node_dim=28, edge_dim=3).to(device)
    complex_model.load_state_dict(torch.load("results/skempi/complex_gnn/best_model.pt", map_location="cpu"))
    complex_model.eval()
    
    monomer_model = LegacyComplexVarGAT(node_dim=28, edge_dim=3).to(device)
    monomer_model.load_state_dict(torch.load("results/skempi/monomer_gnn/best_model.pt", map_location="cpu"))
    monomer_model.eval()
    
    seq_model = SequenceMLP(input_dim=seq_vectors.shape[1]).to(device)
    seq_model.load_state_dict(torch.load("results/skempi/sequence_baseline/best_model.pt", map_location="cpu"))
    seq_model.eval()
    
    cls_labels = torch.tensor(df["binary_label"].fillna(0.0).to_numpy(), dtype=torch.float32).to(device)
    reg_labels = torch.tensor(df["ddg"].fillna(0.0).to_numpy(), dtype=torch.float32).to(device)
    cls_mask = torch.tensor(df["binary_label"].notna().astype(int).to_numpy(), dtype=torch.float32).to(device)
    reg_mask = torch.tensor(df["ddg"].notna().astype(int).to_numpy(), dtype=torch.float32).to(device)
    
    results = {"sequence": [], "monomer_gnn": [], "complex_gnn": []}
    
    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        print(f"Evaluating fold {fold_idx + 1}...")
        
        from torch.utils.data import TensorDataset, DataLoader as TorchDataLoader
        seq_loader = TorchDataLoader(TensorDataset(seq_features_tensor[test_idx].to(device), cls_labels[test_idx], reg_labels[test_idx], cls_mask[test_idx], reg_mask[test_idx]), batch_size=64)
        seq_metrics, _ = _evaluate_tabular_model(seq_model, seq_loader, "test")
        results["sequence"].append(seq_metrics.get("auroc", np.nan))
        
        monomer_loader = GraphDataLoader([monomer_graphs[i] for i in test_idx], batch_size=64)
        monomer_metrics, _ = _evaluate_graph_model(monomer_model, monomer_loader)
        results["monomer_gnn"].append(monomer_metrics.get("auroc", np.nan))
        
        complex_loader = GraphDataLoader([complex_graphs[i] for i in test_idx], batch_size=64)
        complex_metrics, _ = _evaluate_graph_model(complex_model, complex_loader)
        results["complex_gnn"].append(complex_metrics.get("auroc", np.nan))
        
    summary = {}
    for model_name, aurocs in results.items():
        summary[model_name] = {"mean": float(np.mean(aurocs)), "std": float(np.std(aurocs)), "folds": [float(x) for x in aurocs]}
        
    Path("results/crossval").mkdir(parents=True, exist_ok=True)
    with open("results/crossval/crossval_summary.json", "w") as f: json.dump(summary, f, indent=2)
    print("Final results:", summary)

if __name__ == "__main__":
    main()
