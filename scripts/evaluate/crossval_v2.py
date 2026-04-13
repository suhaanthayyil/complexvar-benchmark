#!/usr/bin/env python3
"""
Cross-validation using V2 models and V2 graphs with matching dimensions.

This script uses:
- Models: results/skempi/complex_gnn_v2/ and monomer_gnn_v2/ (36 node, 11 edge)
- Graphs: data/processed/skempi_v2_graph_split_manifest_filtered.tsv (36 node, 11 edge)
- Properly matched dimensions throughout
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch_geometric.loader import DataLoader as GraphDataLoader

from complexvar.models.gnn import ComplexVarGAT
from complexvar.models.sequence import SequenceMLP, build_sequence_feature_vector
from complexvar.models.train import _evaluate_graph_model, _evaluate_tabular_model, _device
from complexvar.structure.mapping import load_structure_residues

warnings.filterwarnings("ignore")


def load_graph_samples_with_labels(manifest):
    """Load graphs and add labels."""
    samples = []
    cls_labels = manifest["binary_label"].fillna(0.0).to_numpy()
    reg_labels = manifest["ddg"].fillna(0.0).to_numpy()
    cls_mask = manifest["binary_label"].notna().astype(int).to_numpy()
    reg_mask = manifest["ddg"].notna().astype(int).to_numpy()

    for idx, row in manifest.iterrows():
        try:
            data = torch.load(row['graph_path'], map_location='cpu', weights_only=False)
            # Add labels
            data.classification_label = torch.tensor([cls_labels[idx]], dtype=torch.float32)
            data.regression_label = torch.tensor([reg_labels[idx]], dtype=torch.float32)
            data.classification_mask = torch.tensor([cls_mask[idx]], dtype=torch.float32)
            data.regression_mask = torch.tensor([reg_mask[idx]], dtype=torch.float32)
            samples.append(data)
        except Exception as e:
            print(f"Error loading {row.get('graph_path', 'unknown')}: {e}")
            continue
    return samples


def main():
    # Use V2 manifest with filtered valid graphs
    manifest_path = "data/processed/skempi_v2_graph_split_manifest_filtered.tsv"
    if not Path(manifest_path).exists():
        print(f"ERROR: {manifest_path} not found!")
        print("Using fallback to original manifest...")
        manifest_path = "data/processed/skempi_graph_split_manifest.tsv"

    df = pd.read_csv(manifest_path, sep="\t")
    print(f"Loaded manifest with {len(df)} samples")

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(df, groups=df.get("split_group", df.get("protein_cluster", range(len(df))))))

    print("Loading V2 models...")
    device = torch.device(_device())

    # Load V2 models (36 node, 11 edge)
    complex_model = ComplexVarGAT(node_dim=36, edge_dim=11, perturbation_dim=9)
    complex_state = torch.load("results/skempi/complex_gnn_v2/best_model.pt", map_location="cpu", weights_only=False)
    complex_model.load_state_dict(complex_state)
    complex_model.to(device)
    complex_model.eval()

    monomer_model = ComplexVarGAT(node_dim=36, edge_dim=11, perturbation_dim=9)
    monomer_state = torch.load("results/skempi/monomer_gnn_v2/best_model.pt", map_location="cpu", weights_only=False)
    monomer_model.load_state_dict(monomer_state)
    monomer_model.to(device)
    monomer_model.eval()

    print("Detected V2 model dimensions: node_dim=36, edge_dim=11")

    print("Loading graph data...")
    df_complex = df.copy()
    df_monomer = df.copy()
    if "monomer_graph_path" in df_monomer.columns:
        df_monomer["graph_path"] = df_monomer["monomer_graph_path"]

    complex_graphs = load_graph_samples_with_labels(df_complex)
    monomer_graphs = load_graph_samples_with_labels(df_monomer)

    print(f"Loaded {len(complex_graphs)} complex graphs, {len(monomer_graphs)} monomer graphs")

    # Build sequence features
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
    seq_model.load_state_dict(torch.load("results/skempi/sequence_baseline/best_model.pt", map_location="cpu", weights_only=False))
    seq_model.to(device)
    seq_model.eval()

    cls_labels = torch.tensor(df["binary_label"].fillna(0.0).to_numpy(), dtype=torch.float32)
    reg_labels = torch.tensor(df["ddg"].fillna(0.0).to_numpy(), dtype=torch.float32)
    cls_mask = torch.tensor(df["binary_label"].notna().astype(int).to_numpy(), dtype=torch.float32)
    reg_mask = torch.tensor(df["ddg"].notna().astype(int).to_numpy(), dtype=torch.float32)

    results = {"sequence": [], "monomer_gnn_v2": [], "complex_gnn_v2": []}

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

        # Get valid indices
        valid_monomer_idx = [i for i in test_idx if i < len(monomer_graphs)]
        valid_complex_idx = [i for i in test_idx if i < len(complex_graphs)]

        if valid_monomer_idx:
            fold_monomer = [monomer_graphs[i] for i in valid_monomer_idx]
            monomer_loader = GraphDataLoader(fold_monomer, batch_size=32)
            monomer_metrics, _ = _evaluate_graph_model(monomer_model, monomer_loader)
            results["monomer_gnn_v2"].append(monomer_metrics.get("auroc", np.nan))
        else:
            results["monomer_gnn_v2"].append(np.nan)

        if valid_complex_idx:
            fold_complex = [complex_graphs[i] for i in valid_complex_idx]
            complex_loader = GraphDataLoader(fold_complex, batch_size=32)
            complex_metrics, _ = _evaluate_graph_model(complex_model, complex_loader)
            results["complex_gnn_v2"].append(complex_metrics.get("auroc", np.nan))
        else:
            results["complex_gnn_v2"].append(np.nan)

    summary = {}
    for model_name, aurocs in results.items():
        valid_aurocs = [x for x in aurocs if not np.isnan(x)]
        if valid_aurocs:
            summary[model_name] = {
                "mean_auroc": float(np.mean(valid_aurocs)),
                "std_auroc": float(np.std(valid_aurocs)),
                "folds": [float(x) if not np.isnan(x) else None for x in aurocs]
            }
        else:
            summary[model_name] = {
                "mean_auroc": None,
                "std_auroc": None,
                "folds": [None] * len(aurocs)
            }

    Path("results/crossval").mkdir(parents=True, exist_ok=True)
    with open("results/crossval/crossval_summary_v2.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS (V2 MODELS)")
    print("="*60)
    for model_name, stats in summary.items():
        mean = stats['mean_auroc']
        std = stats['std_auroc']
        if mean is not None:
            print(f"{model_name:20s}: {mean:.3f} ± {std:.3f}")
            valid_folds = [f'{x:.3f}' if x is not None else 'N/A' for x in stats['folds']]
            print(f"  Folds: {', '.join(valid_folds)}")
        else:
            print(f"{model_name:20s}: N/A (no valid folds)")
    print("="*60)
    print("\nResults saved to results/crossval/crossval_summary_v2.json")

if __name__ == "__main__":
    main()
