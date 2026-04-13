#!/usr/bin/env python3
"""Train a hybrid model combining GNN embeddings with tabular features."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from torch_geometric.loader import DataLoader

from complexvar.models.gnn import ComplexVarGAT


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


TABULAR_FEATURES = [
    "min_inter_chain_distance",
    "relative_sasa",
    "burial_proxy",
    "local_degree",
    "inter_chain_contacts",
    "b_factor_or_plddt",
    "solvent_proxy",
]


def extract_gnn_embeddings(
    manifest: pd.DataFrame,
    checkpoint_path: str,
    graph_column: str = "graph_path",
) -> np.ndarray:
    """Extract penultimate-layer embeddings from a trained GNN."""
    device = _device()

    # Load one example to get dimensions
    example = torch.load(
        manifest.iloc[0][graph_column], map_location="cpu", weights_only=False
    )
    model = ComplexVarGAT(
        node_dim=int(example.x.shape[1]),
        edge_dim=int(example.edge_attr.shape[1]),
        perturbation_dim=int(example.perturbation.shape[-1]),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Collect graphs
    samples = []
    for row in manifest.itertuples(index=False):
        data = torch.load(
            getattr(row, graph_column), map_location="cpu", weights_only=False
        )
        samples.append(data)

    loader = DataLoader(samples, batch_size=64, shuffle=False)
    all_embeddings = []

    # Hook to capture penultimate layer
    embeddings_buffer = []

    def hook_fn(module, input, output):
        embeddings_buffer.append(output.detach().cpu())

    # Register hook on the last layer before the output heads
    handle = model.readout[-2].register_forward_hook(hook_fn)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            embeddings_buffer.clear()
            _ = model(batch)
            if embeddings_buffer:
                all_embeddings.append(embeddings_buffer[0].numpy())

    handle.remove()

    return np.vstack(all_embeddings)


def prepare_tabular(manifest: pd.DataFrame) -> np.ndarray:
    """Extract tabular features from manifest."""
    features = []
    for col in TABULAR_FEATURES:
        if col in manifest.columns:
            vals = manifest[col].copy()
            vals = vals.replace([np.inf, -np.inf], np.nan)
            median_val = vals.median()
            vals = vals.fillna(median_val if not np.isnan(median_val) else 0.0)
            features.append(vals.values)

    # Add perturbation features from constants
    from complexvar.constants import AMINO_ACIDS
    charge = {"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.5}
    hydro = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }
    volume = {
        "A": 88.6, "C": 108.5, "D": 111.1, "E": 138.4, "F": 189.9,
        "G": 60.1, "H": 153.2, "I": 166.7, "K": 168.6, "L": 166.7,
        "M": 162.9, "N": 114.1, "P": 112.7, "Q": 143.8, "R": 173.4,
        "S": 89.0, "T": 116.1, "V": 140.0, "W": 227.8, "Y": 193.6,
    }

    if "wildtype" in manifest.columns and "mutant" in manifest.columns:
        delta_charge = []
        delta_hydro = []
        delta_vol = []
        for _, row in manifest.iterrows():
            wt = str(row.get("wildtype", "A"))
            mt = str(row.get("mutant", "A"))
            delta_charge.append(charge.get(mt, 0) - charge.get(wt, 0))
            delta_hydro.append(hydro.get(mt, 0) - hydro.get(wt, 0))
            delta_vol.append(volume.get(mt, 0) - volume.get(wt, 0))
        features.extend([
            np.array(delta_charge),
            np.array(delta_hydro),
            np.array(delta_vol),
        ])

    return np.column_stack(features)


def main():
    manifest_path = "data/processed/skempi_graph_split_manifest.tsv"
    complex_checkpoint = "results/skempi/complex_gnn/best_model.pt"
    monomer_checkpoint = "results/skempi/monomer_gnn/best_model.pt"
    output_dir = Path("results/skempi/hybrid")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(manifest_path, sep="\t")

    # Split data
    train_mask = manifest["split"] == "train"
    val_mask = manifest["split"] == "val"
    test_mask = manifest["split"] == "test"

    print("Extracting complex GNN embeddings...")
    complex_emb = extract_gnn_embeddings(manifest, complex_checkpoint, "graph_path")
    print(f"  Complex embeddings shape: {complex_emb.shape}")

    print("Extracting monomer GNN embeddings...")
    monomer_emb = extract_gnn_embeddings(manifest, monomer_checkpoint, "monomer_graph_path")
    print(f"  Monomer embeddings shape: {monomer_emb.shape}")

    print("Preparing tabular features...")
    tabular = prepare_tabular(manifest)
    print(f"  Tabular features shape: {tabular.shape}")

    # Combine all features
    combined = np.hstack([complex_emb, monomer_emb, tabular])
    print(f"  Combined features shape: {combined.shape}")

    labels = manifest["binary_label"].values.astype(float)

    # Train HGB hybrid
    print("\nTraining HGB hybrid model...")
    train_valid = ~np.isnan(labels[train_mask.values])
    X_train = combined[train_mask.values][train_valid]
    y_train = labels[train_mask.values][train_valid]

    test_valid = ~np.isnan(labels[test_mask.values])
    X_test = combined[test_mask.values][test_valid]
    y_test = labels[test_mask.values][test_valid]

    # Replace any remaining NaN/inf in features
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    hgb = HistGradientBoostingClassifier(
        max_iter=200,
        max_depth=6,
        learning_rate=0.05,
        random_state=42,
    )
    hgb.fit(X_train, y_train)
    hgb_scores = hgb.predict_proba(X_test)[:, 1]

    hgb_auroc = roc_auc_score(y_test, hgb_scores)
    hgb_auprc = average_precision_score(y_test, hgb_scores)
    hgb_mcc = matthews_corrcoef(y_test, (hgb_scores > 0.5).astype(int))
    print(f"  HGB Hybrid: AUROC={hgb_auroc:.4f}, AUPRC={hgb_auprc:.4f}, MCC={hgb_mcc:.4f}")

    # Train MLP hybrid
    print("\nTraining MLP hybrid model...")
    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.15,
        random_state=42,
    )
    mlp.fit(X_train, y_train)
    mlp_scores = mlp.predict_proba(X_test)[:, 1]

    mlp_auroc = roc_auc_score(y_test, mlp_scores)
    mlp_auprc = average_precision_score(y_test, mlp_scores)
    mlp_mcc = matthews_corrcoef(y_test, (mlp_scores > 0.5).astype(int))
    print(f"  MLP Hybrid: AUROC={mlp_auroc:.4f}, AUPRC={mlp_auprc:.4f}, MCC={mlp_mcc:.4f}")

    # Interface-proximal subset
    test_manifest = manifest[test_mask].copy()
    test_prox = test_manifest["interface_proximal"].values[test_valid].astype(int) == 1
    if test_prox.sum() >= 10 and len(np.unique(y_test[test_prox])) >= 2:
        prox_hgb_auroc = roc_auc_score(y_test[test_prox], hgb_scores[test_prox])
        prox_mlp_auroc = roc_auc_score(y_test[test_prox], mlp_scores[test_prox])
        print(f"\n  Interface-proximal HGB: AUROC={prox_hgb_auroc:.4f}")
        print(f"  Interface-proximal MLP: AUROC={prox_mlp_auroc:.4f}")
    else:
        prox_hgb_auroc = None
        prox_mlp_auroc = None

    # Save results
    results = {
        "hybrid_hgb": {
            "overall": {
                "auroc": round(hgb_auroc, 4),
                "auprc": round(hgb_auprc, 4),
                "mcc": round(hgb_mcc, 4),
                "n": int(test_valid.sum()),
            },
            "interface_proximal": {
                "auroc": round(prox_hgb_auroc, 4) if prox_hgb_auroc else None,
            },
        },
        "hybrid_mlp": {
            "overall": {
                "auroc": round(mlp_auroc, 4),
                "auprc": round(mlp_auprc, 4),
                "mcc": round(mlp_mcc, 4),
                "n": int(test_valid.sum()),
            },
            "interface_proximal": {
                "auroc": round(prox_mlp_auroc, 4) if prox_mlp_auroc else None,
            },
        },
    }

    (output_dir / "hybrid_results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )

    # Save predictions
    pred_df = test_manifest[test_valid].copy()
    pred_df["hgb_hybrid_score"] = hgb_scores
    pred_df["mlp_hybrid_score"] = mlp_scores
    pred_df.to_csv(output_dir / "predictions.tsv", sep="\t", index=False)

    print(f"\nResults saved to {output_dir}")

    # Print comparison table
    print(f"\n{'='*60}")
    print("FULL MODEL COMPARISON")
    print(f"{'='*60}")
    print(f"{'Model':<25} {'AUROC':>8} {'AUPRC':>8} {'MCC':>8}")
    print(f"{'-'*60}")
    print(f"{'Sequence baseline':<25} {'0.6457':>8} {'0.6146':>8} {'0.2212':>8}")
    print(f"{'Monomer GNN':<25} {'0.7465':>8} {'0.6337':>8} {'0.3467':>8}")
    print(f"{'Complex GNN':<25} {'0.7538':>8} {'0.6345':>8} {'0.3530':>8}")
    print(f"{'Structure logistic':<25} {'0.7883':>8} {'0.6814':>8} {'0.3856':>8}")
    print(f"{'Structure HGB':<25} {'0.7919':>8} {'0.6872':>8} {'0.4081':>8}")
    print(f"{'Hybrid HGB':<25} {hgb_auroc:>8.4f} {hgb_auprc:>8.4f} {hgb_mcc:>8.4f}")
    print(f"{'Hybrid MLP':<25} {mlp_auroc:>8.4f} {mlp_auprc:>8.4f} {mlp_mcc:>8.4f}")


if __name__ == "__main__":
    main()
