#!/usr/bin/env python3
"""Run ablation studies on the trained complex GNN model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)
from torch_geometric.loader import DataLoader

from complexvar.models.gnn import ComplexVarGAT


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str, example_data) -> ComplexVarGAT:
    model = ComplexVarGAT(
        node_dim=int(example_data.x.shape[1]),
        edge_dim=int(example_data.edge_attr.shape[1]),
        perturbation_dim=int(example_data.perturbation.shape[-1]),
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_metrics(labels, scores, ddg_true=None, ddg_pred=None):
    result = {}
    valid = ~np.isnan(labels) & ~np.isnan(scores)
    labels_v, scores_v = labels[valid], scores[valid]
    if len(labels_v) >= 10 and len(np.unique(labels_v)) >= 2:
        preds_binary = (scores_v > 0.5).astype(int)
        result["auroc"] = round(float(roc_auc_score(labels_v, scores_v)), 4)
        result["auprc"] = round(float(average_precision_score(labels_v, scores_v)), 4)
        result["mcc"] = round(float(matthews_corrcoef(labels_v, preds_binary)), 4)
    result["n"] = int(valid.sum())
    if ddg_true is not None and ddg_pred is not None:
        valid_reg = ~np.isnan(ddg_true) & ~np.isnan(ddg_pred)
        if valid_reg.sum() >= 10:
            rho = spearmanr(ddg_true[valid_reg], ddg_pred[valid_reg])
            rho_val = np.asarray(
                rho.statistic if hasattr(rho, "statistic") else rho[0]
            ).item()
            result["spearman"] = round(rho_val, 4)
    return result


def run_ablation(
    manifest_path: str,
    checkpoint_path: str,
    ablation_type: str,
) -> dict:
    """Run one ablation on the test set.

    ablation_type can be:
        "none" -- no ablation (baseline)
        "remove_interchain_edges" -- remove all inter-chain edges at test time
        "zero_edge_distance" -- zero out edge distance features
        "zero_structural_features" -- zero out structural node features (SASA, SS, pLDDT)
        "shuffle_interchain" -- randomly shuffle which edges are inter-chain
    """
    manifest = pd.read_csv(manifest_path, sep="\t")
    test = manifest[manifest["split"] == "test"].copy()

    # Load graphs
    samples = []
    metadata = []
    for row in test.itertuples(index=False):
        graph_path = row.graph_path
        data = torch.load(graph_path, map_location="cpu", weights_only=False)

        # Apply ablation
        if ablation_type == "remove_interchain_edges":
            if hasattr(data, "edge_attr") and data.edge_attr.shape[1] >= 2:
                inter_chain_flag = data.edge_attr[:, 1]
                keep_mask = inter_chain_flag == 0
                data.edge_index = data.edge_index[:, keep_mask]
                data.edge_attr = data.edge_attr[keep_mask]
        elif ablation_type == "zero_edge_distance":
            if hasattr(data, "edge_attr"):
                data.edge_attr[:, 0] = 0.0
        elif ablation_type == "zero_structural_features":
            if hasattr(data, "x") and data.x.shape[1] >= 25:
                data.x[:, 20:] = 0.0  # Zero SASA, SS, pLDDT, chain
        elif ablation_type == "shuffle_interchain":
            if hasattr(data, "edge_attr") and data.edge_attr.shape[1] >= 2:
                perm = torch.randperm(data.edge_attr.shape[0])
                data.edge_attr[:, 1] = data.edge_attr[perm, 1]

        data.sample_id = row.sample_id
        samples.append(data)
        metadata.append({
            "sample_id": row.sample_id,
            "label": getattr(row, "binary_label", np.nan),
            "ddg": getattr(row, "ddg", np.nan),
            "interface_proximal": getattr(row, "interface_proximal", 0),
        })

    # Load model
    device = _device()
    model = load_model(checkpoint_path, samples[0])
    model = model.to(device)

    # Run inference
    loader = DataLoader(samples, batch_size=64, shuffle=False)
    all_scores = []
    all_regression = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch)
            scores = torch.sigmoid(outputs["classification"]).cpu().numpy()
            regression = outputs["regression"].cpu().numpy()
            all_scores.extend(scores.tolist())
            all_regression.extend(regression.tolist())

    meta_df = pd.DataFrame(metadata)
    meta_df["score"] = all_scores
    meta_df["prediction"] = all_regression

    # Compute metrics
    labels = meta_df["label"].values.astype(float)
    scores = np.array(all_scores)
    ddg_true = meta_df["ddg"].values.astype(float)
    ddg_pred = np.array(all_regression)

    results = {
        "ablation": ablation_type,
        "overall": compute_metrics(labels, scores, ddg_true, ddg_pred),
    }

    # Interface-proximal subset
    prox_mask = meta_df["interface_proximal"].astype(int) == 1
    if prox_mask.sum() >= 10:
        results["interface_proximal"] = compute_metrics(
            labels[prox_mask], scores[prox_mask],
            ddg_true[prox_mask], ddg_pred[prox_mask],
        )

    return results


def main():
    manifest_path = "data/processed/skempi_graph_split_manifest.tsv"
    checkpoint_path = "results/skempi/complex_gnn/best_model.pt"
    output_dir = Path("results/ablations")
    output_dir.mkdir(parents=True, exist_ok=True)

    ablations = [
        "none",
        "remove_interchain_edges",
        "zero_edge_distance",
        "zero_structural_features",
        "shuffle_interchain",
    ]

    all_results = {}
    for ablation in ablations:
        print(f"Running ablation: {ablation}")
        result = run_ablation(manifest_path, checkpoint_path, ablation)
        all_results[ablation] = result
        print(f"  Overall: {result['overall']}")
        if "interface_proximal" in result:
            print(f"  Interface-proximal: {result['interface_proximal']}")

    # Save
    output_path = output_dir / "ablation_results.json"
    output_path.write_text(json.dumps(all_results, indent=2) + "\n")
    print(f"\nResults saved to {output_path}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Ablation':<30} {'AUROC':>8} {'AUPRC':>8} {'Spear':>8} {'n':>6}")
    print(f"{'='*70}")
    for name, res in all_results.items():
        o = res["overall"]
        print(f"{name:<30} {o.get('auroc', 'N/A'):>8} {o.get('auprc', 'N/A'):>8} "
              f"{o.get('spearman', 'N/A'):>8} {o.get('n', 0):>6}")


if __name__ == "__main__":
    main()
