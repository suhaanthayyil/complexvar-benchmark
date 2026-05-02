#!/usr/bin/env python3
"""Full 5-fold protein-grouped cross-validation with fresh training per fold.

Trains both complex_gnn and monomer_gnn from scratch on V2 graphs (36 node,
11 edge features) for each fold. Uses GroupKFold on structure_id to prevent
same-complex leakage.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import roc_auc_score

try:
    import torch
    from torch_geometric.loader import DataLoader as GraphDataLoader
except ImportError as exc:
    raise RuntimeError("torch and torch-geometric required") from exc

from complexvar.models.gnn import ComplexVarGAT
from complexvar.models.train import (
    _load_graph_samples,
    _evaluate_graph_model,
    _device,
    masked_multitask_loss,
)

import copy
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


def train_one_fold(
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame,
    graph_column: str = "graph_path",
    batch_size: int = 32,
    epochs: int = 200,
    patience: int = 20,
    learning_rate: float = 1e-3,
) -> tuple[dict, pd.DataFrame]:
    """Train a GNN from scratch and evaluate on fold test set."""
    work_train = train_manifest.copy()
    work_val = val_manifest.copy()
    work_test = test_manifest.copy()

    work_train["graph_path"] = work_train[graph_column]
    work_val["graph_path"] = work_val[graph_column]
    work_test["graph_path"] = work_test[graph_column]

    train_samples = _load_graph_samples(work_train)
    val_samples = _load_graph_samples(work_val)
    test_samples = _load_graph_samples(work_test)

    # Auto-detect dimensions from first sample
    example = train_samples[0]
    node_dim = int(example.x.shape[1])
    edge_dim = int(example.edge_attr.shape[1])
    perturbation_dim = int(example.perturbation.shape[-1])

    model = ComplexVarGAT(
        node_dim=node_dim,
        edge_dim=edge_dim,
        perturbation_dim=perturbation_dim,
    )

    device = _device()
    model = model.to(device)

    train_loader = GraphDataLoader(train_samples, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_samples, batch_size=batch_size, shuffle=False)
    test_loader = GraphDataLoader(test_samples, batch_size=batch_size, shuffle=False)

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = copy.deepcopy(model.state_dict())
    best_val_auroc = float("-inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            outputs = model(batch)
            loss, _, _ = masked_multitask_loss(
                outputs=outputs,
                classification_targets=batch.classification_label,
                regression_targets=batch.regression_label,
                classification_mask=batch.classification_mask.bool(),
                regression_mask=batch.regression_mask.bool(),
            )
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validate
        val_metrics, _ = _evaluate_graph_model(model, val_loader)
        val_auroc = val_metrics.get("auroc", float("nan"))
        if pd.notna(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break

    model.load_state_dict(best_state)
    test_metrics, test_predictions = _evaluate_graph_model(model, test_loader)
    return test_metrics, test_predictions


def bootstrap_ci(values: list[float], n_boot: int = 10000, seed: int = 42) -> dict:
    """Bootstrap 95% CI for a list of values."""
    rng = np.random.RandomState(seed)
    arr = np.array(values)
    boot = [np.mean(rng.choice(arr, size=len(arr), replace=True)) for _ in range(n_boot)]
    boot = np.array(boot)
    return {
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "ci_lower": round(float(np.percentile(boot, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot, 97.5)), 4),
    }


def main():
    manifest_path = Path("data/processed/skempi_v2_graph_split_manifest_filtered.tsv")
    df = pd.read_csv(manifest_path, sep="\t")
    print(f"Loaded manifest: {len(df)} samples")

    # Use structure_id as the protein group for GroupKFold
    group_col = "structure_id" if "structure_id" in df.columns else "pdb_id"
    groups = df[group_col].values
    print(f"Grouping by {group_col}: {len(set(groups))} unique groups")

    gkf = GroupKFold(n_splits=5)
    fold_splits = list(gkf.split(df, groups=groups))

    results = {
        "complex_gnn": {"folds": [], "aurocs": []},
        "monomer_gnn": {"folds": [], "aurocs": []},
    }
    fold_deltas = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(fold_splits):
        print(f"\n{'='*60}")
        print(f"FOLD {fold_idx + 1}/5")
        print(f"{'='*60}")
        print(f"Train+Val: {len(train_val_idx)}, Test: {len(test_idx)}")

        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)

        # Split train_val into train and val using group-aware split
        tv_groups = train_val_df[group_col].values
        gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42 + fold_idx)
        train_idx_inner, val_idx_inner = next(gss.split(train_val_df, groups=tv_groups))
        train_df = train_val_df.iloc[train_idx_inner].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx_inner].reset_index(drop=True)

        print(f"  Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

        # Train complex GNN (uses graph_path = complex graph)
        print(f"\n  Training complex GNN for fold {fold_idx+1}...")
        t0 = time.time()
        complex_metrics, complex_preds = train_one_fold(
            train_df, val_df, test_df,
            graph_column="graph_path",
            batch_size=32,
        )
        complex_auroc = complex_metrics.get("auroc", float("nan"))
        print(f"  Complex GNN fold {fold_idx+1} AUROC: {complex_auroc:.4f} ({time.time()-t0:.0f}s)")

        # Train monomer GNN (uses monomer_graph_path)
        print(f"\n  Training monomer GNN for fold {fold_idx+1}...")
        t0 = time.time()
        monomer_metrics, monomer_preds = train_one_fold(
            train_df, val_df, test_df,
            graph_column="monomer_graph_path",
            batch_size=32,
        )
        monomer_auroc = monomer_metrics.get("auroc", float("nan"))
        print(f"  Monomer GNN fold {fold_idx+1} AUROC: {monomer_auroc:.4f} ({time.time()-t0:.0f}s)")

        results["complex_gnn"]["folds"].append({
            "fold": fold_idx + 1,
            "auroc": round(complex_auroc, 4),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        })
        results["complex_gnn"]["aurocs"].append(complex_auroc)

        results["monomer_gnn"]["folds"].append({
            "fold": fold_idx + 1,
            "auroc": round(monomer_auroc, 4),
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
        })
        results["monomer_gnn"]["aurocs"].append(monomer_auroc)

        delta = complex_auroc - monomer_auroc
        fold_deltas.append(delta)
        print(f"\n  Fold {fold_idx+1} delta (complex - monomer): {delta:+.4f}")

    # Compute summary statistics
    print(f"\n\n{'='*60}")
    print("CROSS-VALIDATION SUMMARY")
    print(f"{'='*60}")

    summary = {}
    for model_name in ["complex_gnn", "monomer_gnn"]:
        aurocs = results[model_name]["aurocs"]
        ci = bootstrap_ci(aurocs)
        summary[model_name] = {
            "mean_auroc": ci["mean"],
            "std_auroc": ci["std"],
            "ci_lower": ci["ci_lower"],
            "ci_upper": ci["ci_upper"],
            "per_fold_auroc": [round(a, 4) for a in aurocs],
        }
        print(f"{model_name:20s}: {ci['mean']:.4f} +/- {ci['std']:.4f} "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")
        print(f"  Per-fold: {[f'{a:.4f}' for a in aurocs]}")

    # Paired delta CI
    delta_ci = bootstrap_ci(fold_deltas)
    summary["paired_delta"] = {
        "mean_delta": delta_ci["mean"],
        "std_delta": delta_ci["std"],
        "ci_lower": delta_ci["ci_lower"],
        "ci_upper": delta_ci["ci_upper"],
        "per_fold_delta": [round(d, 4) for d in fold_deltas],
    }
    print(f"\nPaired delta (complex - monomer):")
    print(f"  Mean: {delta_ci['mean']:+.4f} [{delta_ci['ci_lower']:+.4f}, {delta_ci['ci_upper']:+.4f}]")
    print(f"  Per-fold: {[f'{d:+.4f}' for d in fold_deltas]}")

    # Save
    output_path = Path("results/crossval/crossval_v2_full.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
