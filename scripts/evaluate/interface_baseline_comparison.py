#!/usr/bin/env python3
"""Compare all 5 models on the interface-proximal test subset.

Reports AUROC + 95% CI for each model and paired bootstrap deltas of
complex_gnn vs each baseline (10k iterations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


RESULTS = Path("results")


def bootstrap_auroc(labels, scores, n_bootstrap=10000, seed=42):
    """Bootstrap 95% CI for AUROC."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b, s_b = labels[idx], scores[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, s_b))
    boot_aucs = np.array(boot_aucs)
    return {
        "auroc": round(float(roc_auc_score(labels, scores)), 4),
        "mean": round(float(boot_aucs.mean()), 4),
        "std": round(float(boot_aucs.std()), 4),
        "ci_lower": round(float(np.percentile(boot_aucs, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_aucs, 97.5)), 4),
    }


def paired_bootstrap_delta(labels, scores_a, scores_b, n_bootstrap=10000, seed=42):
    """Paired bootstrap test: delta = AUROC(A) - AUROC(B)."""
    rng = np.random.RandomState(seed)
    n = len(labels)
    deltas = []
    wins_a = 0
    total = 0
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue
        auc_a = roc_auc_score(y_b, scores_a[idx])
        auc_b = roc_auc_score(y_b, scores_b[idx])
        deltas.append(auc_a - auc_b)
        if auc_a > auc_b:
            wins_a += 1
        total += 1
    deltas = np.array(deltas)
    p_value = 1.0 - (wins_a / max(total, 1))
    return {
        "delta_mean": round(float(deltas.mean()), 4),
        "delta_ci_lower": round(float(np.percentile(deltas, 2.5)), 4),
        "delta_ci_upper": round(float(np.percentile(deltas, 97.5)), 4),
        "p_value": round(float(p_value), 4),
    }


def load_predictions(model_dir: str) -> pd.DataFrame:
    """Load enriched or plain predictions."""
    base = RESULTS / "skempi" / model_dir
    enriched = base / "predictions_enriched.tsv"
    plain = base / "predictions.tsv"
    path = enriched if enriched.exists() else plain
    return pd.read_csv(path, sep="\t")


def main():
    # All 5 models
    models = {
        "sequence": "sequence_baseline",
        "monomer_gnn": "monomer_gnn",
        "complex_gnn": "complex_gnn",
        "logistic": "ddg_proxy_logistic",
        "hgb": "struct_hgb",
    }

    # Load predictions
    predictions = {}
    for name, model_dir in models.items():
        df = load_predictions(model_dir)
        if "split" in df.columns:
            df = df[df["split"] == "test"].copy()
        predictions[name] = df
        print(f"Loaded {name}: {len(df)} test samples")

    # Find common sample_ids across all models
    common_ids = None
    for name, df in predictions.items():
        ids = set(df["sample_id"].values)
        common_ids = ids if common_ids is None else common_ids & ids
    common_ids = sorted(common_ids)
    print(f"\nCommon samples across all 5 models: {len(common_ids)}")

    # Align all predictions by common sample_ids
    aligned = {}
    for name, df in predictions.items():
        df = df[df["sample_id"].isin(common_ids)].copy()
        df = df.set_index("sample_id").loc[common_ids].reset_index()
        aligned[name] = df

    # Get labels and interface-proximal mask
    ref_df = aligned["complex_gnn"]
    labels = ref_df["label"].values.astype(float)
    valid = ~np.isnan(labels)

    # Interface-proximal mask: distance <= 8A to partner chain
    # The interface_proximal column already uses this definition
    iface_col = ref_df.get("interface_proximal", pd.Series(0, index=ref_df.index))
    iface_mask = iface_col.astype(int).values == 1
    subset_mask = valid & iface_mask

    n_interface = int(subset_mask.sum())
    print(f"Interface-proximal test samples with valid labels: {n_interface}")

    if n_interface < 20:
        print("ERROR: Too few interface-proximal samples for reliable comparison")
        return

    iface_labels = labels[subset_mask]
    print(f"  Positive class: {int(iface_labels.sum())}, "
          f"Negative: {n_interface - int(iface_labels.sum())}")

    # Compute AUROC + CI for each model on interface-proximal subset
    print(f"\n{'='*70}")
    print("INTERFACE-PROXIMAL AUROC COMPARISON (all 5 models)")
    print(f"{'='*70}")

    model_results = {}
    for name in models:
        scores = aligned[name]["score"].values.astype(float)
        iface_scores = scores[subset_mask]
        mask = ~np.isnan(iface_scores)
        if mask.sum() < 20:
            print(f"  {name}: insufficient valid scores")
            continue
        ci = bootstrap_auroc(iface_labels[mask], iface_scores[mask])
        model_results[name] = ci
        print(f"  {name:20s}: AUROC = {ci['auroc']:.4f} "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # Paired bootstrap: complex_gnn vs each other model
    print(f"\n{'='*70}")
    print("PAIRED BOOTSTRAP: complex_gnn vs each baseline")
    print(f"{'='*70}")

    complex_scores = aligned["complex_gnn"]["score"].values.astype(float)[subset_mask]
    paired_results = {}
    for name in ["sequence", "monomer_gnn", "logistic", "hgb"]:
        other_scores = aligned[name]["score"].values.astype(float)[subset_mask]
        mask = ~np.isnan(complex_scores) & ~np.isnan(other_scores)
        if mask.sum() < 20:
            continue
        delta = paired_bootstrap_delta(
            iface_labels[mask], complex_scores[mask], other_scores[mask]
        )
        paired_results[f"complex_vs_{name}"] = delta
        print(f"  complex_gnn vs {name:15s}: delta = {delta['delta_mean']:+.4f} "
              f"[{delta['delta_ci_lower']:+.4f}, {delta['delta_ci_upper']:+.4f}] "
              f"p = {delta['p_value']:.4f}")

    # Also compute overall (all test) for comparison
    print(f"\n{'='*70}")
    print("OVERALL TEST SET AUROC (for reference)")
    print(f"{'='*70}")
    overall_results = {}
    for name in models:
        scores = aligned[name]["score"].values.astype(float)
        mask = valid & ~np.isnan(scores)
        if mask.sum() < 20:
            continue
        ci = bootstrap_auroc(labels[mask], scores[mask])
        overall_results[name] = ci
        print(f"  {name:20s}: AUROC = {ci['auroc']:.4f} "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    # Save results
    output = {
        "interface_proximal": {
            "n_samples": n_interface,
            "n_positive": int(iface_labels.sum()),
            "n_negative": n_interface - int(iface_labels.sum()),
            "model_aurocs": model_results,
            "paired_bootstrap": paired_results,
        },
        "overall": {
            "n_samples": int(valid.sum()),
            "model_aurocs": overall_results,
        },
    }

    output_path = RESULTS / "metrics" / "interface_proximal_comparison.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
