#!/usr/bin/env python3
"""Statistical significance tests comparing model performance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    roc_auc_score,
)


def bootstrap_auroc(
    labels: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for AUROC."""
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
        "mean": round(float(boot_aucs.mean()), 4),
        "std": round(float(boot_aucs.std()), 4),
        "ci_lower": round(float(np.percentile(boot_aucs, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_aucs, 97.5)), 4),
    }


def bootstrap_spearman(
    true: np.ndarray,
    pred: np.ndarray,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Compute bootstrap confidence interval for Spearman correlation."""
    rng = np.random.RandomState(seed)
    n = len(true)
    boot_rhos = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        result = spearmanr(true[idx], pred[idx])
        rho = result.statistic if hasattr(result, "statistic") else result[0]
        rho = np.asarray(rho).item()
        boot_rhos.append(rho)
    boot_rhos = np.array(boot_rhos)
    return {
        "mean": round(float(boot_rhos.mean()), 4),
        "std": round(float(boot_rhos.std()), 4),
        "ci_lower": round(float(np.percentile(boot_rhos, 2.5)), 4),
        "ci_upper": round(float(np.percentile(boot_rhos, 97.5)), 4),
    }


def paired_bootstrap_test(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    metric_fn,
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> dict:
    """Paired bootstrap test: is model A better than model B?"""
    rng = np.random.RandomState(seed)
    n = len(labels)
    wins_a = 0
    total = 0
    deltas = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        y_b = labels[idx]
        if len(np.unique(y_b)) < 2:
            continue
        metric_a = metric_fn(y_b, scores_a[idx])
        metric_b = metric_fn(y_b, scores_b[idx])
        deltas.append(metric_a - metric_b)
        if metric_a > metric_b:
            wins_a += 1
        total += 1
    deltas = np.array(deltas)
    p_value = 1.0 - (wins_a / max(total, 1))
    return {
        "delta_mean": round(float(deltas.mean()), 4),
        "delta_ci_lower": round(float(np.percentile(deltas, 2.5)), 4),
        "delta_ci_upper": round(float(np.percentile(deltas, 97.5)), 4),
        "p_value": round(float(p_value), 4),
        "wins_a_frac": round(float(wins_a / max(total, 1)), 4),
    }


def delong_test(
    labels: np.ndarray,
    scores_a: np.ndarray,
    scores_b: np.ndarray,
) -> dict:
    """DeLong test for comparing two AUROCs on same data.

    Returns z-statistic and two-sided p-value.
    Based on the DeLong et al. (1988) method.
    """
    from scipy.stats import norm

    n1 = int(labels.sum())
    n0 = len(labels) - n1

    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]

    # Structural components for each model
    def _placement_values(scores):
        v10 = np.zeros(n1)
        v01 = np.zeros(n0)
        for i, pi in enumerate(pos_idx):
            v10[i] = np.mean(
                (scores[pi] > scores[neg_idx]).astype(float)
                + 0.5 * (scores[pi] == scores[neg_idx]).astype(float)
            )
        for j, nj in enumerate(neg_idx):
            v01[j] = np.mean(
                (scores[pos_idx] > scores[nj]).astype(float)
                + 0.5 * (scores[pos_idx] == scores[nj]).astype(float)
            )
        return v10, v01

    v10_a, v01_a = _placement_values(scores_a)
    v10_b, v01_b = _placement_values(scores_b)

    auc_a = v10_a.mean()
    auc_b = v10_b.mean()

    # Covariance matrix of the two AUCs
    s10 = np.cov(np.column_stack([v10_a, v10_b]).T)
    s01 = np.cov(np.column_stack([v01_a, v01_b]).T)

    s_matrix = s10 / n1 + s01 / n0

    diff = auc_a - auc_b
    var_diff = s_matrix[0, 0] + s_matrix[1, 1] - 2 * s_matrix[0, 1]

    if var_diff <= 0:
        return {
            "auc_a": round(float(auc_a), 4),
            "auc_b": round(float(auc_b), 4),
            "z_statistic": 0.0,
            "p_value": 1.0,
        }

    z = diff / np.sqrt(var_diff)
    p_value = 2 * norm.sf(abs(z))

    return {
        "auc_a": round(float(auc_a), 4),
        "auc_b": round(float(auc_b), 4),
        "z_statistic": round(float(z), 4),
        "p_value": round(float(p_value), 4),
    }


def main():
    output_dir = Path("results/significance")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load prediction files
    results_base = Path("results/skempi")
    models = {
        "sequence": "sequence_baseline",
        "monomer_gnn": "monomer_gnn",
        "complex_gnn": "complex_gnn",
    }

    predictions = {}
    for name, model_dir in models.items():
        enriched = results_base / model_dir / "predictions_enriched.tsv"
        plain = results_base / model_dir / "predictions.tsv"
        path = enriched if enriched.exists() else plain
        predictions[name] = pd.read_csv(path, sep="\t")

    # Align samples across all models using sample_id
    common_ids = None
    for name, df in predictions.items():
        if "split" in df.columns:
            df = df[df["split"] == "test"]
        ids = set(df["sample_id"].values)
        common_ids = ids if common_ids is None else common_ids & ids

    common_ids = sorted(common_ids)
    aligned = {}
    for name, df in predictions.items():
        if "split" in df.columns:
            df = df[df["split"] == "test"]
        df = df[df["sample_id"].isin(common_ids)]
        df = df.set_index("sample_id").loc[common_ids].reset_index()
        aligned[name] = df

    labels = aligned["complex_gnn"]["label"].values.astype(float)
    valid = ~np.isnan(labels)

    results = {}

    # 1) Bootstrap CIs for each model's AUROC
    print("Computing bootstrap confidence intervals...")
    for name in models:
        scores = aligned[name]["score"].values.astype(float)
        mask = valid & ~np.isnan(scores)
        ci = bootstrap_auroc(labels[mask], scores[mask])
        results[f"{name}_auroc_bootstrap"] = ci
        print(
            f"  {name}: AUROC = {ci['mean']:.4f} "
            f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        )

    # 2) Bootstrap CIs for Spearman (regression)
    print("\nBootstrap CIs for Spearman correlation...")
    for name in models:
        df = aligned[name]
        if "prediction" not in df.columns:
            continue
        ddg = df["ddg"].values.astype(float) if "ddg" in df.columns else None
        pred = df["prediction"].values.astype(float)
        if ddg is None:
            continue
        mask = np.isfinite(ddg) & np.isfinite(pred)
        if mask.sum() < 20:
            continue
        ci = bootstrap_spearman(ddg[mask], pred[mask])
        results[f"{name}_spearman_bootstrap"] = ci
        print(
            f"  {name}: Spearman = {ci['mean']:.4f} "
            f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]"
        )

    # 3) Paired bootstrap: complex vs monomer
    print("\nPaired bootstrap test: complex GNN vs monomer GNN...")
    scores_complex = aligned["complex_gnn"]["score"].values.astype(float)
    scores_monomer = aligned["monomer_gnn"]["score"].values.astype(float)
    mask = valid & ~np.isnan(scores_complex) & ~np.isnan(scores_monomer)

    paired = paired_bootstrap_test(
        labels[mask], scores_complex[mask], scores_monomer[mask],
        roc_auc_score,
    )
    results["complex_vs_monomer_bootstrap"] = paired
    print(
        f"  Delta AUROC = {paired['delta_mean']:.4f} "
        f"[{paired['delta_ci_lower']:.4f}, {paired['delta_ci_upper']:.4f}], "
        f"p = {paired['p_value']:.4f}"
    )

    # 4) Paired bootstrap: complex vs sequence
    print("\nPaired bootstrap test: complex GNN vs sequence...")
    scores_seq = aligned["sequence"]["score"].values.astype(float)
    mask_seq = valid & ~np.isnan(scores_complex) & ~np.isnan(scores_seq)

    paired_seq = paired_bootstrap_test(
        labels[mask_seq], scores_complex[mask_seq], scores_seq[mask_seq],
        roc_auc_score,
    )
    results["complex_vs_sequence_bootstrap"] = paired_seq
    print(
        f"  Delta AUROC = {paired_seq['delta_mean']:.4f} "
        f"[{paired_seq['delta_ci_lower']:.4f}, "
        f"{paired_seq['delta_ci_upper']:.4f}], "
        f"p = {paired_seq['p_value']:.4f}"
    )

    # 5) DeLong tests
    print("\nDeLong tests...")
    delong_cm = delong_test(
        labels[mask].astype(int),
        scores_complex[mask],
        scores_monomer[mask],
    )
    results["complex_vs_monomer_delong"] = delong_cm
    print(
        f"  Complex vs Monomer: z = {delong_cm['z_statistic']:.4f}, "
        f"p = {delong_cm['p_value']:.4f}"
    )

    delong_cs = delong_test(
        labels[mask_seq].astype(int),
        scores_complex[mask_seq],
        scores_seq[mask_seq],
    )
    results["complex_vs_sequence_delong"] = delong_cs
    print(
        f"  Complex vs Sequence: z = {delong_cs['z_statistic']:.4f}, "
        f"p = {delong_cs['p_value']:.4f}"
    )

    # 6) Interface-proximal subset tests
    print("\nInterface-proximal subset tests...")
    iface_col = aligned["complex_gnn"].get(
        "interface_proximal",
        pd.Series(0, index=aligned["complex_gnn"].index),
    )
    iface_mask = iface_col.astype(int).values == 1
    combined_mask = valid & iface_mask
    combined_mask = (
        combined_mask
        & ~np.isnan(scores_complex)
        & ~np.isnan(scores_monomer)
    )

    if combined_mask.sum() >= 20:
        iface_paired = paired_bootstrap_test(
            labels[combined_mask],
            scores_complex[combined_mask],
            scores_monomer[combined_mask],
            roc_auc_score,
        )
        results["interface_proximal_complex_vs_monomer"] = iface_paired
        print(
            f"  Interface-proximal delta AUROC = "
            f"{iface_paired['delta_mean']:.4f} "
            f"[{iface_paired['delta_ci_lower']:.4f}, "
            f"{iface_paired['delta_ci_upper']:.4f}], "
            f"p = {iface_paired['p_value']:.4f}"
        )

    # Save
    out_path = output_dir / "significance_results.json"
    out_path.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nResults saved to {out_path}")

    # Summary table
    print(f"\n{'='*70}")
    print("SIGNIFICANCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'Comparison':<40} {'Delta':>8} {'p-value':>8}")
    print(f"{'-'*70}")

    for key in [
        "complex_vs_monomer_bootstrap",
        "complex_vs_sequence_bootstrap",
        "interface_proximal_complex_vs_monomer",
    ]:
        if key in results:
            r = results[key]
            label = key.replace("_bootstrap", "").replace("_", " ")
            print(
                f"{label:<40} "
                f"{r['delta_mean']:>+8.4f} "
                f"{r['p_value']:>8.4f}"
            )


if __name__ == "__main__":
    main()
