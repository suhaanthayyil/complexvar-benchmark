#!/usr/bin/env python3
"""Generate all publication figures for the ComplexVar benchmark."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, precision_recall_curve, roc_curve

# -- Style ----------------------------------------------------------------

sns.set_theme(style="whitegrid", font_scale=1.2)
plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)

MODEL_COLORS = {
    "Sequence": "#888888",
    "Monomer GNN": "#3B82F6",
    "Complex GNN": "#EF4444",
    "Structure (logistic)": "#22C55E",
    "Structure (HGB)": "#F59E0B",
    "Hybrid HGB": "#A855F7",
    "Hybrid MLP": "#EC4899",
}

RESULTS = Path("results")
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def _load_predictions(model_dir: str) -> pd.DataFrame:
    path = RESULTS / "skempi" / model_dir / "predictions_enriched.tsv"
    if not path.exists():
        path = RESULTS / "skempi" / model_dir / "predictions.tsv"
    return pd.read_csv(path, sep="\t")


def _test_only(df: pd.DataFrame) -> pd.DataFrame:
    if "split" in df.columns:
        return df[df["split"] == "test"].copy()
    return df.copy()


def _interface_subset(df: pd.DataFrame, proximal: bool) -> pd.DataFrame:
    if "interface_proximal" not in df.columns:
        return df
    val = 1 if proximal else 0
    return df[df["interface_proximal"].astype(int) == val].copy()


# =========================================================================
# Figure 1: ROC and PR curves (4 panels)
# =========================================================================
def make_figure1() -> None:
    models = {
        "Sequence": "sequence_baseline",
        "Monomer GNN": "monomer_gnn",
        "Complex GNN": "complex_gnn",
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    panels = [
        ("A", "All variants", None, axes[0, 0], "roc"),
        ("B", "Interface-proximal", True, axes[0, 1], "roc"),
        ("C", "Interface-distal", False, axes[0, 2], "roc"),
        ("D", "All variants (PR)", None, axes[1, 0], "pr"),
        ("E", "Interface-proximal (PR)", True, axes[1, 1], "pr"),
        ("F", "Calibration (All)", None, axes[1, 2], "calibration"),
    ]

    for panel_label, title, proximal, ax, curve_type in panels:
        for model_name, model_dir in models.items():
            df = _test_only(_load_predictions(model_dir))
            df = df.dropna(subset=["label", "score"])
            if proximal is not None:
                df = _interface_subset(df, proximal)
            if len(df) < 10 or df["label"].nunique() < 2:
                continue

            labels = df["label"].astype(int).values
            scores = df["score"].astype(float).values
            color = MODEL_COLORS[model_name]

            if curve_type == "roc":
                fpr, tpr, _ = roc_curve(labels, scores)
                area = auc(fpr, tpr)
                ax.plot(fpr, tpr, color=color, lw=2,
                        label=f"{model_name} (AUROC={area:.3f})")
                ax.set_xlabel("False positive rate")
                ax.set_ylabel("True positive rate")
                ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
            elif curve_type == "pr":
                precision, recall, _ = precision_recall_curve(labels, scores)
                area = auc(recall, precision)
                ax.plot(recall, precision, color=color, lw=2,
                        label=f"{model_name} (AUPRC={area:.3f})")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
            elif curve_type == "calibration":
                prob_true, prob_pred = calibration_curve(labels, scores, n_bins=10)
                ax.plot(prob_pred, prob_true, "o-", color=color, lw=2,
                        label=f"{model_name}")
                ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4, label="Perfect")
                ax.set_xlabel("Mean predicted probability")
                ax.set_ylabel("Fraction of positives")

        ax.set_title(f"{panel_label}. {title}", fontweight="bold", loc="left")
        if curve_type != "calibration":
            legend_loc = "lower left" if curve_type == "roc" else "upper right"
            ax.legend(fontsize=8, loc=legend_loc)
        else:
            ax.legend(fontsize=7, loc="upper left")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)

    fig.tight_layout()
    fig.savefig(FIGURES / "fig1_roc_pr_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig1_roc_pr_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("Figure 1 saved.")


# =========================================================================
# Figure 2: SKEMPI regression scatter
# =========================================================================
def make_figure2() -> None:
    models = {
        "Sequence": "sequence_baseline",
        "Monomer GNN": "monomer_gnn",
        "Complex GNN": "complex_gnn",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for idx, (model_name, model_dir) in enumerate(models.items()):
        ax = axes[idx]
        df = _test_only(_load_predictions(model_dir))
        df = df.dropna(subset=["ddg", "prediction"])

        labels = df["ddg"].values
        preds = df["prediction"].values
        iface_col = df.get("interface_proximal", pd.Series(0, index=df.index))
        is_iface = iface_col.astype(int).values

        from scipy.stats import pearsonr, spearmanr
        r_val = pearsonr(labels, preds)
        rho_val = spearmanr(labels, preds)
        r_stat = r_val.statistic if hasattr(r_val, "statistic") else r_val[0]
        rho_stat = rho_val.statistic if hasattr(rho_val, "statistic") else rho_val[0]
        rho_stat = np.asarray(rho_stat).item()

        colors = np.where(is_iface == 1, MODEL_COLORS["Complex GNN"], "#BBBBBB")
        ax.scatter(labels, preds, c=colors, alpha=0.5, s=12, edgecolors="none")

        lo = min(labels.min(), preds.min()) - 0.5
        hi = max(labels.max(), preds.max()) + 0.5
        lims = [lo, hi]
        ax.plot(lims, lims, "k--", lw=0.8, alpha=0.4)

        ax.set_xlabel("Experimental ddG (kcal/mol)")
        ax.set_ylabel("Predicted ddG proxy")
        ax.set_title(model_name, fontweight="bold")
        ax.annotate(
            f"Pearson r = {r_stat:.3f}\nSpearman rho = {rho_stat:.3f}",
            xy=(0.05, 0.95), xycoords="axes fraction", va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
        )

    # Add legend for interface coloring
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=MODEL_COLORS["Complex GNN"],
               markersize=8, label="Interface-proximal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#BBBBBB",
               markersize=8, label="Interface-distal"),
    ]
    axes[2].legend(handles=legend_elements, fontsize=9, loc="lower right")

    fig.tight_layout()
    fig.savefig(FIGURES / "fig2_skempi_regression.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig2_skempi_regression.png", bbox_inches="tight")
    plt.close(fig)
    print("Figure 2 saved.")


# =========================================================================
# Figure 3: VUS reclassification by disease category
# =========================================================================
def make_figure3() -> None:
    vus_path = RESULTS / "vus_predictions" / "top_vus_scored.tsv"
    if not vus_path.exists():
        print("Skipping Figure 3: VUS predictions not found.")
        return

    df = pd.read_csv(vus_path, sep="\t")

    # Assign disease categories
    disease_map = {
        "cardiovascular": ["cardiomyopathy", "cardiac", "arrhythmia", "dilated",
                           "brugada", "long qt", "aortic", "heart"],
        "cancer": ["cancer", "tumor", "neoplasm", "carcinoma", "sarcoma",
                   "hereditary cancer", "cowden", "lynch"],
        "neurological": ["intellectual disability", "epilepsy", "neuropathy",
                         "charcot", "amyotrophic", "alexander", "ataxia",
                         "leukodystrophy", "lesch-nyhan"],
        "developmental": ["developmental", "congenital", "bardet", "skeletal",
                          "osteogenesis", "dyserythropoietic", "inborn"],
        "hematological": ["bleeding", "hemoglobin", "anemia", "platelet",
                          "thrombocytopenia", "hematological"],
    }

    def classify_disease(condition: str) -> str:
        if pd.isna(condition):
            return "other"
        cond_lower = condition.lower()
        for category, keywords in disease_map.items():
            if any(kw in cond_lower for kw in keywords):
                return category
        return "other"

    df["disease_category"] = df["condition"].apply(classify_disease)

    # Only show scored as likely pathogenic (probability > 0.5)
    pathogenic = df[df["pathogenicity_probability"] > 0.5].copy()

    category_counts = pathogenic["disease_category"].value_counts().sort_values()

    category_colors = {
        "cardiovascular": "#EF4444",
        "cancer": "#8B5CF6",
        "neurological": "#3B82F6",
        "developmental": "#22C55E",
        "hematological": "#F59E0B",
        "other": "#6B7280",
    }
    colors = [category_colors.get(c, "#6B7280") for c in category_counts.index]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(category_counts.index, category_counts.values, color=colors, edgecolor="white")
    ax.set_xlabel("Number of VUS scored as likely pathogenic")
    ax.set_title("VUS reclassification by disease category", fontweight="bold")

    for bar, count in zip(bars, category_counts.values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                str(count), va="center", fontsize=10)

    ax.set_xlim(0, category_counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig3_vus_reclassification.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig3_vus_reclassification.png", bbox_inches="tight")
    plt.close(fig)
    print("Figure 3 saved.")


# =========================================================================
# Figure 4: Model comparison summary heatmap
# =========================================================================
def make_figure4() -> None:
    """Summary heatmap of all metrics across models and variant subsets."""
    metrics_dir = RESULTS / "metrics"

    model_names = [
        "Sequence", "Monomer GNN", "Complex GNN",
        "Structure (logistic)", "Structure (HGB)",
        "Hybrid HGB",
    ]
    metric_prefixes = [
        "skempi_sequence", "skempi_monomer", "skempi_complex",
        "skempi_ddg_proxy", "skempi_struct_hgb",
        None,
    ]

    rows = []
    for model_name, prefix in zip(model_names, metric_prefixes):
        # Handle hybrid model separately (loads from its own results)
        if prefix is None and model_name == "Hybrid HGB":
            hybrid_path = RESULTS / "skempi" / "hybrid" / "hybrid_results.json"
            if hybrid_path.exists():
                hdata = json.loads(hybrid_path.read_text())
                hgb = hdata.get("hybrid_hgb", {})
                overall = hgb.get("overall", {})
                rows.append({
                    "Model": model_name,
                    "Subset": "All",
                    "AUROC": overall.get("auroc", np.nan),
                    "AUPRC": overall.get("auprc", np.nan),
                    "MCC": overall.get("mcc", np.nan),
                })
                iface_auroc = hgb.get("interface_proximal", {}).get("auroc")
                if iface_auroc is not None:
                    rows.append({
                        "Model": model_name,
                        "Subset": "Interface-proximal",
                        "AUROC": iface_auroc,
                        "AUPRC": np.nan,
                        "MCC": np.nan,
                    })
            continue

        all_path = metrics_dir / f"{prefix}_all_metrics.json"
        iface_path = metrics_dir / f"{prefix}_interface_metrics.json"

        if all_path.exists():
            data = json.loads(all_path.read_text())
            cls = data.get("classification", data)
            rows.append({
                "Model": model_name,
                "Subset": "All",
                "AUROC": cls.get("auroc", np.nan),
                "AUPRC": cls.get("auprc", np.nan),
                "MCC": cls.get("mcc", np.nan),
            })

        if iface_path.exists():
            data = json.loads(iface_path.read_text())
            for region_key, region_label in [
                ("interface_proximal", "Interface-proximal"),
                ("interface_distal", "Interface-distal"),
            ]:
                if region_key in data:
                    cls = data[region_key]
                    rows.append({
                        "Model": model_name,
                        "Subset": region_label,
                        "AUROC": cls.get("auroc", np.nan),
                        "AUPRC": cls.get("auprc", np.nan),
                        "MCC": cls.get("mcc", np.nan),
                    })

    table = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["AUROC", "AUPRC", "MCC"]

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        pivot = table.pivot(index="Model", columns="Subset", values=metric)
        col_order = ["All", "Interface-proximal", "Interface-distal"]
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
        row_order = [m for m in model_names if m in pivot.index]
        pivot = pivot.reindex(row_order)

        mask = pivot.isna()
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn", ax=ax,
            vmin=0, vmax=1 if metric != "MCC" else 0.5,
            mask=mask, linewidths=0.5,
            cbar_kws={"shrink": 0.8},
        )
        ax.set_title(metric, fontweight="bold")
        ax.set_ylabel("")

    fig.suptitle("Model comparison across variant subsets", fontweight="bold", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIGURES / "fig4_model_comparison.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig4_model_comparison.png", bbox_inches="tight")
    plt.close(fig)
    print("Figure 4 saved.")


# =========================================================================
# Supplementary: training curves
# =========================================================================
def make_supplementary_training_curves() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    models = {
        "Sequence": "sequence_baseline",
        "Monomer GNN": "monomer_gnn",
        "Complex GNN": "complex_gnn",
    }

    for idx, (model_name, model_dir) in enumerate(models.items()):
        ax = axes[idx]
        log_path = RESULTS / "skempi" / model_dir / "training_log.tsv"
        if not log_path.exists():
            continue
        log = pd.read_csv(log_path, sep="\t")
        ax.plot(log["epoch"], log["train_loss"], label="Train loss", color="#3B82F6")
        if "val_loss" in log.columns:
            ax.plot(log["epoch"], log["val_loss"], label="Val loss", color="#EF4444")
        if "val_auroc" in log.columns:
            ax2 = ax.twinx()
            ax2.plot(log["epoch"], log["val_auroc"], label="Val AUROC",
                     color="#22C55E", ls="--")
            ax2.set_ylabel("Val AUROC")
            ax2.legend(fontsize=8, loc="center right")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(model_name, fontweight="bold")
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(FIGURES / "supp_training_curves.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "supp_training_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("Supplementary training curves saved.")


# =========================================================================
# Figure 5: Ablation study bar chart
# =========================================================================
def make_figure5() -> None:
    """Bar chart showing ablation impact on AUROC and Spearman."""
    ablation_path = RESULTS / "ablations" / "ablation_results.json"
    if not ablation_path.exists():
        print("Skipping Figure 5: ablation results not found.")
        return

    data = json.loads(ablation_path.read_text())
    names = []
    aurocs = []
    spearmans = []
    display_names = {
        "none": "Baseline (no ablation)",
        "remove_interchain_edges": "Remove inter-chain edges",
        "zero_edge_distance": "Zero edge distances",
        "zero_structural_features": "Zero structural features",
        "shuffle_interchain": "Shuffle inter-chain labels",
    }

    for key in [
        "none",
        "remove_interchain_edges",
        "zero_edge_distance",
        "zero_structural_features",
        "shuffle_interchain",
    ]:
        if key in data:
            names.append(display_names.get(key, key))
            overall = data[key].get("overall", {})
            aurocs.append(overall.get("auroc", 0))
            spearmans.append(overall.get("spearman", 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    y_pos = np.arange(len(names))
    bar_colors = ["#22C55E"] + ["#EF4444"] * (len(names) - 1)

    # AUROC panel
    ax = axes[0]
    bars = ax.barh(y_pos, aurocs, color=bar_colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("AUROC")
    ax.set_title("A. Classification (AUROC)", fontweight="bold", loc="left")
    ax.set_xlim(0.5, 0.8)
    for bar, val in zip(bars, aurocs):
        ax.text(
            val + 0.003, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )

    # Spearman panel
    ax = axes[1]
    bars = ax.barh(y_pos, spearmans, color=bar_colors, edgecolor="white")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names)
    ax.set_xlabel("Spearman rho")
    ax.set_title(
        "B. Regression (Spearman)", fontweight="bold", loc="left",
    )
    ax.set_xlim(0.0, 0.55)
    for bar, val in zip(bars, spearmans):
        ax.text(
            val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )

    fig.suptitle(
        "Ablation study: complex GNN test-time feature removal",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(FIGURES / "fig5_ablation_study.pdf", bbox_inches="tight")
    fig.savefig(FIGURES / "fig5_ablation_study.png", bbox_inches="tight")
    plt.close(fig)
    print("Figure 5 saved.")


# =========================================================================
# Figure 6: Statistical significance forest plot
# =========================================================================
def make_figure6() -> None:
    """Forest plot of bootstrap confidence intervals."""
    sig_path = RESULTS / "significance" / "significance_results.json"
    if not sig_path.exists():
        print("Skipping Figure 6: significance results not found.")
        return

    data = json.loads(sig_path.read_text())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: AUROC bootstrap CIs
    ax = axes[0]
    model_keys = [
        ("sequence_auroc_bootstrap", "Sequence"),
        ("monomer_gnn_auroc_bootstrap", "Monomer GNN"),
        ("complex_gnn_auroc_bootstrap", "Complex GNN"),
    ]
    colors_list = ["#888888", "#3B82F6", "#EF4444"]
    y_vals = list(range(len(model_keys)))

    for i, ((key, label), color) in enumerate(
        zip(model_keys, colors_list)
    ):
        if key not in data:
            continue
        d = data[key]
        mean = d["mean"]
        lo = d["ci_lower"]
        hi = d["ci_upper"]
        ax.plot([lo, hi], [i, i], color=color, lw=3, solid_capstyle="round")
        ax.plot(mean, i, "o", color=color, markersize=10, zorder=5)
        ax.text(
            hi + 0.005, i,
            f"{mean:.3f} [{lo:.3f}, {hi:.3f}]",
            va="center", fontsize=9,
        )

    ax.set_yticks(y_vals)
    ax.set_yticklabels([mk[1] for mk in model_keys])
    ax.set_xlabel("AUROC (95% bootstrap CI)")
    ax.set_title(
        "A. Model AUROC with 95% CI", fontweight="bold", loc="left",
    )
    ax.set_xlim(0.55, 0.85)
    ax.invert_yaxis()

    # Panel B: Paired comparison deltas
    ax = axes[1]
    comparisons = [
        (
            "complex_vs_sequence_bootstrap",
            "Complex vs Sequence",
        ),
        (
            "complex_vs_monomer_bootstrap",
            "Complex vs Monomer",
        ),
        (
            "interface_proximal_complex_vs_monomer",
            "Complex vs Monomer\n(interface-proximal)",
        ),
    ]

    for i, (key, label) in enumerate(comparisons):
        if key not in data:
            continue
        d = data[key]
        delta = d["delta_mean"]
        lo = d["delta_ci_lower"]
        hi = d["delta_ci_upper"]
        p = d["p_value"]
        color = "#EF4444" if p < 0.05 else "#888888"
        ax.plot(
            [lo, hi], [i, i], color=color, lw=3, solid_capstyle="round",
        )
        ax.plot(delta, i, "D", color=color, markersize=8, zorder=5)
        p_str = f"p={p:.4f}" if p >= 0.0001 else "p<0.0001"
        ax.text(
            hi + 0.002, i,
            f"+{delta:.4f} [{lo:+.4f}, {hi:+.4f}] {p_str}",
            va="center", fontsize=9,
        )

    ax.axvline(0, color="black", lw=0.8, ls="--", alpha=0.5)
    ax.set_yticks(range(len(comparisons)))
    ax.set_yticklabels([c[1] for c in comparisons])
    ax.set_xlabel("Delta AUROC (95% bootstrap CI)")
    ax.set_title(
        "B. Pairwise comparisons", fontweight="bold", loc="left",
    )
    ax.set_xlim(-0.02, 0.18)
    ax.invert_yaxis()

    fig.suptitle(
        "Statistical significance of model differences",
        fontweight="bold", fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(
        FIGURES / "fig6_significance.pdf", bbox_inches="tight",
    )
    fig.savefig(
        FIGURES / "fig6_significance.png", bbox_inches="tight",
    )
    plt.close(fig)
    print("Figure 6 saved.")


# =========================================================================
# Main
# =========================================================================
if __name__ == "__main__":
    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    make_figure5()
    make_figure6()
    make_supplementary_training_curves()
    print(f"\nAll figures saved to {FIGURES}/")
