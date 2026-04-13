#!/usr/bin/env python3
"""Package full benchmark deliverables for professor review."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(
        source,
        destination,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo", ".DS_Store"),
    )


def _copy_if_exists(source: Path, destination: Path) -> bool:
    if source.exists():
        _copy_file(source, destination)
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="deliverables")
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- Manifests --
    manifest_dir = output_dir / "manifests"
    manifest_sources = [
        "data/manifests/checksums.md5",
        "data/manifests/download_manifest.tsv",
        "data/manifests/download_summary.json",
        "data/manifests/skempi_pdb_manifest.tsv",
        "data/manifests/intact_download_manifest.tsv",
        "data/manifests/burke_files_manifest.tsv",
        "data/manifests/burke_high_confidence_complexes.tsv",
        "data/manifests/clinvar_download_manifest.tsv",
        "data/manifests/alphafold_monomer_manifest.tsv",
    ]
    for rel in manifest_sources:
        source = repo_root / rel
        if source.exists():
            _copy_file(source, manifest_dir / source.name)

    # -- Split files --
    splits_dir = output_dir / "splits"
    for name in [
        "skempi_split_manifest.tsv",
        "skempi_graph_split_manifest.tsv",
        "combined_split_manifest.tsv",
        "skempi_split_audit.json",
        "combined_split_audit.json",
    ]:
        _copy_if_exists(repo_root / "data/processed" / name, splits_dir / name)

    # -- Metrics --
    metrics_dir = output_dir / "metrics"
    source_metrics = repo_root / "results/metrics"
    if source_metrics.exists():
        for json_file in sorted(source_metrics.glob("*.json")):
            _copy_file(json_file, metrics_dir / json_file.name)

    # -- Figures --
    figures_dir = output_dir / "figures"
    source_figures = repo_root / "results/figures"
    if source_figures.exists():
        for fig_file in sorted(source_figures.glob("*")):
            if fig_file.suffix in {".pdf", ".png"}:
                _copy_file(fig_file, figures_dir / fig_file.name)

    # -- VUS predictions --
    vus_dir = output_dir / "vus_predictions"
    _copy_if_exists(
        repo_root / "results/vus_predictions/top_vus_scored.tsv",
        vus_dir / "top_vus_scored.tsv",
    )

    # -- Manuscript drafts --
    manuscript_dir = output_dir / "manuscript"
    for name in ["main.md", "supplementary.md"]:
        _copy_if_exists(
            repo_root / "docs/manuscript" / name, manuscript_dir / name
        )

    # -- Ablation results --
    ablation_dir = output_dir / "ablation_results"
    _copy_if_exists(
        repo_root / "results/ablations/ablation_results.json",
        ablation_dir / "ablation_results.json",
    )

    # -- Significance results --
    sig_dir = output_dir / "significance_results"
    _copy_if_exists(
        repo_root / "results/significance/significance_results.json",
        sig_dir / "significance_results.json",
    )

    # -- Stratified results --
    strat_dir = output_dir / "stratified_results"
    _copy_if_exists(
        repo_root / "results/stratified/stratified_results.json",
        strat_dir / "stratified_results.json",
    )

    # -- Hybrid model results --
    _copy_if_exists(
        repo_root / "results/skempi/hybrid/hybrid_results.json",
        output_dir / "hybrid_results" / "hybrid_results.json",
    )

    # -- Model training metadata --
    models_dir = output_dir / "model_metadata"
    for model_name in [
        "sequence_baseline",
        "monomer_gnn",
        "complex_gnn",
        "ddg_proxy_logistic",
        "struct_hgb",
    ]:
        model_src = repo_root / "results/skempi" / model_name
        if model_src.exists():
            dest = models_dir / model_name
            for fname in ["training_metadata.json", "training_log.tsv"]:
                _copy_if_exists(model_src / fname, dest / fname)

    # -- Download logs --
    logs_source = repo_root / "results/download/logs"
    if logs_source.exists():
        _copy_tree(logs_source, output_dir / "download_logs")

    # -- Final audit --
    _write_final_audit(repo_root, output_dir)

    # -- README --
    _write_readme(output_dir)

    print(f"Deliverables packaged to {output_dir}")


def _write_final_audit(repo_root: Path, output_dir: Path) -> None:
    lines = [
        "# ComplexVar Benchmark Final Audit",
        "",
        "## Pipeline completion status",
        "",
    ]

    checks = [
        ("Data download", "data/manifests/download_manifest.tsv"),
        ("SKEMPI normalization", "data/processed/skempi_normalized.tsv"),
        ("IntAct normalization", "data/processed/intact_normalized.tsv"),
        ("ClinVar normalization", "data/processed/clinvar_normalized.tsv"),
        ("SKEMPI structure mapping", "data/processed/skempi_mapped.tsv"),
        ("IntAct structure mapping", "data/processed/intact_mapped.tsv"),
        ("ClinVar structure mapping", "data/processed/clinvar_mapped.tsv"),
        ("SKEMPI graph cache", "data/processed/graphs/skempi/graph_cache_summary.json"),
        ("IntAct graph cache", "data/processed/graphs/intact/graph_cache_summary.json"),
        ("ClinVar VUS graph cache", "data/processed/graphs/clinvar_vus/graph_cache_summary.json"),
        ("Protein clustering", "data/processed/protein_clusters.tsv"),
        ("Split manifest", "data/processed/skempi_graph_split_manifest.tsv"),
        ("Sequence baseline", "results/skempi/sequence_baseline/best_model.pt"),
        ("Monomer GNN", "results/skempi/monomer_gnn_v2/best_model.pt"),
        ("Complex GNN", "results/skempi/complex_gnn_v2/best_model.pt"),
        ("Structural baselines", "results/skempi/ddg_proxy_logistic/model.joblib"),
        ("Classification metrics", "results/metrics/skempi_complex_all_metrics.json"),
        ("Regression metrics", "results/metrics/skempi_complex_regression_metrics.json"),
        ("VUS predictions", "results/vus_predictions/top_vus_scored.tsv"),
        ("Figure 1 (ROC/PR)", "results/figures/fig1_roc_pr_curves.pdf"),
        ("Figure 2 (regression)", "results/figures/fig2_skempi_regression.pdf"),
        ("Figure 3 (VUS)", "results/figures/fig3_vus_reclassification.pdf"),
        ("Figure 4 (comparison)", "results/figures/fig4_model_comparison.pdf"),
        ("Figure 5 (ablation)", "results/figures/fig5_ablation_study.pdf"),
        ("Figure 6 (significance)", "results/figures/fig6_significance.pdf"),
        ("Ablation results", "results/ablations/ablation_results.json"),
        ("Significance results", "results/significance/significance_results.json"),
        ("Hybrid model results", "results/skempi/hybrid/hybrid_results.json"),
        ("Manuscript draft", "docs/manuscript/main.md"),
        ("Supplementary", "docs/manuscript/supplementary.md"),
    ]

    for label, rel_path in checks:
        exists = (repo_root / rel_path).exists()
        status = "PASS" if exists else "MISSING"
        lines.append(f"- [{status}] {label}: {rel_path}")

    lines.extend([
        "",
        "## Supported claims",
        "",
        "- Complex GNN outperforms monomer GNN and sequence baseline on "
        "SKEMPI classification (AUROC, AUPRC, MCC).",
        "- The advantage is most pronounced on interface-proximal variants.",
        "- Regression performance follows the same ordering "
        "(complex > monomer > sequence by Spearman correlation).",
        "- 277 ClinVar VUS were scored at protein-protein interfaces.",
        "",
        "## Open risks",
        "",
        "- The improvement from monomer to complex is modest (AUROC 0.747 "
        "to 0.754 overall, 0.714 to 0.722 interface-proximal).",
        "- Strong structural baselines (logistic, HGB) achieve comparable "
        "or higher AUROC (0.788, 0.792), which limits the GNN-specific "
        "contribution claim.",
        "- VUS predictions are unvalidated computationally-derived "
        "prioritization scores.",
        "- The benchmark uses a single split; cross-validation would "
        "strengthen confidence.",
    ])

    (output_dir / "final_audit.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _write_readme(output_dir: Path) -> None:
    readme = "\n".join([
        "# ComplexVar Benchmark Deliverables",
        "",
        "This folder contains all outputs from the ComplexVar benchmark pipeline.",
        "",
        "## Contents",
        "",
        "- manifests/ -- data acquisition and structure manifests",
        "- splits/ -- train/val/test split assignments with leakage audit",
        "- metrics/ -- all classification and regression metric JSON files",
        "- figures/ -- publication-ready figures (PDF and PNG)",
        "- vus_predictions/ -- scored ClinVar VUS variants",
        "- ablation_results/ -- ablation study JSON results",
        "- significance_results/ -- statistical significance tests",
        "- stratified_results/ -- stratified evaluation by subset",
        "- hybrid_results/ -- hybrid model (GNN + tabular) results",
        "- manuscript/ -- full manuscript and supplementary drafts",
        "- model_metadata/ -- training logs and hyperparameters",
        "- download_logs/ -- data acquisition logs",
        "- final_audit.md -- pipeline completion audit",
    ])
    (output_dir / "README.md").write_text(readme + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
