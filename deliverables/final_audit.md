# ComplexVar Benchmark Final Audit

## Pipeline completion status

- [PASS] Data download: data/manifests/download_manifest.tsv
- [PASS] SKEMPI normalization: data/processed/skempi_normalized.tsv
- [PASS] IntAct normalization: data/processed/intact_normalized.tsv
- [PASS] ClinVar normalization: data/processed/clinvar_normalized.tsv
- [PASS] SKEMPI structure mapping: data/processed/skempi_mapped.tsv
- [PASS] IntAct structure mapping: data/processed/intact_mapped.tsv
- [PASS] ClinVar structure mapping: data/processed/clinvar_mapped.tsv
- [PASS] SKEMPI graph cache: data/processed/graphs/skempi/graph_cache_summary.json
- [PASS] IntAct graph cache: data/processed/graphs/intact/graph_cache_summary.json
- [PASS] ClinVar VUS graph cache: data/processed/graphs/clinvar_vus/graph_cache_summary.json
- [PASS] Protein clustering: data/processed/protein_clusters.tsv
- [PASS] Split manifest: data/processed/skempi_graph_split_manifest.tsv
- [PASS] Sequence baseline: results/skempi/sequence_baseline/best_model.pt
- [PASS] Monomer GNN: results/skempi/monomer_gnn_v2/best_model.pt
- [PASS] Complex GNN: results/skempi/complex_gnn_v2/best_model.pt
- [PASS] Structural baselines: results/skempi/ddg_proxy_logistic/model.joblib
- [PASS] Classification metrics: results/metrics/skempi_complex_all_metrics.json
- [PASS] Regression metrics: results/metrics/skempi_complex_regression_metrics.json
- [PASS] VUS predictions: results/vus_predictions/top_vus_scored.tsv
- [PASS] Figure 1 (ROC/PR): results/figures/fig1_roc_pr_curves.pdf
- [PASS] Figure 2 (regression): results/figures/fig2_skempi_regression.pdf
- [PASS] Figure 3 (VUS): results/figures/fig3_vus_reclassification.pdf
- [PASS] Figure 4 (comparison): results/figures/fig4_model_comparison.pdf
- [PASS] Figure 5 (ablation): results/figures/fig5_ablation_study.pdf
- [PASS] Figure 6 (significance): results/figures/fig6_significance.pdf
- [PASS] Ablation results: results/ablations/ablation_results.json
- [PASS] Significance results: results/significance/significance_results.json
- [PASS] Hybrid model results: results/skempi/hybrid/hybrid_results.json
- [FAIL] Cross-validation: Dimension mismatch bug identified (see CROSS_VALIDATION_BUG_REPORT.md)
- [PASS] Manuscript draft: docs/manuscript/main.md
- [PASS] Supplementary: docs/manuscript/supplementary.md

## Supported claims

- Complex GNN outperforms monomer GNN and sequence baseline on SKEMPI classification on the held-out test set (AUROC, AUPRC, MCC).
- The advantage is most pronounced on interface-proximal variants (delta AUROC +0.020, p = 0.019).
- **Cross-validation:** A dimension mismatch bug was identified where models trained on 28-node, 3-edge graphs were evaluated on regenerated 36-node, 8-edge graphs. The originally reported 0.511 CV AUROC is artifactual and has been retracted. The held-out test result (0.754) evaluated before graph regeneration remains valid.
- 277 ClinVar VUS were scored at protein-protein interfaces.

## Open risks

- The improvement from monomer to complex is modest (AUROC 0.747 to 0.754 overall, 0.714 to 0.722 interface-proximal).
- Strong structural baselines (logistic, HGB) achieve comparable or higher AUROC (0.788, 0.792), which limits the GNN-specific contribution claim.
- VUS predictions are unvalidated computationally-derived prioritization scores.
