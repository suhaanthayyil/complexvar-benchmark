# Changelog

## Unreleased

### Added

- Rebuilt repository as `ComplexVar Benchmark`
- Added reproducible package skeleton, workflow, docs, configs, tests, and
  manuscript assets
- Added sequence-only, structural baseline, and GNN model scaffolding
- Added deterministic manifest, split, and leakage-audit tooling
- Full 5-fold protein-grouped cross-validation with fresh training per fold
  (scripts/evaluate/crossval_v2_full.py)
- Interface-proximal baseline comparison across all 5 models
  (scripts/evaluate/interface_baseline_comparison.py)

### Changed

- Replaced the deleted prior project package layout with `src/complexvar`
- Repointed CI and lint configuration to the new package
- Manuscript abstract and discussion revised to reflect honest comparison
  with structural baselines
- VUS predictions moved to supplementary (unvalidated)
- Cross-validation Table S9 updated with real per-fold numbers
- Added explicit limitations section to Discussion

### Removed

- Removed internal status documents (IMPLEMENTATION_STATUS.md,
  FINAL_STATUS.md, REVIEWER_UPDATE_MESSAGE.md, CV_BUG_RESOLUTION_SUMMARY.md,
  CROSS_VALIDATION_BUG_REPORT.md)

### Known Limitations

- Full external dataset downloads are scripted but not executed in the repo by
  default
- GNN training requires optional `torch` and `torch-geometric` extras
- Some IMEx/IntAct effect-label extraction paths remain source-schema dependent
  and are documented explicitly in `docs/`
