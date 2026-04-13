# Changelog

## Unreleased

### Added

- Rebuilt repository as `ComplexVar Benchmark`
- Added reproducible package skeleton, workflow, docs, configs, tests, and
  manuscript assets
- Added sequence-only, structural baseline, and GNN model scaffolding
- Added deterministic manifest, split, and leakage-audit tooling
- Added GitHub workflows, issue templates, pull request template, and
  contribution guide
- Added packaged `deliverables/` handoff bundle for downstream paper drafting

### Changed

- Replaced the deleted prior project package layout with `src/complexvar`
- Repointed CI and lint configuration to the new package

### Known Limitations

- Full external dataset downloads are scripted but not executed in the repo by
  default
- GNN training requires optional `torch` and `torch-geometric` extras
- Some IMEx/IntAct effect-label extraction paths remain source-schema dependent
  and are documented explicitly in `docs/`
