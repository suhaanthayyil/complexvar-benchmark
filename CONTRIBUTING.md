# Contributing

## Development setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[workflow,boost]"
pip install -e ".[sequence,gnn]"
```

## Required checks

Run these before opening a pull request:

```bash
python -m ruff check src scripts tests
pytest
```

If your change affects the workflow or benchmark outputs, also run:

```bash
python -m complexvar.cli make-toy-dataset --output-dir data/processed/toy
python -m complexvar.cli write-toy-manifests --input-dir data/processed/toy --output-dir data/manifests
python -m complexvar.cli build-toy-graphs --input-dir data/processed/toy --output data/processed/toy/graph_manifest.tsv
python -m complexvar.cli train-baseline --features data/processed/toy/classification_features.tsv --splits data/manifests/toy_split_manifest.tsv --target-column label --group-column protein_group --output-dir results/toy/baseline
python -m complexvar.cli evaluate-classification --predictions results/toy/baseline/predictions.tsv --output results/toy/baseline/metrics.json
```

## Repository standards

- Keep claims narrow and evidence-backed.
- Do not commit bulk raw mirrors unless licensing has been reviewed.
- Preserve fixed split files and leakage checks.
- Document blockers and skipped steps explicitly.
- Prefer small, reviewable pull requests.
