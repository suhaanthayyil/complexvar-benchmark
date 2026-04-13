# Reproduction Guide

## Quick Validation

```bash
pip install -e ".[dev]"
pytest -q
python -m complexvar.cli check-text-policy --root .
```

## Full Benchmark Reproduction

### 1. Install the environment

```bash
conda env create -f environment.yml
conda activate complexvar
pip install -e ".[workflow,sequence,gnn,boost]"
```

### 2. Download public datasets

```bash
scripts/download/download_all.sh \
  --datasets skempi,intact,burke,clinvar,alphafold
```

### 3. Normalize and map datasets

```bash
python3 scripts/preprocess/normalize_skempi.py \
  --input data/raw/skempi/skempi_v2.csv \
  --output data/processed/skempi_normalized.tsv

python3 scripts/preprocess/normalize_intact.py \
  --input data/raw/intact/mutations/intact_feature_mutations.tsv \
  --output data/processed/intact_normalized.tsv

python3 scripts/preprocess/filter_clinvar.py \
  --input data/raw/clinvar/variant_summary.txt.gz \
  --output data/processed/clinvar_normalized.tsv

python3 scripts/preprocess/build_structure_manifest.py \
  --summary-csv data/raw/burke/table_AF2_HURI_HuMap_UNIQUE.csv \
  --structure-root data/raw/burke/pdbs/high_confidence \
  --output data/processed/burke_structure_manifest.tsv

python3 scripts/preprocess/map_variants_to_structures.py \
  --source skempi \
  --variants data/processed/skempi_normalized.tsv \
  --pdb-dir data/raw/skempi/pdbs \
  --output data/processed/skempi_mapped.tsv

python3 scripts/preprocess/map_variants_to_structures.py \
  --source intact \
  --variants data/processed/intact_normalized.tsv \
  --burke-manifest data/processed/burke_structure_manifest.tsv \
  --burke-structure-root data/raw/burke/pdbs/high_confidence \
  --output data/processed/intact_mapped.tsv

python3 scripts/preprocess/map_variants_to_structures.py \
  --source clinvar \
  --variants data/processed/clinvar_normalized.tsv \
  --burke-manifest data/processed/burke_structure_manifest.tsv \
  --burke-structure-root data/raw/burke/pdbs/high_confidence \
  --burke-summary-csv data/raw/burke/table_AF2_HURI_HuMap_UNIQUE.csv \
  --output data/processed/clinvar_mapped.tsv
```

### 4. Build graphs and split manifests

```bash
python3 scripts/preprocess/build_graph_cache.py \
  --mapping data/processed/skempi_mapped.tsv \
  --output-dir data/processed/graphs/skempi

python3 scripts/preprocess/build_graph_cache.py \
  --mapping data/processed/intact_mapped.tsv \
  --output-dir data/processed/graphs/intact

python3 scripts/preprocess/build_protein_table.py \
  --inputs data/processed/skempi_mapped.tsv data/processed/intact_mapped.tsv \
  --monomer-root data/raw/alphafold_monomers/pdb \
  --output data/processed/proteins.tsv

python3 scripts/preprocess/build_split_manifest.py \
  --input data/processed/combined_mapped.tsv \
  --proteins data/processed/proteins.tsv \
  --output data/processed/combined_split_manifest.tsv \
  --audit-output data/processed/combined_split_audit.json
```

### 5. Train models

```bash
python3 scripts/train/train_baseline.py \
  --features data/processed/skempi_split_manifest.tsv \
  --target-column binary_label \
  --model-name ddg_proxy_logistic \
  --output-dir results/skempi/ddg_proxy_logistic

COMPLEXVAR_DEVICE=cpu COMPLEXVAR_SEQUENCE_DEVICE=cpu \
python3 scripts/train/train_sequence.py \
  --mapping data/processed/skempi_split_manifest.tsv \
  --output-dir results/skempi/sequence_baseline

python3 scripts/train/train_gnn.py \
  --manifest data/processed/skempi_graph_split_manifest.tsv \
  --output-dir results/skempi/complex_gnn

python3 scripts/train/train_gnn.py \
  --manifest data/processed/skempi_graph_split_manifest.tsv \
  --graph-column monomer_graph_path \
  --output-dir results/skempi/monomer_gnn
```

### 6. Evaluate and package

```bash
python3 scripts/evaluate/evaluate_classification.py \
  --predictions results/skempi/complex_gnn/predictions.tsv \
  --output results/metrics/all_metrics.json \
  --interface-output results/metrics/interface_stratified_metrics.json
```

## Determinism

- Fixed seeds live in `configs/`
- Split manifests are written once and versioned
- Download manifests and checksums are stored under `data/manifests/`
- The text policy gate is enforced in CI
