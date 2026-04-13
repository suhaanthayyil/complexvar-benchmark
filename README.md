# ComplexVar Benchmark

ComplexVar Benchmark is a reproducible computational biology repository for testing whether protein-complex structural context improves prediction of interaction-disrupting missense variants.

The benchmark is built around a matched 3-way comparison:

1. Sequence-only baseline
2. Monomer-structure graph model
3. Complex-structure graph model

Primary supervision comes from direct interaction-effect datasets:

- `SKEMPI 2.0` for experimental binding-affinity change regression and derived binary disruption labels
- `IMEx/IntAct` mutation annotations for interaction-effect classification

`ClinVar` is used as an external disease-variant application set for pathogenic versus benign analysis and VUS scoring on mapped Burke complexes.

## Scientific Scope

The central benchmark question is:

> Does protein-complex structural context improve prediction of interaction-disrupting missense variants, especially for interface-proximal residues, compared with sequence-only and monomer-only alternatives?

The core claim is restricted to interface variants. Broader pathogenicity statements are treated as secondary analyses.

## Repository Layout

```text
.
|-- configs/
|-- data/
|-- deliverables/
|-- docs/
|-- figures/
|-- results/
|-- scripts/
|-- src/complexvar/
|-- tests/
|-- workflows/
|-- Snakefile
|-- environment.yml
|-- pyproject.toml
```

## Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pip install -e ".[workflow,sequence,gnn,boost]"
```

Core optional system dependencies used by the full pipeline:

- `mmseqs`
- `mkdssp` or `dssp`

The pipeline runs on CPU, CUDA, or Apple MPS. Use `COMPLEXVAR_DEVICE=cpu` to pin torch training to CPU and `COMPLEXVAR_SEQUENCE_DEVICE=cpu` to pin ESM2 embedding to CPU.

## Real Pipeline

### 1. Download public data

```bash
scripts/download/download_all.sh \
  --datasets skempi,intact,burke,clinvar,alphafold
```

This step is idempotent and writes manifests, checksums, and logs under `data/manifests/` and `results/download/logs/`.

### 2. Normalize source datasets

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
```

### 3. Map variants to structures

```bash
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

### 4. Build graph caches

```bash
python3 scripts/preprocess/build_graph_cache.py \
  --mapping data/processed/skempi_mapped.tsv \
  --output-dir data/processed/graphs/skempi

python3 scripts/preprocess/build_graph_cache.py \
  --mapping data/processed/intact_mapped.tsv \
  --output-dir data/processed/graphs/intact
```

### 5. Build clustering tables and leakage-controlled splits

```bash
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

### 6. Train baselines and graph models

Structural baseline:

```bash
python3 scripts/train/train_baseline.py \
  --features data/processed/skempi_split_manifest.tsv \
  --target-column binary_label \
  --model-name ddg_proxy_logistic \
  --output-dir results/skempi/ddg_proxy_logistic
```

Sequence baseline:

```bash
COMPLEXVAR_DEVICE=cpu COMPLEXVAR_SEQUENCE_DEVICE=cpu \
python3 scripts/train/train_sequence.py \
  --mapping data/processed/skempi_split_manifest.tsv \
  --output-dir results/skempi/sequence_baseline
```

Complex and monomer GNNs:

```bash
python3 scripts/train/train_gnn.py \
  --manifest data/processed/skempi_graph_split_manifest.tsv \
  --output-dir results/skempi/complex_gnn

python3 scripts/train/train_gnn.py \
  --manifest data/processed/skempi_graph_split_manifest.tsv \
  --graph-column monomer_graph_path \
  --output-dir results/skempi/monomer_gnn
```

### 7. Evaluate saved prediction tables

```bash
python3 scripts/evaluate/evaluate_classification.py \
  --predictions results/skempi/ddg_proxy_logistic/predictions.tsv \
  --output results/metrics/skempi_ddg_proxy_all_metrics.json \
  --interface-output results/metrics/skempi_ddg_proxy_interface_metrics.json
```

### 8. Score ClinVar VUS

```bash
python3 scripts/preprocess/build_graph_cache.py \
  --mapping data/processed/clinvar_vus_mapped.tsv \
  --output-dir data/processed/graphs/clinvar_vus

python3 scripts/score_vus.py \
  --manifest data/processed/graphs/clinvar_vus/variant_graph_manifest.tsv \
  --checkpoint results/skempi/complex_gnn/best_model.pt \
  --output results/vus_predictions/top_vus_scored.tsv
```

### 9. Generate figures and package deliverables

```bash
python3 scripts/make_figures.py
python3 scripts/evaluate/crossval_fast.py
python3 scripts/package_deliverables.py
```

## Workflow Entry Points

If `snakemake` is installed, the repository exposes stage-level targets:

```bash
snakemake download
snakemake normalize
snakemake map_structures
snakemake graphs
snakemake splits
snakemake train_baselines
snakemake train_gnn
snakemake evaluate
snakemake figures
snakemake manuscript
snakemake deliverables
```

## Data Policy

Bulk raw downloads are not committed to git. The repository tracks:

- download scripts
- checksums and manifests
- label audits
- split files
- small derived artifacts that are legally redistributable

Large generated files over `100 MB` are intentionally hidden from git history to keep the repository cloneable on GitHub. Large generated artifacts remain reproducible from the included scripts but are kept local rather than versioned.

See [docs/data_source_verification.md](docs/data_source_verification.md) and [docs/licensing_redistribution.md](docs/licensing_redistribution.md).

## Benchmark Defaults

- Burke complex source: Burke et al. human interactome AlphaFold-complex release
- High-confidence cohort: `pDockQ > 0.5`
- Interface definition: minimum heavy-atom distance to partner chain `<= 10.0 A`
- Graph spatial edge cutoff: minimum heavy-atom distance `<= 8.0 A`
- Variant-centered subgraph radius: `12.0 A`
- Leakage control: `MMseqs2` clustering at `30%` sequence identity

## Validation and Tests

Run the repository tests with:

```bash
pytest -q
python -m complexvar.cli check-text-policy --root .
```

The test suite uses small fixture data under `tests/fixtures` and should stay fast on a laptop.

## Deliverables

The repository produces:

- dataset and split manifests
- graph caches
- baseline and GNN checkpoints
- prediction tables
- metric summaries
- cross-validation results (`results/crossval/crossval_summary.json`)
- figure files
- manuscript drafts
- final audit and review bundle

## Compute Budget

Target hardware:

- up to `8 GB` GPU VRAM
- standard desktop RAM
- roughly `30-40 GB` local storage for raw and processed assets

See [docs/compute_budget.md](docs/compute_budget.md) for the reproducibility budget template.
