# Data Source Verification

## Burke et al. human interactome complexes

- Paper: Burke et al., *Nature Structural and Molecular Biology* (2023)
- Paper DOI: `10.1038/s41594-022-00910-8`
- Reported scale: 65,484 predicted human PPIs and 3,137 high-confidence models
- Structured release:
  - summary tables at `https://archive.bioinfo.se/huintaf2/`
  - model archive via Figshare article `16945039`
- License verified from Figshare API: `CC BY 4.0`

## ClinVar

- Source: NCBI FTP
- Primary files:
  - `variant_summary.txt.gz`
  - `clinvar.vcf.gz`
- Access route verified:
  - `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz`
  - `https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz`
- Notes:
  - VCF is smaller and easier for variant-level lookup
  - `variant_summary.txt.gz` is preferred for review status and high-level label
    filtering

## SKEMPI 2.0

- Source portal: `https://life.bsc.es/pid/skempi2`
- Intended use:
  - regression on `ddG`
  - derived binary disruption task
- The scripted downloader stores the portal route and requires a direct dataset
  URL configured at runtime if the portal changes.

## IMEx / IntAct mutations

- FTP route verified:
  - `https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv`
- Current open export is accessible and redistributable enough for download
  scripting.
- Limitation:
  - the currently observed flat file schema does not always expose a clean
    explicit effect label in the first-line preview
  - implementation therefore supports a schema-aware extraction step and raises
    a documented blocker if effect labels are absent from the chosen export

## Supporting sources

- Reactome for pathway enrichment
- STRING for network annotation
- AlphaMissense as a public score lookup baseline when legally and technically
  available for redistribution or scripted retrieval
