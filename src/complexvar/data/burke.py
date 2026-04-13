"""Burke complex manifest helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from complexvar.utils.io import write_json, write_tsv

BURKE_FIGSHARE_API = "https://api.figshare.com/v2/articles/16945039"
BURKE_SUMMARY_URL = (
    "https://archive.bioinfo.se/huintaf2/table_AF2_HURI_HuMap_UNIQUE.csv"
)


def fetch_burke_download_manifest() -> dict:
    response = requests.get(BURKE_FIGSHARE_API, timeout=60)
    response.raise_for_status()
    payload = response.json()
    return {
        "source": "burke_figshare",
        "doi": payload.get("doi"),
        "license": payload.get("license", {}).get("name"),
        "files": payload.get("files", []),
    }


def write_download_manifest(output: str | Path) -> Path:
    manifest = {
        "burke": fetch_burke_download_manifest(),
        "burke_summary_url": BURKE_SUMMARY_URL,
        "clinvar_variant_summary_url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz",
        "clinvar_vcf_url": "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz",
        "intact_mutations_url": "https://ftp.ebi.ac.uk/pub/databases/intact/current/various/mutations.tsv",
        "skempi_url": "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv",
    }
    return write_json(manifest, output)


def burke_file_urls(kind: str) -> dict[str, str]:
    payload = fetch_burke_download_manifest()
    urls: dict[str, str] = {}
    for item in payload["files"]:
        name = item["name"]
        if kind == "complex" and name in {"HuRI.zip", "humap.zip"}:
            urls[name] = item["download_url"]
        if kind == "monomer" and name in {"HuRI-single.zip", "HuMap-single.zip"}:
            urls[name] = item["download_url"]
    if not urls:
        raise ValueError(f"No Burke file URLs found for kind={kind!r}")
    return urls


def build_structure_manifest(
    summary_csv: str | Path,
    output: str | Path,
    pdockq_threshold: float = 0.5,
    structure_root: str | Path | None = None,
) -> Path:
    frame = pd.read_csv(summary_csv)
    required = [
        "unique_ID",
        "id1",
        "id2",
        "pDockQ",
        "Dataset",
        "structure_file",
    ]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required Burke columns: {missing}")

    out = frame[required].copy()
    out["Gen.id1"] = frame["Gen.id1"] if "Gen.id1" in frame.columns else frame["id1"]
    out["Gen.id2"] = frame["Gen.id2"] if "Gen.id2" in frame.columns else frame["id2"]
    out["pDockQ"] = pd.to_numeric(out["pDockQ"], errors="coerce")
    out["is_high_confidence"] = out["pDockQ"] > pdockq_threshold
    out = out.rename(
        columns={
            "unique_ID": "complex_id",
            "id1": "protein_a",
            "id2": "protein_b",
            "Gen.id1": "gene_a",
            "Gen.id2": "gene_b",
            "pDockQ": "pdockq",
            "Dataset": "source_dataset",
        }
    )
    if structure_root is not None:
        structure_root = Path(structure_root)
        out["structure_path"] = out["structure_file"].map(
            lambda name: str((structure_root / str(name)).resolve())
        )
        out["structure_exists"] = out["structure_path"].map(
            lambda value: int(Path(value).exists())
        )
    else:
        out["structure_path"] = ""
        out["structure_exists"] = 0
    return write_tsv(out, output)


def build_gene_accession_table(summary_csv: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(summary_csv)
    required = ["id1", "id2", "Gen.id1", "Gen.id2"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing Burke gene mapping columns: {missing}")
    gene_map = pd.concat(
        [
            frame[["Gen.id1", "id1"]].rename(
                columns={"Gen.id1": "gene_symbol", "id1": "accession"}
            ),
            frame[["Gen.id2", "id2"]].rename(
                columns={"Gen.id2": "gene_symbol", "id2": "accession"}
            ),
        ],
        ignore_index=True,
    ).dropna()
    gene_map["gene_symbol"] = gene_map["gene_symbol"].astype(str)
    gene_map["accession"] = gene_map["accession"].astype(str)
    return gene_map.drop_duplicates().reset_index(drop=True)
