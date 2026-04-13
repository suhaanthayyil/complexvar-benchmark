"""ClinVar normalization utilities."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from complexvar.constants import THREE_TO_ONE
from complexvar.utils.io import write_tsv

CLINVAR_POSITIVE = {
    "Pathogenic",
    "Likely pathogenic",
    "Pathogenic/Likely pathogenic",
}
CLINVAR_NEGATIVE = {"Benign", "Likely benign", "Benign/Likely benign"}
CLINVAR_EXCLUDE_CONTAINS = {
    "uncertain",
    "conflicting",
    "somatic",
    "risk factor",
    "association",
    "drug response",
}
PROTEIN_SUB_RE = re.compile(
    r"\(p\.(?P<wildtype>[A-Z][a-z]{2})(?P<position>\d+)(?P<mutant>[A-Z][a-z]{2})\)"
)
UNIPROT_OTHER_ID_RE = re.compile(r"UniProtKB:(?P<accession>[A-Z0-9]+)")


def _normalize_significance(value: str) -> str:
    return " ".join(str(value).replace("_", " ").split()).strip()


def classify_clinvar_significance(value: str) -> str | None:
    normalized = _normalize_significance(value)
    lowered = normalized.lower()
    if any(token in lowered for token in CLINVAR_EXCLUDE_CONTAINS):
        return None
    if normalized in CLINVAR_POSITIVE:
        return "positive"
    if normalized in CLINVAR_NEGATIVE:
        return "negative"
    return None


def parse_protein_substitution(value: object) -> dict[str, object] | None:
    if value is None or pd.isna(value):
        return None
    match = PROTEIN_SUB_RE.search(str(value))
    if match is None:
        return None
    wildtype = THREE_TO_ONE.get(match.group("wildtype").upper())
    mutant = THREE_TO_ONE.get(match.group("mutant").upper())
    if wildtype is None or mutant is None:
        return None
    return {
        "wildtype": wildtype,
        "mutant": mutant,
        "residue_number": int(match.group("position")),
        "protein_hgvs": match.group(0)[1:-1],
    }


def parse_uniprot_accession(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    match = UNIPROT_OTHER_ID_RE.search(str(value))
    if match is None:
        return ""
    return match.group("accession")


def load_variant_summary(
    path_or_url: str | Path,
    assembly: str = "GRCh38",
) -> pd.DataFrame:
    usecols = [
        "Type",
        "Name",
        "GeneSymbol",
        "ClinicalSignificance",
        "ReviewStatus",
        "PhenotypeList",
        "Assembly",
        "Chromosome",
        "VariationID",
        "PositionVCF",
        "ReferenceAlleleVCF",
        "AlternateAlleleVCF",
        "OtherIDs",
    ]
    chunks = []
    iterator = pd.read_csv(
        path_or_url,
        sep="\t",
        compression="infer",
        usecols=usecols,
        low_memory=False,
        chunksize=100_000,
    )
    for chunk in iterator:
        work = chunk[chunk["Assembly"] == assembly].copy()
        work["label"] = work["ClinicalSignificance"].map(classify_clinvar_significance)
        work = work[
            work["Type"].astype(str).str.lower() == "single nucleotide variant"
        ].copy()
        parsed = work["Name"].map(parse_protein_substitution)
        work = work.loc[parsed.notna()].copy()
        parsed = parsed.loc[work.index]
        work["wildtype"] = parsed.map(lambda item: item["wildtype"])
        work["mutant"] = parsed.map(lambda item: item["mutant"])
        work["residue_number"] = parsed.map(lambda item: item["residue_number"])
        work["protein_hgvs"] = parsed.map(lambda item: item["protein_hgvs"])
        work["uniprot_accession"] = work["OtherIDs"].map(parse_uniprot_accession)
        chunks.append(work)
    if not chunks:
        return pd.DataFrame(columns=usecols)
    return pd.concat(chunks, ignore_index=True)


def normalize_clinvar(
    path_or_url: str | Path, assembly: str = "GRCh38"
) -> pd.DataFrame:
    frame = load_variant_summary(path_or_url, assembly=assembly)
    out = pd.DataFrame(
        {
            "sample_id": [f"clinvar_{idx:07d}" for idx in range(len(frame))],
            "source_dataset": "ClinVar",
            "variation_id": frame["VariationID"].astype(str).to_numpy(),
            "gene_symbol": frame["GeneSymbol"].astype(str).to_numpy(),
            "uniprot_accession": frame["uniprot_accession"].astype(str).to_numpy(),
            "wildtype": frame["wildtype"].astype(str).to_numpy(),
            "mutant": frame["mutant"].astype(str).to_numpy(),
            "residue_number": pd.to_numeric(frame["residue_number"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .to_numpy(),
            "protein_hgvs": frame["protein_hgvs"].astype(str).to_numpy(),
            "clinical_significance": frame["ClinicalSignificance"]
            .astype(str)
            .to_numpy(),
            "label": frame["label"].astype(str).to_numpy(),
            "review_status": frame["ReviewStatus"].astype(str).to_numpy(),
            "condition": frame["PhenotypeList"].astype(str).to_numpy(),
            "name": frame["Name"].astype(str).to_numpy(),
            "assembly": frame["Assembly"].astype(str).to_numpy(),
            "chromosome": frame["Chromosome"].astype(str).to_numpy(),
            "position_vcf": frame["PositionVCF"].astype(str).to_numpy(),
            "reference_allele": frame["ReferenceAlleleVCF"].astype(str).to_numpy(),
            "alternate_allele": frame["AlternateAlleleVCF"].astype(str).to_numpy(),
        }
    )
    out["binary_label"] = out["label"].map({"positive": 1.0, "negative": 0.0})
    out["is_vus"] = out["binary_label"].isna().astype(int)
    return out


def write_filtered_clinvar(
    path_or_url: str | Path,
    output: str | Path,
    assembly: str = "GRCh38",
) -> Path:
    normalized = normalize_clinvar(path_or_url, assembly=assembly)
    return write_tsv(normalized, output)
