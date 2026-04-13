"""IntAct mutation normalization utilities."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from complexvar.constants import THREE_TO_ONE
from complexvar.utils.io import write_tsv

POSITIVE_EFFECTS = {"disrupting", "decreasing"}
NEGATIVE_EFFECTS = {"neutral"}
ENHANCING_EFFECTS = {"enhancing"}

SUBSTITUTION_RE = re.compile(
    r"p\.(?P<wildtype>[A-Z][a-z]{2})(?P<position>\d+)(?P<mutant>[A-Z][a-z]{2}|Ter|=)$"
)


def normalize_effect_label(value: str | None) -> str | None:
    if value is None or pd.isna(value):
        return None
    label = str(value).strip().lower()
    if "disrupt" in label or "abolish" in label:
        return "disrupting"
    if "decreas" in label or "reduced" in label or "weaken" in label:
        return "decreasing"
    if "no effect" in label or "unaffected" in label:
        return "neutral"
    if "increas" in label or "strengthen" in label:
        return "enhancing"
    if "caus" in label or "gain" in label:
        return "enhancing"
    if "undefin" in label or "conflict" in label:
        return None
    return None


def parse_protein_substitution(value: str | None) -> dict[str, object] | None:
    if value is None or pd.isna(value):
        return None
    text = str(value)
    if ";" in text or "del" in text or "ins" in text or "dup" in text or "fs" in text:
        return None
    match = SUBSTITUTION_RE.search(text)
    if match is None:
        return None
    wildtype = THREE_TO_ONE.get(match.group("wildtype").upper())
    mutant_three = match.group("mutant").upper()
    mutant = THREE_TO_ONE.get(mutant_three)
    if mutant_three == "TER":
        return None
    if mutant_three == "=":
        mutant = wildtype
    if wildtype is None or mutant is None:
        return None
    return {
        "wildtype": wildtype,
        "mutant": mutant,
        "residue_number": int(match.group("position")),
        "protein_hgvs": match.group(0),
    }


def _parse_participants(value: object, target_accession: str) -> list[str]:
    if value is None or pd.isna(value):
        return []
    accessions: list[str] = []
    for token in str(value).split("|"):
        token = token.strip().strip("()")
        if not token.startswith("uniprotkb:"):
            continue
        accession = token.split(":", maxsplit=1)[1].split("(", maxsplit=1)[0]
        if accession != target_accession and accession not in accessions:
            accessions.append(accession)
    return accessions


def effect_to_multiclass(value: str | None) -> int | None:
    if value in POSITIVE_EFFECTS:
        return 1
    if value in NEGATIVE_EFFECTS:
        return 0
    if value in ENHANCING_EFFECTS:
        return 2
    return None


def effect_to_binary_label(value: str | None) -> float | None:
    if value in POSITIVE_EFFECTS:
        return 1.0
    if value in NEGATIVE_EFFECTS:
        return 0.0
    return None


def normalize_intact(frame: pd.DataFrame) -> pd.DataFrame:
    columns = {column.lower(): column for column in frame.columns}
    effect_column = columns.get("feature annotation")
    if effect_column is None:
        effect_column = columns.get("feature annotation(s)")
    if effect_column is None:
        raise ValueError("IntAct mutation export is missing Feature annotation.")

    required = {
        "feature": columns.get("feature short label"),
        "protein": columns.get("affected protein ac")
        or columns.get("affected molecule identifier"),
        "interaction": columns.get("interaction ac"),
    }
    missing = [name for name, column in required.items() if column is None]
    if missing:
        raise ValueError(f"Missing required IntAct columns: {missing}")

    symbol_column = columns.get("affected protein symbol") or columns.get(
        "affected molecule symbol"
    )
    name_column = columns.get("affected protein full name") or columns.get(
        "affected molecule full name"
    )
    organism_column = columns.get("affected protein organism") or columns.get(
        "affected molecule organism"
    )
    participants_column = columns.get("interaction participants")
    pubmed_column = columns.get("pubmedid")
    range_column = columns.get("feature range(s)")
    feature_type_column = columns.get("feature type")
    accession_column = columns.get("#feature ac", columns.get("feature ac"))

    work = frame.copy()
    work["protein_accession"] = (
        work[required["protein"]].astype(str).str.replace("uniprotkb:", "", regex=False)
    )
    parsed = work[required["feature"]].map(parse_protein_substitution)
    work["parsed_mutation"] = parsed
    work = work[work["parsed_mutation"].notna()].copy()
    parsed = work["parsed_mutation"]

    feature_type_series = (
        work[feature_type_column].astype(str)
        if feature_type_column is not None
        else pd.Series("", index=work.index, dtype=str)
    )
    raw_effect_series = work[effect_column].astype(str)
    combined_effect_series = feature_type_series.str.cat(
        raw_effect_series,
        sep=" | ",
    )

    out = pd.DataFrame(
        {
            "sample_id": [f"intact_{idx:07d}" for idx in range(len(work))],
            "source_dataset": "IntAct",
            "feature_accession": (
                work[accession_column].astype(str).to_numpy()
                if accession_column is not None
                else ""
            ),
            "interaction_accession": work[required["interaction"]]
            .astype(str)
            .to_numpy(),
            "protein_accession": work["protein_accession"].astype(str).to_numpy(),
            "protein_symbol": (
                work[symbol_column].astype(str).to_numpy()
                if symbol_column is not None
                else ""
            ),
            "protein_name": (
                work[name_column].astype(str).to_numpy()
                if name_column is not None
                else ""
            ),
            "organism": (
                work[organism_column].astype(str).to_numpy()
                if organism_column is not None
                else ""
            ),
            "protein_hgvs": parsed.map(lambda item: item["protein_hgvs"]).to_numpy(),
            "wildtype": parsed.map(lambda item: item["wildtype"]).to_numpy(),
            "mutant": parsed.map(lambda item: item["mutant"]).to_numpy(),
            "residue_number": parsed.map(lambda item: item["residue_number"])
            .astype(int)
            .to_numpy(),
            "feature_range": (
                work[range_column].astype(str).to_numpy()
                if range_column is not None
                else ""
            ),
            "raw_effect_text": combined_effect_series.to_numpy(),
            "feature_type": feature_type_series.to_numpy(),
            "pubmed_id": (
                work[pubmed_column].astype(str).to_numpy()
                if pubmed_column is not None
                else ""
            ),
        }
    )
    out["effect_label"] = out["raw_effect_text"].map(normalize_effect_label)
    out["binary_label"] = out["effect_label"].map(effect_to_binary_label)
    out["multiclass_label"] = out["effect_label"].map(effect_to_multiclass)
    out["classification_label_available"] = out["binary_label"].notna().astype(int)
    out["partner_accessions"] = [
        ";".join(_parse_participants(value, accession))
        for value, accession in zip(
            (
                work[participants_column]
                if participants_column is not None
                else [None] * len(work)
            ),
            out["protein_accession"],
            strict=False,
        )
    ]
    return out.reset_index(drop=True)


def write_normalized_intact(input_path: str | Path, output: str | Path) -> Path:
    frame = pd.read_csv(
        input_path,
        sep="\t",
        on_bad_lines="skip",
        engine="python",
    )
    normalized = normalize_intact(frame)
    return write_tsv(normalized, output)
