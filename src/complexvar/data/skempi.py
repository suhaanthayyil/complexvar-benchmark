"""SKEMPI 2.0 normalization utilities."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from complexvar.utils.io import write_tsv

DEFAULT_TEMPERATURE_K = 298.15
GAS_CONSTANT_KCAL = 1.98720425864083e-3
SINGLE_SUBSTITUTION_RE = re.compile(
    r"^(?P<wildtype>[A-Z])(?P<chain>[A-Za-z0-9])(?P<position>-?\d+)(?P<icode>[A-Za-z]?)(?P<mutant>[A-Z])$"
)


@dataclass(frozen=True)
class SkempiMutation:
    wildtype: str
    chain_id: str
    residue_number: int
    insertion_code: str
    mutant: str


def _candidate_column(columns: list[str], options: list[str]) -> str:
    lowered = {column.lower(): column for column in columns}
    for option in options:
        if option.lower() in lowered:
            return lowered[option.lower()]
    raise ValueError(f"Could not find any of the expected columns: {options}")


def _optional_column(columns: list[str], options: list[str]) -> str | None:
    lowered = {column.lower(): column for column in columns}
    for option in options:
        if option.lower() in lowered:
            return lowered[option.lower()]
    return None


def _parse_temperature(value: object) -> float:
    if value is None or pd.isna(value):
        return DEFAULT_TEMPERATURE_K
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    if match is None:
        return DEFAULT_TEMPERATURE_K
    observed = float(match.group(1))
    if observed < 200:
        return observed
    return observed


def _compute_ddg_from_affinity(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    affinity_mut_column = _candidate_column(
        columns, ["Affinity_mut_parsed", "Affinity_mut (M)", "Affinity_mut"]
    )
    affinity_wt_column = _candidate_column(
        columns, ["Affinity_wt_parsed", "Affinity_wt (M)", "Affinity_wt"]
    )
    temperature_column = _optional_column(columns, ["Temperature"])

    affinity_mut = pd.to_numeric(frame[affinity_mut_column], errors="coerce")
    affinity_wt = pd.to_numeric(frame[affinity_wt_column], errors="coerce")
    if temperature_column is None:
        temperature = pd.Series(DEFAULT_TEMPERATURE_K, index=frame.index)
    else:
        temperature = frame[temperature_column].map(_parse_temperature)

    valid = affinity_mut.gt(0) & affinity_wt.gt(0) & temperature.gt(0)
    ddg = pd.Series(np.nan, index=frame.index, dtype=float)
    ddg.loc[valid] = (
        GAS_CONSTANT_KCAL
        * temperature.loc[valid]
        * np.log(affinity_mut.loc[valid] / affinity_wt.loc[valid])
    )
    return ddg


def parse_skempi_mutation(token: str) -> SkempiMutation | None:
    match = SINGLE_SUBSTITUTION_RE.match(str(token).strip())
    if match is None:
        return None
    return SkempiMutation(
        wildtype=match.group("wildtype"),
        chain_id=match.group("chain"),
        residue_number=int(match.group("position")),
        insertion_code=match.group("icode") or "",
        mutant=match.group("mutant"),
    )


def is_single_substitution(token: str) -> bool:
    return parse_skempi_mutation(token) is not None


def ddg_to_binary_label(value: float | None, threshold: float = 1.0) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(abs(float(value)) > threshold)


def normalize_skempi(
    frame: pd.DataFrame,
    ddg_binary_threshold: float = 1.0,
) -> pd.DataFrame:
    columns = list(frame.columns)
    pdb_column = _candidate_column(columns, ["#Pdb", "PDB", "pdb"])
    mutation_pdb_column = _optional_column(columns, ["Mutation(s)_PDB"])
    mutation_cleaned_column = _candidate_column(
        columns, ["Mutation(s)_cleaned", "Mutation(s)", "mutation"]
    )
    protein_1_column = _optional_column(columns, ["Protein 1"])
    protein_2_column = _optional_column(columns, ["Protein 2"])
    method_column = _optional_column(columns, ["Method"])
    reference_column = _optional_column(columns, ["Reference"])
    temperature_column = _optional_column(columns, ["Temperature"])
    note_column = _optional_column(columns, ["Notes"])
    version_column = _optional_column(columns, ["SKEMPI version"])
    ddg_column = _optional_column(columns, ["ddG", "ddg", "DDG"])

    work = frame.copy()
    if mutation_pdb_column is None:
        work["mutation_pdb"] = work[mutation_cleaned_column].astype(str).str.strip()
    else:
        work["mutation_pdb"] = work[mutation_pdb_column].astype(str).str.strip()
    work["mutation_cleaned"] = work[mutation_cleaned_column].astype(str).str.strip()
    work["is_single_substitution"] = work["mutation_pdb"].map(is_single_substitution)
    work = work[work["is_single_substitution"]].copy()

    parsed = work["mutation_pdb"].map(parse_skempi_mutation)
    work["wildtype"] = parsed.map(lambda item: item.wildtype if item else None)
    work["chain_id"] = parsed.map(lambda item: item.chain_id if item else None)
    work["residue_number"] = parsed.map(
        lambda item: item.residue_number if item else None
    )
    work["insertion_code"] = parsed.map(
        lambda item: item.insertion_code if item else ""
    )
    work["mutant"] = parsed.map(lambda item: item.mutant if item else None)

    if ddg_column is None:
        ddg = _compute_ddg_from_affinity(work, columns)
    else:
        ddg = pd.to_numeric(work[ddg_column], errors="coerce")

    structure_id = work[pdb_column].astype(str).str.strip()
    pdb_id = structure_id.str.extract(r"^([0-9A-Za-z]{4})", expand=False).str.upper()
    parts = structure_id.str.split("_", expand=True)
    if parts.shape[1] >= 3:
        chain_a = parts[1].astype(str)
        chain_b = parts[2].astype(str)
    else:
        chain_a = pd.Series("", index=work.index, dtype=str)
        chain_b = pd.Series("", index=work.index, dtype=str)

    out = pd.DataFrame(
        {
            "sample_id": [f"skempi_{idx:06d}" for idx in range(len(work))],
            "source_dataset": "SKEMPI",
            "structure_id": structure_id.to_numpy(),
            "pdb_id": pdb_id.to_numpy(),
            "chain_a": chain_a.to_numpy(),
            "chain_b": chain_b.to_numpy(),
            "mutated_chain_id": work["chain_id"].astype(str).to_numpy(),
            "wildtype": work["wildtype"].astype(str).to_numpy(),
            "mutant": work["mutant"].astype(str).to_numpy(),
            "residue_number": pd.to_numeric(work["residue_number"], errors="coerce")
            .fillna(-1)
            .astype(int)
            .to_numpy(),
            "insertion_code": work["insertion_code"].astype(str).to_numpy(),
            "mutation_pdb": work["mutation_pdb"].to_numpy(),
            "mutation_cleaned": work["mutation_cleaned"].to_numpy(),
            "protein_1": (
                work[protein_1_column].astype(str).to_numpy()
                if protein_1_column is not None
                else ""
            ),
            "protein_2": (
                work[protein_2_column].astype(str).to_numpy()
                if protein_2_column is not None
                else ""
            ),
            "temperature_k": (
                work[temperature_column].map(_parse_temperature).to_numpy()
                if temperature_column is not None
                else DEFAULT_TEMPERATURE_K
            ),
            "method": (
                work[method_column].astype(str).to_numpy()
                if method_column is not None
                else ""
            ),
            "reference": (
                work[reference_column].astype(str).to_numpy()
                if reference_column is not None
                else ""
            ),
            "notes": (
                work[note_column].astype(str).to_numpy()
                if note_column is not None
                else ""
            ),
            "skempi_version": (
                work[version_column].astype(str).to_numpy()
                if version_column is not None
                else ""
            ),
            "ddg": pd.to_numeric(ddg, errors="coerce").to_numpy(),
        }
    )
    out["binary_label"] = out["ddg"].map(
        lambda value: ddg_to_binary_label(value, threshold=ddg_binary_threshold)
    )
    out["regression_label_available"] = out["ddg"].map(
        lambda value: int(
            not (value is None or (isinstance(value, float) and math.isnan(value)))
        )
    )
    out["classification_label_available"] = out["binary_label"].notna().astype(int)
    return out


def write_normalized_skempi(
    input_path: str | Path,
    output: str | Path,
    ddg_binary_threshold: float = 1.0,
) -> Path:
    frame = pd.read_csv(input_path, sep=";", low_memory=False)
    normalized = normalize_skempi(frame, ddg_binary_threshold=ddg_binary_threshold)
    return write_tsv(normalized, output)
