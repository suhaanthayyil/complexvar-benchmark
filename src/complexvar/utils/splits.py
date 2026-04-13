"""Split and clustering helpers."""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from complexvar.constants import DEFAULT_RANDOM_SEED


@dataclass(frozen=True)
class SplitFractions:
    train: float = 0.70
    val: float = 0.15
    test: float = 0.15


def make_group_splits(
    frame: pd.DataFrame,
    group_column: str,
    fractions: SplitFractions | None = None,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    fractions = fractions or SplitFractions()
    if not np.isclose(fractions.train + fractions.val + fractions.test, 1.0):
        raise ValueError("Split fractions must sum to 1.0")

    groups = sorted(frame[group_column].astype(str).unique().tolist())
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n_groups = len(groups)
    train_cut = int(round(n_groups * fractions.train))
    val_cut = train_cut + int(round(n_groups * fractions.val))

    assignment = {}
    for index, group in enumerate(groups):
        if index < train_cut:
            assignment[group] = "train"
        elif index < val_cut:
            assignment[group] = "val"
        else:
            assignment[group] = "test"

    out = frame.copy()
    out["split"] = out[group_column].astype(str).map(assignment)
    return out


def leakage_summary(
    frame: pd.DataFrame, split_column: str, fields: Iterable[str]
) -> dict:
    summary: dict[str, dict[str, int]] = {}
    for split, split_frame in frame.groupby(split_column):
        summary[str(split)] = {
            field: int(split_frame[field].astype(str).nunique())
            for field in fields
            if field in split_frame.columns
        }
        summary[str(split)]["samples"] = int(len(split_frame))
    return summary


def assign_identity_clusters(
    proteins: pd.DataFrame,
    identity_threshold: float = 0.30,
) -> pd.DataFrame:
    required = {"accession", "sequence"}
    if not required.issubset(proteins.columns):
        missing = sorted(required.difference(proteins.columns))
        raise ValueError(f"Missing protein clustering columns: {missing}")

    proteins = (
        proteins.dropna(subset=["accession", "sequence"]).drop_duplicates().copy()
    )
    mmseqs = shutil.which("mmseqs")
    if mmseqs is None or proteins.empty:
        proteins["cluster_id"] = proteins["accession"].astype(str)
        return proteins[["accession", "cluster_id"]]

    with tempfile.TemporaryDirectory(prefix="complexvar_mmseqs_") as tmp_dir:
        tmp_dir = Path(tmp_dir)
        fasta_path = tmp_dir / "proteins.fasta"
        cluster_prefix = tmp_dir / "clusters"
        result_tsv = tmp_dir / "clusters_cluster.tsv"
        fasta_lines = []
        for row in proteins.itertuples(index=False):
            fasta_lines.append(f">{row.accession}")
            fasta_lines.append(str(row.sequence))
        fasta_path.write_text("\n".join(fasta_lines) + "\n", encoding="utf-8")
        subprocess.run(
            [
                mmseqs,
                "easy-cluster",
                str(fasta_path),
                str(cluster_prefix),
                str(tmp_dir / "tmp"),
                "--min-seq-id",
                str(identity_threshold),
                "-c",
                "0.8",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        clusters = pd.read_csv(
            result_tsv,
            sep="\t",
            header=None,
            names=["cluster_id", "accession"],
        )
        return clusters[["accession", "cluster_id"]].drop_duplicates()


def attach_clusters(
    samples: pd.DataFrame,
    clusters: pd.DataFrame,
    protein_column: str = "protein_accession",
    partner_column: str = "partner_accession",
) -> pd.DataFrame:
    cluster_lookup = dict(
        zip(clusters["accession"], clusters["cluster_id"], strict=False)
    )
    out = samples.copy()
    out["protein_cluster"] = (
        out[protein_column]
        .astype(str)
        .map(cluster_lookup)
        .fillna(out[protein_column].astype(str))
    )
    if partner_column in out.columns:
        out["partner_cluster"] = (
            out[partner_column]
            .astype(str)
            .map(cluster_lookup)
            .fillna(out[partner_column].astype(str))
        )
    else:
        out["partner_cluster"] = ""
    out["family_group"] = out.apply(
        lambda row: "::".join(
            sorted(
                [
                    str(row["protein_cluster"]),
                    str(row["partner_cluster"]),
                ]
            )
        ),
        axis=1,
    )
    return out
