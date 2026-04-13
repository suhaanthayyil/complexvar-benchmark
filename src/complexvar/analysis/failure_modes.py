"""Failure-mode analysis utilities."""

from __future__ import annotations

import pandas as pd


def high_confidence_errors(
    frame: pd.DataFrame,
    label_column: str = "label",
    score_column: str = "score",
    threshold: float = 0.8,
) -> pd.DataFrame:
    predicted = (frame[score_column] >= 0.5).astype(int)
    wrong = predicted != frame[label_column].astype(int)
    confident = frame[score_column].where(
        frame[score_column] >= 0.5, 1.0 - frame[score_column]
    )
    out = frame[wrong & (confident >= threshold)].copy()
    out["confidence"] = confident[wrong & (confident >= threshold)]
    return out.sort_values("confidence", ascending=False)


def compare_complex_vs_monomer(
    complex_frame: pd.DataFrame,
    monomer_frame: pd.DataFrame,
    join_columns: list[str] | None = None,
) -> pd.DataFrame:
    join_columns = join_columns or ["sample_id"]
    merged = complex_frame.merge(
        monomer_frame[join_columns + ["score"]],
        on=join_columns,
        suffixes=("_complex", "_monomer"),
    )
    merged["delta_score"] = merged["score_complex"] - merged["score_monomer"]
    return merged.sort_values("delta_score", ascending=False)
