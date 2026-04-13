"""Biological interpretation utilities."""

from __future__ import annotations

import pandas as pd


def rank_disruptive_variants(
    frame: pd.DataFrame, score_column: str = "score"
) -> pd.DataFrame:
    ranked = frame.sort_values(score_column, ascending=False).reset_index(drop=True)
    ranked["rank"] = range(1, len(ranked) + 1)
    return ranked


def summarize_gene_classes(
    frame: pd.DataFrame, gene_class_column: str = "gene_class"
) -> pd.DataFrame:
    if gene_class_column not in frame.columns:
        return pd.DataFrame(columns=[gene_class_column, "count"])
    summary = frame.groupby(gene_class_column).size().reset_index(name="count")
    return summary.sort_values("count", ascending=False).reset_index(drop=True)
