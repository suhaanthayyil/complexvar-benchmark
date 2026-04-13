"""Evaluation helpers for ComplexVar."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from complexvar.metrics.classification import (
    compute_classification_metrics,
    grouped_bootstrap,
    macro_average_by_group,
)
from complexvar.metrics.regression import compute_regression_metrics
from complexvar.utils.io import write_json


def _subset(
    frame: pd.DataFrame,
    interface_value: int | None = None,
    source_dataset: str | None = None,
    split: str = "test",
) -> pd.DataFrame:
    out = frame.copy()
    if "split" in out.columns:
        out = out[out["split"] == split].copy()
    if interface_value is not None and "interface_proximal" in out.columns:
        out = out[out["interface_proximal"].astype(int) == interface_value].copy()
    if source_dataset is not None and "source_dataset" in out.columns:
        out = out[out["source_dataset"] == source_dataset].copy()
    return out


def summarize_classification_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n_samples": 0}
    summary: dict[str, Any] = {"n_samples": int(len(frame))}
    if frame["label"].nunique() >= 2:
        summary.update(compute_classification_metrics(frame))
        if "protein_group" in frame.columns:
            summary.update(macro_average_by_group(frame, "protein_group"))
            summary["bootstrap_protein_group"] = grouped_bootstrap(
                frame,
                "protein_group",
            )
        if "family_group" in frame.columns:
            summary.update(macro_average_by_group(frame, "family_group"))
            summary["bootstrap_family_group"] = grouped_bootstrap(
                frame,
                "family_group",
            )
    return summary


def summarize_regression_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {"n_samples": 0}
    summary: dict[str, Any] = {"n_samples": int(len(frame))}
    if len(frame) >= 2:
        summary.update(
            compute_regression_metrics(
                frame,
                label_column="ddg",
                prediction_column="prediction",
            )
        )
    return summary


def evaluate_prediction_table(
    predictions_path: str | Path,
    all_metrics_output: str | Path,
    interface_metrics_output: str | Path,
) -> tuple[Path, Path]:
    frame = pd.read_csv(predictions_path, sep="\t")

    classification_frame = (
        frame.dropna(subset=["label", "score"], how="any").copy()
        if {"label", "score"}.issubset(frame.columns)
        else pd.DataFrame()
    )
    regression_frame = (
        frame.dropna(subset=["ddg", "prediction"], how="any").copy()
        if {"ddg", "prediction"}.issubset(frame.columns)
        else pd.DataFrame()
    )

    all_metrics: dict[str, Any] = {}
    if not classification_frame.empty:
        all_metrics["classification"] = summarize_classification_frame(
            _subset(classification_frame)
        )
        if "source_dataset" in classification_frame.columns:
            all_metrics["classification_by_source"] = {
                source: summarize_classification_frame(
                    _subset(classification_frame, source_dataset=source)
                )
                for source in sorted(
                    classification_frame["source_dataset"].dropna().unique()
                )
            }
    if not regression_frame.empty:
        all_metrics["regression"] = summarize_regression_frame(
            _subset(regression_frame)
        )
        if "source_dataset" in regression_frame.columns:
            all_metrics["regression_by_source"] = {
                source: summarize_regression_frame(
                    _subset(regression_frame, source_dataset=source)
                )
                for source in sorted(
                    regression_frame["source_dataset"].dropna().unique()
                )
            }

    interface_metrics: dict[str, Any] = {}
    if not classification_frame.empty:
        interface_metrics["interface_proximal"] = summarize_classification_frame(
            _subset(classification_frame, interface_value=1)
        )
        interface_metrics["interface_distal"] = summarize_classification_frame(
            _subset(classification_frame, interface_value=0)
        )
    if not regression_frame.empty:
        interface_metrics["regression_interface_proximal"] = summarize_regression_frame(
            _subset(regression_frame, interface_value=1)
        )
        interface_metrics["regression_interface_distal"] = summarize_regression_frame(
            _subset(regression_frame, interface_value=0)
        )

    return (
        write_json(all_metrics, all_metrics_output),
        write_json(interface_metrics, interface_metrics_output),
    )
