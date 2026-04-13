"""Classification metrics and grouped evaluation."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from complexvar.constants import DEFAULT_RANDOM_SEED


def expected_calibration_error(
    labels: np.ndarray, scores: np.ndarray, bins: int = 10
) -> float:
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:], strict=False):
        mask = (scores >= left) & (scores < right if right < 1.0 else scores <= right)
        if not np.any(mask):
            continue
        accuracy = labels[mask].mean()
        confidence = scores[mask].mean()
        ece += abs(accuracy - confidence) * mask.mean()
    return float(ece)


def _safe_metric(func, *args, **kwargs) -> float:
    try:
        return float(func(*args, **kwargs))
    except Exception:  # noqa: BLE001
        return float("nan")


def compute_classification_metrics(
    frame: pd.DataFrame,
    label_column: str = "label",
    score_column: str = "score",
    threshold: float = 0.5,
) -> dict[str, float]:
    labels = frame[label_column].to_numpy(dtype=float)
    scores = frame[score_column].to_numpy(dtype=float)
    predictions = (scores >= threshold).astype(int)
    metrics = {
        "auroc": _safe_metric(roc_auc_score, labels, scores),
        "auprc": _safe_metric(average_precision_score, labels, scores),
        "brier": _safe_metric(brier_score_loss, labels, scores),
        "ece": expected_calibration_error(labels, scores),
        "f1": _safe_metric(f1_score, labels, predictions),
        "mcc": _safe_metric(matthews_corrcoef, labels, predictions),
    }
    return metrics


def macro_average_by_group(
    frame: pd.DataFrame,
    group_column: str,
    label_column: str = "label",
    score_column: str = "score",
    threshold: float = 0.5,
) -> dict[str, float]:
    group_metrics = []
    for _, group_frame in frame.groupby(group_column):
        if group_frame[label_column].nunique() < 2:
            continue
        group_metrics.append(
            compute_classification_metrics(
                group_frame,
                label_column=label_column,
                score_column=score_column,
                threshold=threshold,
            )
        )
    if not group_metrics:
        return {}
    keys = group_metrics[0].keys()
    return {
        f"macro_{group_column}_{key}": float(
            np.nanmean([metrics[key] for metrics in group_metrics])
        )
        for key in keys
    }


def grouped_bootstrap(
    frame: pd.DataFrame,
    group_column: str,
    label_column: str = "label",
    score_column: str = "score",
    iterations: int = 1000,
    seed: int = DEFAULT_RANDOM_SEED,
    threshold: float = 0.5,
) -> dict[str, dict[str, float]]:
    groups = list(frame[group_column].astype(str).unique())
    rng = np.random.default_rng(seed)
    samples: list[dict[str, float]] = []
    for _ in range(iterations):
        sampled_groups = rng.choice(groups, size=len(groups), replace=True)
        sampled = pd.concat(
            [
                frame[frame[group_column].astype(str) == group]
                for group in sampled_groups
            ],
            ignore_index=True,
        )
        if sampled[label_column].nunique() < 2:
            continue
        samples.append(
            compute_classification_metrics(
                sampled,
                label_column=label_column,
                score_column=score_column,
                threshold=threshold,
            )
        )
    if not samples:
        return {}
    summary: dict[str, dict[str, float]] = {}
    for key in samples[0]:
        values = np.asarray([sample[key] for sample in samples], dtype=float)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            summary[key] = {"mean": math.nan, "lower": math.nan, "upper": math.nan}
            continue
        summary[key] = {
            "mean": float(finite.mean()),
            "lower": float(np.quantile(finite, 0.025)),
            "upper": float(np.quantile(finite, 0.975)),
        }
    return summary
