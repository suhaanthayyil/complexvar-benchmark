"""Regression metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _safe_stat(func, left, right) -> float:
    try:
        result = func(left, right)
        value = result.statistic if hasattr(result, "statistic") else result[0]
        value = np.asarray(value).item()
    except Exception:  # noqa: BLE001
        return float("nan")
    return float(value)


def compute_regression_metrics(
    frame: pd.DataFrame,
    label_column: str = "label",
    prediction_column: str = "prediction",
) -> dict[str, float]:
    labels = frame[label_column].to_numpy(dtype=float)
    predictions = frame[prediction_column].to_numpy(dtype=float)
    return {
        "pearson": _safe_stat(pearsonr, labels, predictions),
        "spearman": _safe_stat(spearmanr, labels, predictions),
        "rmse": float(np.sqrt(mean_squared_error(labels, predictions))),
        "mae": float(mean_absolute_error(labels, predictions)),
    }
