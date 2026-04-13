import pandas as pd

from complexvar.metrics.classification import (
    compute_classification_metrics,
    macro_average_by_group,
)
from complexvar.metrics.regression import compute_regression_metrics


def test_classification_metrics_smoke():
    frame = pd.DataFrame(
        {
            "label": [0, 0, 1, 1],
            "score": [0.1, 0.2, 0.8, 0.9],
        }
    )
    metrics = compute_classification_metrics(frame)
    assert metrics["auroc"] == 1.0
    assert metrics["auprc"] == 1.0


def test_macro_average_by_group_smoke():
    frame = pd.DataFrame(
        {
            "label": [0, 1, 0, 1],
            "score": [0.1, 0.8, 0.2, 0.9],
            "protein_group": ["p1", "p1", "p2", "p2"],
        }
    )
    metrics = macro_average_by_group(frame, "protein_group")
    assert "macro_protein_group_auroc" in metrics


def test_regression_metrics_smoke():
    frame = pd.DataFrame({"label": [0.0, 1.0, 2.0], "prediction": [0.1, 1.1, 1.8]})
    metrics = compute_regression_metrics(frame)
    assert metrics["rmse"] >= 0.0
