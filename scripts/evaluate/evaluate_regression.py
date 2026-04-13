#!/usr/bin/env python3
"""Regression evaluation wrapper."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from complexvar.metrics.regression import compute_regression_metrics
from complexvar.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.predictions, sep="\t")
    if "split" in frame.columns:
        frame = frame[frame["split"] == "test"].copy()
    frame = frame.dropna(subset=["ddg", "prediction"])
    frame = frame[np.isfinite(frame["ddg"]) & np.isfinite(frame["prediction"])]
    frame = frame.drop(columns=["label"], errors="ignore")
    frame = frame.rename(columns={"ddg": "label"})
    write_json(compute_regression_metrics(frame), args.output)


if __name__ == "__main__":
    main()
