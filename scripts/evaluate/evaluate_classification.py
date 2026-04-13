#!/usr/bin/env python3
"""Evaluate prediction tables and write benchmark metrics."""

from __future__ import annotations

import argparse

from complexvar.evaluation.evaluate import evaluate_prediction_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--interface-output")
    args = parser.parse_args()
    interface_output = args.interface_output or args.output.replace(
        "all_metrics.json", "interface_stratified_metrics.json"
    )
    evaluate_prediction_table(args.predictions, args.output, interface_output)


if __name__ == "__main__":
    main()
