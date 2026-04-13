#!/usr/bin/env python3
"""Filter a manifest to a primary interface-distance cohort."""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-distance", type=float, default=8.0)
    parser.add_argument("--require-graph-columns", action="store_true", default=False)
    args = parser.parse_args()

    frame = pd.read_csv(args.input, sep="\t")
    if "min_inter_chain_distance" not in frame.columns:
        raise ValueError("Manifest must contain min_inter_chain_distance.")
    filtered = frame[
        frame["min_inter_chain_distance"].fillna(float("inf")).astype(float)
        <= args.max_distance
    ].copy()
    if args.require_graph_columns:
        required = ["graph_path", "monomer_graph_path"]
        missing = [column for column in required if column not in filtered.columns]
        if missing:
            raise ValueError(
                f"Manifest is missing required graph columns: {', '.join(missing)}"
            )
        filtered = filtered.dropna(subset=required).copy()
    filtered.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
