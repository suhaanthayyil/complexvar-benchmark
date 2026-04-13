#!/usr/bin/env python3
"""Write a simple PyMOL command file for representative variants."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ranked-variants", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.ranked_variants, sep="\t")
    top = frame.head(5)
    lines = ["# PyMOL commands for representative variants"]
    for _, row in top.iterrows():
        residue = str(row.get("residue_id", row.get("sample_id", "unknown")))
        lines.append(f"# highlight {residue}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
