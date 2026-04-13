#!/usr/bin/env python3
"""Write a split leakage summary."""

from __future__ import annotations

import argparse

import pandas as pd

from complexvar.utils.io import write_json
from complexvar.utils.splits import leakage_summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--fields", default="protein_group,family_group")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.input, sep="\t")
    summary = leakage_summary(
        frame, split_column="split", fields=args.fields.split(",")
    )
    write_json(summary, args.output)


if __name__ == "__main__":
    main()
