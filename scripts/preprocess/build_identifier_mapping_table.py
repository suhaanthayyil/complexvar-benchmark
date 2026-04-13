#!/usr/bin/env python3
"""Build a simple identifier mapping table between two sources."""

from __future__ import annotations

import argparse

import pandas as pd

from complexvar.utils.io import write_tsv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-key", required=True)
    parser.add_argument("--right-key", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    left = pd.read_csv(args.left, sep="\t")
    right = pd.read_csv(args.right, sep="\t")
    mapping = left.merge(
        right,
        left_on=args.left_key,
        right_on=args.right_key,
        how="inner",
        suffixes=("_left", "_right"),
    )
    write_tsv(mapping, args.output)


if __name__ == "__main__":
    main()
