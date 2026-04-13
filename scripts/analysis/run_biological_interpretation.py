#!/usr/bin/env python3
"""Summarize ranked variants by gene class."""

from __future__ import annotations

import argparse

import pandas as pd

from complexvar.analysis.biological import (
    rank_disruptive_variants,
    summarize_gene_classes,
)
from complexvar.utils.io import write_tsv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ranked-output", required=True)
    parser.add_argument("--summary-output", required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.predictions, sep="\t")
    ranked = rank_disruptive_variants(frame)
    summary = summarize_gene_classes(ranked)
    write_tsv(ranked, args.ranked_output)
    write_tsv(summary, args.summary_output)


if __name__ == "__main__":
    main()
