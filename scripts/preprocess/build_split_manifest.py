#!/usr/bin/env python3
"""Build leakage-aware split assignments."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from complexvar.utils.io import write_json, write_tsv
from complexvar.utils.splits import (
    SplitFractions,
    assign_identity_clusters,
    attach_clusters,
    leakage_summary,
    make_group_splits,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--proteins", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--audit-output", required=True)
    parser.add_argument("--identity-threshold", type=float, default=0.30)
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    args = parser.parse_args()

    samples = pd.read_csv(args.input, sep="\t")
    proteins = pd.read_csv(args.proteins, sep="\t")
    clusters = assign_identity_clusters(
        proteins=proteins,
        identity_threshold=args.identity_threshold,
    )
    samples = attach_clusters(samples, clusters)
    if "pdb_id" in samples.columns:
        samples["split_group"] = samples.apply(
            lambda row: (
                f"{row['family_group']}::{row['pdb_id']}"
                if str(row.get("source_dataset", "")) == "SKEMPI"
                else row["family_group"]
            ),
            axis=1,
        )
    else:
        samples["split_group"] = samples["family_group"]
    split_frame = make_group_splits(
        samples,
        group_column="split_group",
        fractions=SplitFractions(
            train=args.train_fraction,
            val=args.val_fraction,
            test=args.test_fraction,
        ),
    )
    write_tsv(split_frame, args.output)
    write_json(
        leakage_summary(
            split_frame,
            split_column="split",
            fields=[
                "protein_cluster",
                "partner_cluster",
                "family_group",
                "split_group",
            ],
        ),
        args.audit_output,
    )
    write_tsv(clusters, Path(args.output).with_name("protein_clusters.tsv"))


if __name__ == "__main__":
    main()
