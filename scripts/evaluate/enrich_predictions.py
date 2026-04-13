#!/usr/bin/env python3
"""Merge prediction tables with manifest metadata."""

from __future__ import annotations

import argparse

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    predictions = pd.read_csv(args.predictions, sep="\t")
    manifest = pd.read_csv(args.manifest, sep="\t")
    keep_columns = [
        column
        for column in [
            "sample_id",
            "protein_accession",
            "partner_accession",
            "protein_cluster",
            "partner_cluster",
            "family_group",
            "interface_proximal",
            "is_interface",
            "source_dataset",
            "split",
        ]
        if column in manifest.columns
    ]
    metadata = manifest[keep_columns].drop_duplicates(subset=["sample_id"])
    enriched = predictions.drop(
        columns=[column for column in keep_columns if column != "sample_id"],
        errors="ignore",
    ).merge(metadata, on="sample_id", how="left")
    if "protein_group" not in enriched.columns:
        if "protein_accession" in enriched.columns:
            enriched["protein_group"] = enriched["protein_accession"]
        elif "protein_cluster" in enriched.columns:
            enriched["protein_group"] = enriched["protein_cluster"]
    enriched.to_csv(args.output, sep="\t", index=False)


if __name__ == "__main__":
    main()
