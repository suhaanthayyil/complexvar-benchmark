#!/usr/bin/env python3
"""Map normalized variant tables onto structure files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from complexvar.data.burke import build_gene_accession_table
from complexvar.structure.mapping import (
    map_accession_variants_to_burke,
    map_skempi_to_structures,
)
from complexvar.utils.io import write_tsv


def _prepare_clinvar_accessions(
    clinvar_path: Path, burke_summary_csv: Path, output: Path
) -> Path:
    clinvar = pd.read_csv(clinvar_path, sep="\t")
    mapping = build_gene_accession_table(burke_summary_csv)
    expanded = clinvar.merge(mapping, on="gene_symbol", how="left")
    expanded["protein_accession"] = expanded["uniprot_accession"].where(
        expanded["uniprot_accession"].astype(str) != "",
        expanded["accession"],
    )
    expanded["partner_accessions"] = ""
    return write_tsv(expanded, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", choices=["skempi", "intact", "clinvar"], required=True
    )
    parser.add_argument("--variants", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pdb-dir")
    parser.add_argument("--burke-manifest")
    parser.add_argument("--burke-structure-root")
    parser.add_argument("--burke-summary-csv")
    args = parser.parse_args()

    output = Path(args.output)
    if args.source == "skempi":
        if not args.pdb_dir:
            raise ValueError("--pdb-dir is required for SKEMPI mapping")
        map_skempi_to_structures(
            skempi_path=args.variants,
            pdb_dir=args.pdb_dir,
            output=output,
        )
        return

    if not args.burke_manifest or not args.burke_structure_root:
        raise ValueError(
            "--burke-manifest and --burke-structure-root are required for Burke mapping"
        )

    variants_path = Path(args.variants)
    if args.source == "clinvar":
        if not args.burke_summary_csv:
            raise ValueError("--burke-summary-csv is required for ClinVar mapping")
        prepared = output.with_name(f"{output.stem}.prepared.tsv")
        variants_path = _prepare_clinvar_accessions(
            clinvar_path=variants_path,
            burke_summary_csv=Path(args.burke_summary_csv),
            output=prepared,
        )

    map_accession_variants_to_burke(
        variants_path=variants_path,
        burke_manifest_path=args.burke_manifest,
        structure_root=args.burke_structure_root,
        output=output,
        only_human=args.source == "intact",
    )


if __name__ == "__main__":
    main()
