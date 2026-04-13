#!/usr/bin/env python3
"""Build a protein sequence table for clustering and split generation."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from complexvar.structure.mapping import load_structure_residues
from complexvar.utils.io import write_tsv


def _accession_root(accession: str) -> str:
    text = str(accession).strip()
    if not text or text == "nan":
        return ""
    return text.split("-")[0]


def _chain_sequence(structure_path: str | Path, chain_id: str) -> str:
    residues = load_structure_residues(structure_path)
    sequence = [
        residue.residue_code
        for residue in residues
        if residue.chain_id == chain_id and residue.residue_code != "X"
    ]
    return "".join(sequence)


def _monomer_sequence(monomer_root: Path, accession: str) -> str:
    pdb_path = monomer_root / f"{_accession_root(accession)}.pdb"
    if not pdb_path.exists():
        return ""
    residues = load_structure_residues(pdb_path)
    sequence = [
        residue.residue_code for residue in residues if residue.residue_code != "X"
    ]
    return "".join(sequence)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--monomer-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    monomer_root = Path(args.monomer_root)
    records: dict[str, dict[str, str]] = {}

    for input_path in args.inputs:
        frame = pd.read_csv(input_path, sep="\t")
        for row in frame.itertuples(index=False):
            structure_path = getattr(row, "structure_path", "")
            if getattr(row, "protein_accession", ""):
                accession = str(row.protein_accession)
                if accession not in records:
                    sequence = ""
                    if ":" in accession and structure_path:
                        chain_id = accession.split(":")[-1]
                        sequence = _chain_sequence(structure_path, chain_id)
                    else:
                        sequence = _monomer_sequence(monomer_root, accession)
                    if sequence:
                        records[accession] = {
                            "accession": accession,
                            "sequence": sequence,
                        }
            if getattr(row, "partner_accession", ""):
                accession = str(row.partner_accession)
                if accession not in records:
                    sequence = ""
                    if ":" in accession and structure_path:
                        chain_id = accession.split(":")[-1]
                        sequence = _chain_sequence(structure_path, chain_id)
                    else:
                        sequence = _monomer_sequence(monomer_root, accession)
                    if sequence:
                        records[accession] = {
                            "accession": accession,
                            "sequence": sequence,
                        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_tsv(pd.DataFrame(records.values()).sort_values("accession"), output)


if __name__ == "__main__":
    main()
