#!/usr/bin/env python3
"""Train the sequence-only baseline."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from complexvar.models.sequence import SequenceMLP, build_sequence_feature_vector
from complexvar.models.train import train_tabular_model
from complexvar.structure.mapping import load_structure_residues

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for train_sequence.py") from exc


def _chain_sequence(structure_path: str, chain_id: str) -> str:
    cache_key = (structure_path, chain_id)
    if cache_key in _CHAIN_SEQUENCE_CACHE:
        return _CHAIN_SEQUENCE_CACHE[cache_key]
    residues = load_structure_residues(structure_path)
    sequence = "".join(
        residue.residue_code
        for residue in residues
        if residue.chain_id == chain_id and residue.residue_code != "X"
    )
    _CHAIN_SEQUENCE_CACHE[cache_key] = sequence
    return sequence


_CHAIN_SEQUENCE_CACHE: dict[tuple[str, str], str] = {}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    frame = pd.read_csv(args.mapping, sep="\t")
    feature_rows = []
    for row in frame.itertuples(index=False):
        structure_path = row.structure_path
        chain_id = row.mutated_chain_id if hasattr(row, "mutated_chain_id") else "A"
        sequence = _chain_sequence(structure_path, chain_id)
        position = int(row.residue_number) - 1
        if position < 0 or position >= len(sequence):
            continue
        encoding = build_sequence_feature_vector(
            sequence=sequence,
            position=position,
            wildtype=row.wildtype,
            mutant=row.mutant,
        )
        record = {
            "sample_id": row.sample_id,
            "split": row.split,
            "binary_label": getattr(row, "binary_label", float("nan")),
            "ddg": getattr(row, "ddg", float("nan")),
        }
        for index, value in enumerate(encoding.vector):
            record[f"f_{index}"] = float(value)
        feature_rows.append(record)
    features = pd.DataFrame(feature_rows)
    train = features[features["split"] == "train"].copy()
    val = features[features["split"] == "val"].copy()
    test = features[features["split"] == "test"].copy()
    feature_columns = [column for column in features.columns if column.startswith("f_")]
    train_features = torch.tensor(
        train[feature_columns].to_numpy(), dtype=torch.float32
    )
    val_features = torch.tensor(val[feature_columns].to_numpy(), dtype=torch.float32)
    test_features = torch.tensor(test[feature_columns].to_numpy(), dtype=torch.float32)
    train_cls = torch.tensor(
        train["binary_label"].fillna(0.0).to_numpy(), dtype=torch.float32
    )
    val_cls = torch.tensor(
        val["binary_label"].fillna(0.0).to_numpy(), dtype=torch.float32
    )
    test_cls = torch.tensor(
        test["binary_label"].fillna(0.0).to_numpy(), dtype=torch.float32
    )
    train_reg = torch.tensor(train["ddg"].fillna(0.0).to_numpy(), dtype=torch.float32)
    val_reg = torch.tensor(val["ddg"].fillna(0.0).to_numpy(), dtype=torch.float32)
    test_reg = torch.tensor(test["ddg"].fillna(0.0).to_numpy(), dtype=torch.float32)
    train_cls_mask = torch.tensor(
        train["binary_label"].notna().astype(int).to_numpy(), dtype=torch.float32
    )
    val_cls_mask = torch.tensor(
        val["binary_label"].notna().astype(int).to_numpy(), dtype=torch.float32
    )
    test_cls_mask = torch.tensor(
        test["binary_label"].notna().astype(int).to_numpy(), dtype=torch.float32
    )
    train_reg_mask = torch.tensor(
        train["ddg"].notna().astype(int).to_numpy(), dtype=torch.float32
    )
    val_reg_mask = torch.tensor(
        val["ddg"].notna().astype(int).to_numpy(), dtype=torch.float32
    )
    test_reg_mask = torch.tensor(
        test["ddg"].notna().astype(int).to_numpy(), dtype=torch.float32
    )
    model = SequenceMLP(input_dim=len(feature_columns))
    metadata_columns = [
        column
        for column in [
            "sample_id",
            "source_dataset",
            "protein_group",
            "family_group",
            "interface_proximal",
            "is_interface",
        ]
        if column in features.columns
    ]
    train_tabular_model(
        train_features=train_features,
        val_features=val_features,
        test_features=test_features,
        train_cls=train_cls,
        val_cls=val_cls,
        test_cls=test_cls,
        train_reg=train_reg,
        val_reg=val_reg,
        test_reg=test_reg,
        train_cls_mask=train_cls_mask,
        val_cls_mask=val_cls_mask,
        test_cls_mask=test_cls_mask,
        train_reg_mask=train_reg_mask,
        val_reg_mask=val_reg_mask,
        test_reg_mask=test_reg_mask,
        model=model,
        output_dir=Path(args.output_dir),
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_metadata=val[metadata_columns].reset_index(drop=True),
        test_metadata=test[metadata_columns].reset_index(drop=True),
    )


if __name__ == "__main__":
    main()
