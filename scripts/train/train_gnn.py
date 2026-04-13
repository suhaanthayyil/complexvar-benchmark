#!/usr/bin/env python3
"""Train the graph model on variant subgraphs."""

from __future__ import annotations

import argparse

import pandas as pd

from complexvar.models.gnn import ComplexVarGAT
from complexvar.models.train import train_graph_model

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch is required for train_gnn.py") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--graph-column",
        default="graph_path",
        choices=["graph_path", "monomer_graph_path"],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    if "split" not in manifest.columns:
        raise ValueError("Manifest must contain a split column.")
    work = manifest.copy()
    work["graph_path"] = work[args.graph_column]
    train_manifest = work[work["split"] == "train"].copy()
    val_manifest = work[work["split"] == "val"].copy()
    test_manifest = work[work["split"] == "test"].copy()
    if train_manifest.empty or val_manifest.empty:
        raise ValueError("Training and validation rows are required.")
    example = torch.load(
        train_manifest.iloc[0]["graph_path"],
        map_location="cpu",
        weights_only=False,
    )
    model = ComplexVarGAT(
        node_dim=int(example.x.shape[1]),
        edge_dim=int(example.edge_attr.shape[1]),
        perturbation_dim=int(example.perturbation.shape[-1]),
    )
    train_graph_model(
        train_manifest=train_manifest,
        val_manifest=val_manifest,
        test_manifest=test_manifest,
        model=model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
    )


if __name__ == "__main__":
    main()
