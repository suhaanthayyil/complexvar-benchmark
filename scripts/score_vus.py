#!/usr/bin/env python3
"""Score mapped ClinVar variants with a trained graph model."""

from __future__ import annotations

import argparse

import pandas as pd

from complexvar.models.gnn import ComplexVarGAT
from complexvar.utils.io import write_tsv

try:
    import torch
    from torch_geometric.loader import DataLoader
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("torch and torch-geometric are required for VUS scoring") from exc


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--graph-column", default="graph_path")
    parser.add_argument("--top-n", type=int, default=500)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, sep="\t")
    vus = manifest[manifest.get("is_vus", 0).fillna(0).astype(int) == 1].copy()
    if "interface_proximal" in vus.columns:
        vus = vus[vus["interface_proximal"].fillna(0).astype(int) == 1].copy()
    if vus.empty:
        raise ValueError("No interface-proximal VUS rows were found in the manifest.")

    example = torch.load(vus.iloc[0][args.graph_column], map_location="cpu", weights_only=False)
    model = ComplexVarGAT(
        node_dim=int(example.x.shape[1]),
        edge_dim=int(example.edge_attr.shape[1]),
        perturbation_dim=int(example.perturbation.shape[-1]),
    )
    state_dict = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model = model.to(_device())
    model.eval()

    samples = []
    for row in vus.itertuples(index=False):
        data = torch.load(getattr(row, args.graph_column), map_location="cpu", weights_only=False)
        data.sample_id = row.sample_id
        samples.append(data)

    loader = DataLoader(samples, batch_size=64, shuffle=False)
    score_rows = []
    with torch.no_grad():
        offset = 0
        for batch in loader:
            batch = batch.to(_device())
            outputs = model(batch)
            scores = torch.sigmoid(outputs["classification"]).detach().cpu().numpy()
            regression = outputs["regression"].detach().cpu().numpy()
            for index in range(len(scores)):
                meta = vus.iloc[offset + index]
                score_rows.append(
                    {
                        **meta.to_dict(),
                        "pathogenicity_probability": float(scores[index]),
                        "predicted_ddg_proxy": float(regression[index]),
                    }
                )
            offset += len(scores)

    output = pd.DataFrame(score_rows).sort_values(
        ["pathogenicity_probability", "predicted_ddg_proxy"],
        ascending=[False, False],
    )
    output = output.head(args.top_n).copy()
    write_tsv(output, args.output)


if __name__ == "__main__":
    main()
