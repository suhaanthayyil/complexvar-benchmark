#!/usr/bin/env python3
"""Build full-graph bundles and variant subgraphs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from complexvar.graphs.builder import (
    build_full_graph_object,
    build_variant_subgraph,
)
from complexvar.utils.io import ensure_parent, write_tsv

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--edge-cutoff", type=float, default=8.0)
    parser.add_argument("--interface-cutoff", type=float, default=10.0)
    parser.add_argument("--radius", type=float, default=12.0)
    args = parser.parse_args()

    mapping = pd.read_csv(args.mapping, sep="\t")
    output_dir = Path(args.output_dir)
    full_dir = output_dir / "full"
    variant_dir = output_dir / "variants"
    full_dir.mkdir(parents=True, exist_ok=True)
    variant_dir.mkdir(parents=True, exist_ok=True)

    bundle_manifest_rows = []
    bundle_cache: dict[str, dict] = {}
    for structure_path in sorted(
        mapping["structure_path"].dropna().astype(str).unique()
    ):
        graph = build_full_graph_object(
            pdb_path=structure_path,
            edge_cutoff=args.edge_cutoff,
            interface_cutoff=args.interface_cutoff,
        )
        stem = Path(structure_path).stem
        node_path = full_dir / f"{stem}.nodes.tsv"
        edge_path = full_dir / f"{stem}.edges.tsv"
        bundle_path = ensure_parent(full_dir / f"{stem}.bundle.json")
        write_tsv(graph["node_table"], node_path)
        write_tsv(graph["edge_table"], edge_path)
        payload = {
            "pdb_path": graph["pdb_path"],
            "nodes": str(node_path.resolve()),
            "edges": str(edge_path.resolve()),
        }
        if "data" in graph and torch is not None:
            pt_path = full_dir / f"{stem}.pt"
            torch.save(graph["data"], pt_path)
            payload["torch_geometric"] = str(pt_path.resolve())
        bundle_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        bundle_manifest_rows.append(
            {
                "structure_path": structure_path,
                "bundle_path": str(bundle_path.resolve()),
            }
        )
        bundle_cache[structure_path] = graph

    variant_rows = []
    for row in mapping.itertuples(index=False):
        graph = bundle_cache[str(row.structure_path)]
        complex_path = variant_dir / f"{row.sample_id}.complex.pt"
        monomer_path = variant_dir / f"{row.sample_id}.monomer.pt"
        build_variant_subgraph(
            full_graph_bundle=graph,
            residue_id=row.residue_id,
            wildtype=row.wildtype,
            mutant=row.mutant,
            output_path=complex_path,
            radius_angstrom=args.radius,
            monomer_only=False,
        )
        build_variant_subgraph(
            full_graph_bundle=graph,
            residue_id=row.residue_id,
            wildtype=row.wildtype,
            mutant=row.mutant,
            output_path=monomer_path,
            radius_angstrom=args.radius,
            monomer_only=True,
        )
        variant_rows.append(
            {
                **row._asdict(),
                "interface_proximal": int(getattr(row, "is_interface", 0)),
                "graph_path": str(complex_path.resolve()),
                "monomer_graph_path": str(monomer_path.resolve()),
                "full_graph_bundle_path": str(
                    (
                        full_dir / f"{Path(row.structure_path).stem}.bundle.json"
                    ).resolve()
                ),
            }
        )

    write_tsv(
        pd.DataFrame(bundle_manifest_rows), output_dir / "full_graph_manifest.tsv"
    )
    write_tsv(pd.DataFrame(variant_rows), output_dir / "variant_graph_manifest.tsv")
    (output_dir / "graph_cache_summary.json").write_text(
        json.dumps(
            {
                "structures": len(bundle_manifest_rows),
                "variants": len(variant_rows),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
