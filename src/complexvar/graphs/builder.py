"""Graph construction utilities."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from complexvar.features import amino_acid_one_hot, mutation_descriptor
from complexvar.structure.mapping import (
    _dssp_secondary_structure,
    _residue_relative_sasa,
    build_contact_summary,
    load_structure_residues,
)
from complexvar.utils.io import ensure_parent, write_tsv

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from torch_geometric.data import Data
except ImportError:  # pragma: no cover
    Data = None


SECONDARY_STRUCTURE_STATES = ("helix", "sheet", "loop")


def secondary_structure_one_hot(label: str) -> list[int]:
    normalized = str(label).strip().lower()
    if normalized not in SECONDARY_STRUCTURE_STATES:
        normalized = "loop"
    return [int(normalized == state) for state in SECONDARY_STRUCTURE_STATES]


def _node_feature_row(row: pd.Series) -> list[float]:
    min_distance = row.get("min_inter_chain_distance", math.inf)
    if pd.isna(min_distance) or not math.isfinite(float(min_distance)):
        interface_distance_score = 0.0
    else:
        interface_distance_score = math.exp(-min(float(min_distance), 20.0) / 8.0)
    return [
        *amino_acid_one_hot(str(row["residue_code"])),
        float(
            row.get("relative_sasa", math.nan)
            if pd.notna(row.get("relative_sasa", math.nan))
            else 0.0
        ),
        *secondary_structure_one_hot(str(row.get("secondary_structure", "loop"))),
        float(row.get("b_factor_norm", 0.0)),
        float(row.get("is_interface", 0.0)),
        float(row.get("local_degree", 0.0)),
        float(row.get("inter_chain_contacts", 0.0)),
        float(row.get("burial_proxy", 0.0)),
        float(row.get("solvent_proxy", 0.0)),
        float(interface_distance_score),
        float(row.get("center_distance_norm", 0.0)),
        float(row.get("partner_chain_flag", 0.0)),
        float(row.get("relative_x", 0.0)),
        float(row.get("relative_y", 0.0)),
        float(row.get("relative_z", 0.0)),
    ]


def _edge_feature_row(row: pd.Series) -> list[float]:
    distance = float(row["distance"])
    return [
        distance,
        float(row["is_inter_chain"]),
        float(row["is_backbone"]),
        1.0 / max(distance, 0.1),
        float(row.get("delta_sasa", 0.0)),
        float(row.get("sequence_separation", 0.0)),
        float(row.get("src_inter_chain_contacts", 0.0)),
        float(row.get("tgt_inter_chain_contacts", 0.0)),
        float(row.get("orientation_angle", 0.0)),
        float(row.get("hotspot_proxy", 0.0)),
        float(row.get("bsa_proxy", 0.0)),
    ]


def build_residue_graph_tables(
    pdb_path: str | Path,
    edge_cutoff: float = 8.0,
    interface_cutoff: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pdb_path = Path(pdb_path)
    residues = load_structure_residues(pdb_path)
    nodes, edges = build_contact_summary(
        residues,
        edge_cutoff=edge_cutoff,
        interface_cutoff=interface_cutoff,
    )
    sasa_lookup = _residue_relative_sasa(pdb_path)
    ss_lookup = _dssp_secondary_structure(pdb_path)
    nodes = nodes.copy()
    nodes["relative_sasa"] = nodes["residue_id"].map(sasa_lookup).fillna(0.0)
    nodes["secondary_structure"] = nodes["residue_id"].map(ss_lookup).fillna("loop")
    max_b = float(nodes["b_factor"].max()) if not nodes.empty else 0.0
    if max_b > 0:
        nodes["b_factor_norm"] = nodes["b_factor"] / max_b
    else:
        nodes["b_factor_norm"] = 0.0
    return nodes, edges


def build_full_graph_object(
    pdb_path: str | Path,
    edge_cutoff: float = 8.0,
    interface_cutoff: float = 10.0,
) -> dict[str, Any]:
    nodes, edges = build_residue_graph_tables(
        pdb_path=pdb_path,
        edge_cutoff=edge_cutoff,
        interface_cutoff=interface_cutoff,
    )
    nodes = nodes.reset_index(drop=True).copy()
    nodes["node_index"] = range(len(nodes))
    residue_to_index = dict(zip(nodes["residue_id"], nodes["node_index"], strict=False))

    # Build lookups for edge enrichment
    sasa_map = dict(
        zip(nodes["residue_id"], nodes.get("relative_sasa", 0.0), strict=False)
    )
    icc_map = dict(
        zip(nodes["residue_id"], nodes.get("inter_chain_contacts", 0), strict=False)
    )

    edges = edges.copy()
    # Add delta SASA between edge endpoints
    src_sasa = edges["source_residue_id"].map(sasa_map).fillna(0.0)
    tgt_sasa = edges["target_residue_id"].map(sasa_map).fillna(0.0)
    edges["delta_sasa"] = (src_sasa - tgt_sasa).abs()
    # Add source/target inter-chain contact counts (normalized)
    max_icc = max(max(icc_map.values(), default=1), 1)
    edges["src_inter_chain_contacts"] = (
        edges["source_residue_id"].map(icc_map).fillna(0.0) / max_icc
    )
    edges["tgt_inter_chain_contacts"] = (
        edges["target_residue_id"].map(icc_map).fillna(0.0) / max_icc
    )
    # Fill missing sequence_separation
    if "sequence_separation" not in edges.columns:
        edges["sequence_separation"] = 0.0

    if "center_distance_norm" not in nodes.columns:
        nodes["center_distance_norm"] = 0.0
    if "partner_chain_flag" not in nodes.columns:
        nodes["partner_chain_flag"] = 0.0
    if "interface_distance_score" not in nodes.columns:
        nodes["interface_distance_score"] = nodes["min_inter_chain_distance"].apply(
            lambda value: 0.0
            if pd.isna(value) or not math.isfinite(float(value))
            else math.exp(-min(float(value), 20.0) / 8.0)
        )
    if "relative_x" not in nodes.columns:
        nodes["relative_x"] = 0.0
    if "relative_y" not in nodes.columns:
        nodes["relative_y"] = 0.0
    if "relative_z" not in nodes.columns:
        nodes["relative_z"] = 0.0

    # Compute orientation angle: cosine of angle between CA-CB vectors across edge
    def _compute_orientation_angle(row: pd.Series) -> float:
        """Compute cosine of angle between residue normal vectors."""
        try:
            src_cb = (
                row.get("source_cb_x"),
                row.get("source_cb_y"),
                row.get("source_cb_z"),
            )
            tgt_cb = (
                row.get("target_cb_x"),
                row.get("target_cb_y"),
                row.get("target_cb_z"),
            )
            src_ca = (
                row.get("source_ca_x"),
                row.get("source_ca_y"),
                row.get("source_ca_z"),
            )
            tgt_ca = (
                row.get("target_ca_x"),
                row.get("target_ca_y"),
                row.get("target_ca_z"),
            )
            if any(v is None or pd.isna(v) for v in [*src_cb, *tgt_cb, *src_ca, *tgt_ca]):
                return 0.0
            vec_src = (src_cb[0] - src_ca[0], src_cb[1] - src_ca[1], src_cb[2] - src_ca[2])
            vec_tgt = (tgt_cb[0] - tgt_ca[0], tgt_cb[1] - tgt_ca[1], tgt_cb[2] - tgt_ca[2])
            norm_src = math.sqrt(sum(v ** 2 for v in vec_src))
            norm_tgt = math.sqrt(sum(v ** 2 for v in vec_tgt))
            if norm_src < 1e-6 or norm_tgt < 1e-6:
                return 0.0
            dot = sum(a * b for a, b in zip(vec_src, vec_tgt))
            cos_angle = dot / (norm_src * norm_tgt)
            return float(cos_angle)
        except Exception:
            return 0.0

    edges["orientation_angle"] = edges.apply(_compute_orientation_angle, axis=1)

    # Compute hotspot proxy: exp(-distance/5.0) * inter_chain_contact_count
    max_icc = max(max(icc_map.values(), default=1), 1)
    edges["hotspot_proxy"] = edges.apply(
        lambda row: math.exp(-float(row["distance"]) / 5.0)
        * (row.get("is_inter_chain", 0) * max_icc),
        axis=1,
    )

    # Compute BSA proxy: estimated buried surface area contribution
    edges["bsa_proxy"] = edges.apply(
        lambda row: float(row.get("delta_sasa", 0.0))
        * (1.0 / max(float(row["distance"]), 0.1))
        * float(row.get("is_inter_chain", 0)),
        axis=1,
    )

    edges["source_index"] = edges["source_residue_id"].map(residue_to_index)
    edges["target_index"] = edges["target_residue_id"].map(residue_to_index)
    edges = edges.dropna(subset=["source_index", "target_index"]).copy()
    edges["source_index"] = edges["source_index"].astype(int)
    edges["target_index"] = edges["target_index"].astype(int)
    graph: dict[str, Any] = {
        "pdb_path": str(Path(pdb_path).resolve()),
        "node_table": nodes,
        "edge_table": edges,
        "node_feature_columns": [
            "aa_one_hot",
            "relative_sasa",
            "secondary_structure",
            "b_factor_norm",
            "is_interface",
            "local_degree",
            "inter_chain_contacts",
            "burial_proxy",
            "solvent_proxy",
            "interface_distance_score",
            "center_distance_norm",
            "partner_chain_flag",
            "relative_x",
            "relative_y",
            "relative_z",
        ],
        "edge_feature_columns": [
            "distance",
            "is_inter_chain",
            "is_backbone",
            "distance_inverse",
            "delta_sasa",
            "sequence_separation",
            "src_inter_chain_contacts",
            "tgt_inter_chain_contacts",
            "orientation_angle",
            "hotspot_proxy",
            "bsa_proxy",
        ],
    }
    if torch is not None and Data is not None:
        x = torch.tensor(
            [_node_feature_row(row) for _, row in nodes.iterrows()], dtype=torch.float32
        )
        edge_index = torch.tensor(
            edges[["source_index", "target_index"]].to_numpy().T,
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [_edge_feature_row(row) for _, row in edges.iterrows()],
            dtype=torch.float32,
        )
        graph["data"] = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            residue_ids=nodes["residue_id"].astype(str).tolist(),
            chain_ids=nodes["chain_id"].astype(str).tolist(),
        )
    return graph


def write_graph_bundle(
    pdb_path: str | Path,
    output_dir: str | Path,
    edge_cutoff: float = 8.0,
    interface_cutoff: float = 10.0,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graph = build_full_graph_object(
        pdb_path=pdb_path,
        edge_cutoff=edge_cutoff,
        interface_cutoff=interface_cutoff,
    )
    stem = Path(pdb_path).stem
    node_path = output_dir / f"{stem}.nodes.tsv"
    edge_path = output_dir / f"{stem}.edges.tsv"
    write_tsv(graph["node_table"], node_path)
    write_tsv(graph["edge_table"], edge_path)
    bundle_path = ensure_parent(output_dir / f"{stem}.bundle.json")
    payload = {
        "pdb_path": graph["pdb_path"],
        "nodes": str(node_path),
        "edges": str(edge_path),
    }
    pt_path = output_dir / f"{stem}.pt"
    if "data" in graph and torch is not None:
        torch.save(graph["data"], pt_path)
        payload["torch_geometric"] = str(pt_path)
    bundle_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return bundle_path


def build_variant_subgraph(
    full_graph_bundle: dict[str, Any],
    residue_id: str,
    wildtype: str,
    mutant: str,
    output_path: str | Path,
    radius_angstrom: float = 15.0,
    monomer_only: bool = False,
) -> Path:
    nodes = full_graph_bundle["node_table"].copy()
    edges = full_graph_bundle["edge_table"].copy()
    center_row = nodes[nodes["residue_id"] == residue_id]
    if center_row.empty:
        raise ValueError(f"Residue {residue_id} not found in graph bundle.")
    center = center_row.iloc[0]
    center_xyz = (float(center["ca_x"]), float(center["ca_y"]), float(center["ca_z"]))
    nodes["center_distance"] = nodes.apply(
        lambda row: math.sqrt(
            (float(row["ca_x"]) - center_xyz[0]) ** 2
            + (float(row["ca_y"]) - center_xyz[1]) ** 2
            + (float(row["ca_z"]) - center_xyz[2]) ** 2
        ),
        axis=1,
    )
    sub_nodes = nodes[nodes["center_distance"] <= radius_angstrom].copy()
    if monomer_only:
        sub_nodes = sub_nodes[sub_nodes["chain_id"] == center["chain_id"]].copy()
    sub_nodes["center_distance_norm"] = (
        sub_nodes["center_distance"] / max(float(radius_angstrom), 1.0)
    ).clip(0.0, 1.0)
    sub_nodes["partner_chain_flag"] = (
        sub_nodes["chain_id"].astype(str) != str(center["chain_id"])
    ).astype(float)
    sub_nodes["interface_distance_score"] = sub_nodes["min_inter_chain_distance"].apply(
        lambda value: 0.0
        if pd.isna(value) or not math.isfinite(float(value))
        else math.exp(-min(float(value), 20.0) / 8.0)
    )
    sub_nodes["relative_x"] = (
        sub_nodes["ca_x"].astype(float) - center_xyz[0]
    ) / max(float(radius_angstrom), 1.0)
    sub_nodes["relative_y"] = (
        sub_nodes["ca_y"].astype(float) - center_xyz[1]
    ) / max(float(radius_angstrom), 1.0)
    sub_nodes["relative_z"] = (
        sub_nodes["ca_z"].astype(float) - center_xyz[2]
    ) / max(float(radius_angstrom), 1.0)
    keep_ids = set(sub_nodes["residue_id"].astype(str))
    sub_edges = edges[
        edges["source_residue_id"].isin(keep_ids)
        & edges["target_residue_id"].isin(keep_ids)
    ].copy()
    if monomer_only:
        sub_edges = sub_edges[sub_edges["is_inter_chain"] == 0].copy()
    sub_nodes = sub_nodes.reset_index(drop=True)
    sub_nodes["subgraph_index"] = range(len(sub_nodes))
    index_lookup = dict(
        zip(sub_nodes["residue_id"], sub_nodes["subgraph_index"], strict=False)
    )
    sub_edges["source_index"] = sub_edges["source_residue_id"].map(index_lookup)
    sub_edges["target_index"] = sub_edges["target_residue_id"].map(index_lookup)
    sub_edges = sub_edges.dropna(subset=["source_index", "target_index"]).copy()
    sub_edges["source_index"] = sub_edges["source_index"].astype(int)
    sub_edges["target_index"] = sub_edges["target_index"].astype(int)
    perturbation = mutation_descriptor(wildtype=wildtype, mutant=mutant)

    output_path = ensure_parent(output_path)
    payload: dict[str, Any] = {
        "residue_id": residue_id,
        "wildtype": wildtype,
        "mutant": mutant,
        "perturbation": perturbation,
        "nodes": sub_nodes.to_dict(orient="records"),
        "edges": sub_edges.to_dict(orient="records"),
        "monomer_only": monomer_only,
    }
    if torch is not None and Data is not None:
        x = torch.tensor(
            [_node_feature_row(row) for _, row in sub_nodes.iterrows()],
            dtype=torch.float32,
        )
        edge_index = torch.tensor(
            sub_edges[["source_index", "target_index"]].to_numpy().T,
            dtype=torch.long,
        )
        edge_attr = torch.tensor(
            [_edge_feature_row(row) for _, row in sub_edges.iterrows()],
            dtype=torch.float32,
        )
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            mutant_index=int(
                sub_nodes[sub_nodes["residue_id"] == residue_id]["subgraph_index"].iloc[
                    0
                ]
            ),
            perturbation=torch.tensor(
                [list(perturbation.values())], dtype=torch.float32
            ),
            residue_ids=sub_nodes["residue_id"].astype(str).tolist(),
            chain_ids=sub_nodes["chain_id"].astype(str).tolist(),
        )
        torch.save(data, output_path)
    else:
        output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return output_path
