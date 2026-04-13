"""Variant-to-structure mapping utilities."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from Bio.PDB import DSSP, PDBParser
from scipy.spatial import cKDTree

from complexvar.constants import THREE_TO_ONE
from complexvar.features import interface_burial_proxy, solvent_exposure_proxy
from complexvar.utils.io import write_tsv

try:
    import freesasa  # type: ignore
except ImportError:  # pragma: no cover
    freesasa = None


@dataclass(frozen=True)
class ResidueAtom:
    name: str
    coord: tuple[float, float, float]


@dataclass(frozen=True)
class ResidueRecord:
    chain_id: str
    residue_number: int
    insertion_code: str
    residue_name: str
    residue_code: str
    sequence_index: int
    ca_coord: tuple[float, float, float] | None
    b_factor: float
    atoms: tuple[ResidueAtom, ...]

    @property
    def residue_id(self) -> str:
        suffix = self.insertion_code if self.insertion_code else ""
        return f"{self.chain_id}:{self.residue_number}{suffix}"


def _distance(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right, strict=False)))


def load_structure_residues(pdb_path: str | Path) -> list[ResidueRecord]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complexvar", str(pdb_path))
    residues: list[ResidueRecord] = []
    sequence_index = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                hetero_flag, resseq, icode = residue.id
                if hetero_flag.strip():
                    continue
                residue_name = residue.get_resname().upper()
                residue_code = THREE_TO_ONE.get(residue_name, "X")
                atoms = tuple(
                    ResidueAtom(
                        name=str(atom.get_name()),
                        coord=tuple(float(value) for value in atom.coord),
                    )
                    for atom in residue
                )
                ca_coord = None
                b_factor = 0.0
                if "CA" in residue:
                    ca_atom = residue["CA"]
                    ca_coord = tuple(float(value) for value in ca_atom.coord)
                    b_factor = float(ca_atom.get_bfactor())
                residues.append(
                    ResidueRecord(
                        chain_id=str(chain.id),
                        residue_number=int(resseq),
                        insertion_code=(
                            str(icode).strip() if str(icode).strip() != "?" else ""
                        ),
                        residue_name=residue_name,
                        residue_code=residue_code,
                        sequence_index=sequence_index,
                        ca_coord=ca_coord,
                        b_factor=b_factor,
                        atoms=atoms,
                    )
                )
                sequence_index += 1
        break
    return residues


def _min_heavy_atom_distance(left: ResidueRecord, right: ResidueRecord) -> float:
    best = math.inf
    for atom_left in left.atoms:
        for atom_right in right.atoms:
            current = _distance(atom_left.coord, atom_right.coord)
            if current < best:
                best = current
    return best


def _backbone_neighbors(residues: list[ResidueRecord]) -> set[tuple[str, str]]:
    neighbors: set[tuple[str, str]] = set()
    by_chain: dict[str, list[ResidueRecord]] = {}
    for residue in residues:
        by_chain.setdefault(residue.chain_id, []).append(residue)
    for chain_residues in by_chain.values():
        chain_residues = sorted(
            chain_residues,
            key=lambda residue: (residue.residue_number, residue.insertion_code),
        )
        for left, right in zip(chain_residues[:-1], chain_residues[1:], strict=False):
            neighbors.add((left.residue_id, right.residue_id))
            neighbors.add((right.residue_id, left.residue_id))
    return neighbors


def build_contact_summary(
    residues: list[ResidueRecord],
    edge_cutoff: float = 8.0,
    interface_cutoff: float = 10.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    neighbor_edges = _backbone_neighbors(residues)
    node_rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    local_degree_map = {residue.residue_id: 0 for residue in residues}
    inter_chain_map = {residue.residue_id: 0 for residue in residues}
    min_inter_chain_map = {residue.residue_id: math.inf for residue in residues}
    max_cutoff = max(edge_cutoff, interface_cutoff)
    atom_coords: list[tuple[float, float, float]] = []
    atom_residue_indices: list[int] = []
    residue_pair_min: dict[tuple[int, int], float] = {}
    for residue_index, residue in enumerate(residues):
        for atom in residue.atoms:
            atom_coords.append(atom.coord)
            atom_residue_indices.append(residue_index)

    if atom_coords:
        tree = cKDTree(atom_coords)
        for atom_left_index, atom_right_index in tree.query_pairs(r=max_cutoff):
            residue_left_index = atom_residue_indices[atom_left_index]
            residue_right_index = atom_residue_indices[atom_right_index]
            if residue_left_index == residue_right_index:
                continue
            pair_key = (
                (residue_left_index, residue_right_index)
                if residue_left_index < residue_right_index
                else (residue_right_index, residue_left_index)
            )
            distance = _distance(
                atom_coords[atom_left_index],
                atom_coords[atom_right_index],
            )
            previous = residue_pair_min.get(pair_key)
            if previous is None or distance < previous:
                residue_pair_min[pair_key] = distance

    for (left_index, right_index), distance in residue_pair_min.items():
        left = residues[left_index]
        right = residues[right_index]
        if left.chain_id != right.chain_id:
            min_inter_chain_map[left.residue_id] = min(
                min_inter_chain_map[left.residue_id], distance
            )
            min_inter_chain_map[right.residue_id] = min(
                min_inter_chain_map[right.residue_id], distance
            )
        if distance <= edge_cutoff:
            is_inter_chain = int(left.chain_id != right.chain_id)
            is_backbone = int((left.residue_id, right.residue_id) in neighbor_edges)
            edge_rows.append(
                {
                    "source_residue_id": left.residue_id,
                    "target_residue_id": right.residue_id,
                    "distance": distance,
                    "is_inter_chain": is_inter_chain,
                    "is_backbone": is_backbone,
                }
            )
            edge_rows.append(
                {
                    "source_residue_id": right.residue_id,
                    "target_residue_id": left.residue_id,
                    "distance": distance,
                    "is_inter_chain": is_inter_chain,
                    "is_backbone": is_backbone,
                }
            )
            local_degree_map[left.residue_id] += 1
            local_degree_map[right.residue_id] += 1
            if is_inter_chain:
                inter_chain_map[left.residue_id] += 1
                inter_chain_map[right.residue_id] += 1

    for left in residues:
        local_degree = local_degree_map[left.residue_id]
        inter_chain_contacts = inter_chain_map[left.residue_id]
        min_inter_chain_distance = min_inter_chain_map[left.residue_id]
        solvent_proxy = solvent_exposure_proxy(local_degree)
        node_rows.append(
            {
                "residue_id": left.residue_id,
                "chain_id": left.chain_id,
                "residue_number": left.residue_number,
                "insertion_code": left.insertion_code,
                "residue_code": left.residue_code,
                "sequence_index": left.sequence_index,
                "ca_x": left.ca_coord[0] if left.ca_coord is not None else math.nan,
                "ca_y": left.ca_coord[1] if left.ca_coord is not None else math.nan,
                "ca_z": left.ca_coord[2] if left.ca_coord is not None else math.nan,
                "b_factor": left.b_factor,
                "min_inter_chain_distance": min_inter_chain_distance,
                "is_interface": int(min_inter_chain_distance <= interface_cutoff),
                "local_degree": local_degree,
                "inter_chain_contacts": inter_chain_contacts,
                "solvent_proxy": solvent_proxy,
                "burial_proxy": interface_burial_proxy(
                    local_degree=local_degree,
                    inter_chain_contacts=inter_chain_contacts,
                    solvent_proxy=solvent_proxy,
                    interface_distance=(
                        min_inter_chain_distance
                        if math.isfinite(min_inter_chain_distance)
                        else None
                    ),
                ),
            }
        )
    return pd.DataFrame(node_rows), pd.DataFrame(edge_rows)


def _dssp_secondary_structure(pdb_path: Path) -> dict[str, str]:
    mkdssp_path = shutil.which("mkdssp") or shutil.which("dssp")
    if mkdssp_path is None:
        return {}
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complexvar", str(pdb_path))
    model = next(structure.get_models())
    dssp = DSSP(model, str(pdb_path), dssp=mkdssp_path)
    assignments: dict[str, str] = {}
    for key, value in dssp.property_dict.items():
        chain_id = str(key[0])
        residue_id = key[1]
        number = int(residue_id[1])
        insertion_code = (
            str(residue_id[2]).strip() if str(residue_id[2]).strip() != " " else ""
        )
        residue_key = f"{chain_id}:{number}{insertion_code}"
        ss = value[2]
        if ss in {"H", "G", "I"}:
            assignments[residue_key] = "helix"
        elif ss in {"E", "B"}:
            assignments[residue_key] = "sheet"
        else:
            assignments[residue_key] = "loop"
    return assignments


def _residue_relative_sasa(pdb_path: Path) -> dict[str, float]:
    if freesasa is None:
        return {}
    structure = freesasa.Structure(str(pdb_path))
    result = freesasa.calc(structure)
    residue_areas: dict[str, float] = {}
    for chain_id, chain_data in result.residueAreas().items():
        for residue_key, area_data in chain_data.items():
            relative_total = getattr(area_data, "relativeTotal", None)
            if relative_total is not None:
                residue_areas[f"{chain_id}:{str(residue_key).strip()}"] = float(
                    relative_total
                )
            else:
                residue_areas[f"{chain_id}:{str(residue_key).strip()}"] = float(
                    area_data.total
                )
    if not residue_areas:
        return {}
    if max(residue_areas.values()) <= 1.5:
        return residue_areas
    max_area = max(residue_areas.values())
    if max_area <= 0:
        return {}
    return {key: value / max_area for key, value in residue_areas.items()}


def map_skempi_to_structures(
    skempi_path: str | Path,
    pdb_dir: str | Path,
    output: str | Path,
) -> Path:
    frame = pd.read_csv(skempi_path, sep="\t")
    pdb_dir = Path(pdb_dir)
    rows: list[dict[str, Any]] = []
    grouped = frame.groupby("pdb_id")
    for pdb_id, pdb_frame in grouped:
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue
        residues = load_structure_residues(pdb_path)
        nodes, _ = build_contact_summary(residues)
        node_lookup = nodes.set_index("residue_id").to_dict(orient="index")
        sasa_lookup = _residue_relative_sasa(pdb_path)
        ss_lookup = _dssp_secondary_structure(pdb_path)
        residue_index = {
            (residue.chain_id, residue.residue_number, residue.insertion_code): residue
            for residue in residues
        }
        for row in pdb_frame.itertuples(index=False):
            insertion_code = ""
            if hasattr(row, "insertion_code") and pd.notna(row.insertion_code):
                insertion_code = str(row.insertion_code).strip()
            key = (
                row.mutated_chain_id,
                int(row.residue_number),
                insertion_code,
            )
            residue = residue_index.get(key)
            if residue is None:
                continue
            if residue.residue_code != row.wildtype:
                continue
            node_features = node_lookup.get(residue.residue_id, {})
            partner_chain_id = (
                row.chain_b if row.mutated_chain_id == row.chain_a else row.chain_a
            )
            rows.append(
                {
                    **row._asdict(),
                    "structure_path": str(pdb_path.resolve()),
                    "structure_kind": "experimental_complex",
                    "protein_accession": f"{row.pdb_id}:{row.mutated_chain_id}",
                    "partner_accession": f"{row.pdb_id}:{partner_chain_id}",
                    "partner_chain_id": partner_chain_id,
                    "residue_id": residue.residue_id,
                    "residue_code_observed": residue.residue_code,
                    "relative_sasa": sasa_lookup.get(residue.residue_id, math.nan),
                    "secondary_structure": ss_lookup.get(residue.residue_id, "loop"),
                    "b_factor_or_plddt": float(residue.b_factor),
                    "interface_proximal": int(node_features.get("is_interface", 0)),
                    **node_features,
                }
            )
    return write_tsv(pd.DataFrame(rows), output)


def map_accession_variants_to_burke(
    variants_path: str | Path,
    burke_manifest_path: str | Path,
    structure_root: str | Path,
    output: str | Path,
    only_human: bool = True,
) -> Path:
    def accession_root(value: Any) -> str:
        text = str(value).strip()
        if not text or text == "nan":
            return ""
        return text.split("-")[0]

    variants = pd.read_csv(variants_path, sep="\t")
    manifest = pd.read_csv(burke_manifest_path, sep="\t")
    manifest = manifest[
        manifest["is_high_confidence"] & manifest["structure_exists"]
    ].copy()
    manifest["protein_a_root"] = manifest["protein_a"].map(accession_root)
    manifest["protein_b_root"] = manifest["protein_b"].map(accession_root)
    if only_human and "organism" in variants.columns:
        variants = variants[
            variants["organism"].astype(str).str.contains("9606", na=False)
        ].copy()
    structure_root = Path(structure_root)
    rows: list[dict[str, Any]] = []
    structure_cache: dict[str, dict[str, Any]] = {}

    def load_cached_structure(structure_path: Path) -> dict[str, Any]:
        cache_key = str(structure_path.resolve())
        cached = structure_cache.get(cache_key)
        if cached is not None:
            return cached
        residues = load_structure_residues(structure_path)
        nodes, _ = build_contact_summary(residues)
        cached = {
            "residues": residues,
            "node_lookup": nodes.set_index("residue_id").to_dict(orient="index"),
            "sasa_lookup": _residue_relative_sasa(structure_path),
            "ss_lookup": _dssp_secondary_structure(structure_path),
        }
        structure_cache[cache_key] = cached
        return cached

    for variant in variants.itertuples(index=False):
        variant_accession = str(variant.protein_accession)
        variant_root = accession_root(variant_accession)
        if not variant_root:
            continue
        if not hasattr(variant, "residue_number") or pd.isna(variant.residue_number):
            continue
        matches = manifest[
            (manifest["protein_a"] == variant_accession)
            | (manifest["protein_b"] == variant_accession)
            | (manifest["protein_a_root"] == variant_root)
            | (manifest["protein_b_root"] == variant_root)
        ]
        partner_candidates: list[str] = []
        partner_text = getattr(variant, "partner_accessions", "")
        if pd.notna(partner_text) and str(partner_text).strip() not in {"", "nan"}:
            partner_candidates = [
                token for token in str(partner_text).split(";") if token
            ]
            partner_roots = {accession_root(token) for token in partner_candidates}
            matches = matches[
                (
                    (
                        (matches["protein_a"] == variant_accession)
                        | (matches["protein_a_root"] == variant_root)
                    )
                    & (
                        matches["protein_b"].isin(partner_candidates)
                        | matches["protein_b_root"].isin(partner_roots)
                    )
                )
                | (
                    (
                        (matches["protein_b"] == variant_accession)
                        | (matches["protein_b_root"] == variant_root)
                    )
                    & (
                        matches["protein_a"].isin(partner_candidates)
                        | matches["protein_a_root"].isin(partner_roots)
                    )
                )
            ]
        for match in matches.itertuples(index=False):
            structure_file = getattr(match, "structure_file", None)
            structure_path = (
                structure_root / str(structure_file)
                if structure_file
                else Path(str(match.structure_path))
            )
            if not structure_path.exists():
                continue
            cached = load_cached_structure(structure_path)
            residues = cached["residues"]
            target_chain = (
                "A" if accession_root(match.protein_a) == variant_root else "B"
            )
            candidate = next(
                (
                    residue
                    for residue in residues
                    if residue.chain_id == target_chain
                    and residue.residue_number == int(variant.residue_number)
                ),
                None,
            )
            if candidate is None:
                continue
            if candidate.residue_code != getattr(
                variant, "wildtype", candidate.residue_code
            ):
                continue
            rows.append(
                {
                    **variant._asdict(),
                    "complex_id": match.complex_id,
                    "structure_path": str(structure_path.resolve()),
                    "structure_kind": "alphafold_complex",
                    "mutated_chain_id": target_chain,
                    "partner_accession": (
                        match.protein_b if target_chain == "A" else match.protein_a
                    ),
                    "partner_chain_id": "B" if target_chain == "A" else "A",
                    "residue_id": candidate.residue_id,
                    "relative_sasa": cached["sasa_lookup"].get(
                        candidate.residue_id, math.nan
                    ),
                    "secondary_structure": cached["ss_lookup"].get(
                        candidate.residue_id, "loop"
                    ),
                    "b_factor_or_plddt": float(candidate.b_factor),
                    "interface_proximal": int(
                        cached["node_lookup"]
                        .get(candidate.residue_id, {})
                        .get("is_interface", 0)
                    ),
                    **cached["node_lookup"].get(candidate.residue_id, {}),
                }
            )
            break
    return write_tsv(pd.DataFrame(rows), output)
