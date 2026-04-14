#!/usr/bin/env python3
"""External validation using MaveDB + HuRI network.

This script implements external validation for ComplexVar following
Dr. Frederick Roth's recommendations:
1. Download DMS maps from MaveDB for Homo sapiens
2. Download HuRI binary interaction network
3. Intersect: genes with both DMS data AND binary interactions
4. Map variants to AlphaFold complex structures
5. Run trained complex GNN model
6. Compute validation metrics
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from Bio.PDB import PDBParser, NeighborSearch
from scipy import stats
from sklearn.metrics import roc_auc_score, roc_curve

from complexvar.graphs.builder import build_full_graph_object, build_variant_subgraph
from complexvar.models.gnn import ComplexVarGAT

try:
    import torch
    from torch_geometric.loader import DataLoader
except ImportError:
    raise RuntimeError("torch and torch-geometric required")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# Amino acid three-letter to one-letter code
AA_CODE = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}


def download_mavedb_scoresets(output_dir: Path) -> pd.DataFrame:
    """Download published DMS score sets from MaveDB API."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading MaveDB score sets for Homo sapiens...")

    url = "https://api.mavedb.org/api/v1/score-sets/"
    params = {"target_organism_name": "Homo sapiens", "limit": 1000}

    response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    data = response.json()

    score_sets = []
    for item in data.get("results", []):
        if item.get("publishedDate") is None:
            continue

        target = item.get("target", {})
        target_type = target.get("targetType", "")

        if "protein" not in target_type.lower():
            continue

        target_genes = target.get("targetGenes", [])
        external_ids = target.get("externalIdentifiers", [])

        gene_name = None
        uniprot_id = None

        for gene in target_genes:
            if gene.get("name"):
                gene_name = gene["name"]
                break

        for ext_id in external_ids:
            if ext_id.get("dbName", "").lower() == "uniprot":
                uniprot_id = ext_id.get("identifier")
                break

        if gene_name or uniprot_id:
            score_sets.append({
                "scoreset_id": item.get("urn"),
                "title": item.get("title"),
                "gene_name": gene_name,
                "uniprot_id": uniprot_id,
                "target_type": target_type,
                "num_variants": item.get("numVariants", 0),
                "published_date": item.get("publishedDate"),
            })

    df = pd.DataFrame(score_sets)
    logger.info(f"Found {len(df)} published score sets for Homo sapiens")

    output_path = output_dir / "mavedb_scoresets.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"Saved to {output_path}")

    return df


def download_huri_network(output_dir: Path) -> pd.DataFrame:
    """Download HuRI binary protein interaction network."""
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading HuRI network...")
    url = "http://www.interactome-atlas.org/data/HuRI.tsv"

    response = requests.get(url, timeout=120)
    response.raise_for_status()

    from io import StringIO
    df = pd.read_csv(StringIO(response.text), sep="\t")

    logger.info(f"Downloaded {len(df)} interactions")

    output_path = output_dir / "huri_network.tsv"
    df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved to {output_path}")

    return df


def intersect_mavedb_huri(
    mavedb_df: pd.DataFrame,
    huri_df: pd.DataFrame,
    burke_df: pd.DataFrame,
    output_dir: Path
) -> pd.DataFrame:
    """Find genes with DMS data, HuRI interactions, AND Burke structures."""

    logger.info("Intersecting MaveDB + HuRI + Burke structures...")

    # Get UniProt IDs from all sources
    mavedb_uniprot = set(mavedb_df["uniprot_id"].dropna())

    # Burke structures (id1 and id2 are UniProt IDs)
    burke_uniprot = set()
    burke_uniprot.update(burke_df["id1"].dropna())
    burke_uniprot.update(burke_df["id2"].dropna())

    # For HuRI, we need to map Ensembl to UniProt via Burke
    # (Burke has both Ensembl and UniProt mappings)
    huri_genes = set()
    for col in huri_df.columns[:2]:
        huri_genes.update(huri_df[col].dropna())

    # Find UniProts that are in MaveDB AND have Burke structures
    intersection = mavedb_uniprot & burke_uniprot

    logger.info(f"Found {len(intersection)} genes with DMS data AND Burke structures")

    # Build result dataframe
    result_rows = []
    for uniprot_id in intersection:
        mavedb_info = mavedb_df[mavedb_df["uniprot_id"] == uniprot_id].iloc[0]

        # Count Burke structures for this protein
        n_structures = len(burke_df[
            (burke_df["id1"] == uniprot_id) | (burke_df["id2"] == uniprot_id)
        ])

        # Check if protein has HuRI interactions (via Burke Ensembl mapping)
        burke_rows = burke_df[
            (burke_df["id1"] == uniprot_id) | (burke_df["id2"] == uniprot_id)
        ]

        has_huri = False
        for _, row in burke_rows.iterrows():
            ensg1 = row.get("ENSG_1(HURI)", "")
            ensg2 = row.get("ENSG_2(HURI)", "")
            if ensg1 in huri_genes or ensg2 in huri_genes:
                has_huri = True
                break

        result_rows.append({
            "gene_name": mavedb_info["gene_name"],
            "uniprot_id": uniprot_id,
            "scoreset_id": mavedb_info["scoreset_id"],
            "num_variants": mavedb_info["num_variants"],
            "num_burke_structures": n_structures,
            "has_huri_interaction": has_huri,
        })

    result_df = pd.DataFrame(result_rows)

    # Filter to those with HuRI interactions
    result_df = result_df[result_df["has_huri_interaction"]].copy()

    logger.info(f"Final intersection: {len(result_df)} genes with DMS + Burke + HuRI")

    output_path = output_dir / "mavedb_huri_burke_intersection.csv"
    result_df.to_csv(output_path, index=False)

    return result_df


def download_variant_scores(scoreset_id: str) -> pd.DataFrame | None:
    """Download variant scores for a MaveDB score set."""

    logger.info(f"Downloading scores for {scoreset_id}...")

    url = f"https://api.mavedb.org/api/v1/score-sets/{scoreset_id}/scores/"
    params = {"limit": 10000}

    try:
        response = requests.get(url, params=params, timeout=120)
        response.raise_for_status()
        data = response.json()

        scores = []
        for item in data.get("results", []):
            hgvs_pro = item.get("hgvsProTargetRef")
            score = item.get("score")

            if hgvs_pro and score is not None:
                scores.append({"hgvs_pro": hgvs_pro, "score": score})

        if not scores:
            logger.warning(f"No scores for {scoreset_id}")
            return None

        df = pd.DataFrame(scores)
        logger.info(f"Downloaded {len(df)} variants")
        return df

    except Exception as e:
        logger.error(f"Failed to download {scoreset_id}: {e}")
        return None


def parse_hgvs_protein(hgvs: str) -> dict[str, Any] | None:
    """Parse HGVS protein notation.

    Example: p.Gly123Asp -> {position: 123, wt: G, mut: D}
    """
    pattern = r"p\.([A-Z][a-z]{2})?(\d+)([A-Z][a-z]{2})?"
    match = re.search(pattern, hgvs)

    if not match:
        return None

    wt_long = match.group(1)
    pos = int(match.group(2))
    mut_long = match.group(3)

    wt = AA_CODE.get(wt_long, wt_long) if wt_long else None
    mut = AA_CODE.get(mut_long, mut_long) if mut_long else None

    return {"position": pos, "wt": wt, "mut": mut}


def compute_interface_distance(
    pdb_path: Path,
    chain: str,
    position: int,
    partner_chain: str,
) -> float | None:
    """Compute minimum heavy-atom distance from residue to partner chain."""

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("complex", pdb_path)
    except Exception as e:
        logger.warning(f"Failed to parse {pdb_path}: {e}")
        return None

    # Get target residue
    target_residue = None
    for model in structure:
        for ch in model:
            if ch.id == chain:
                for res in ch:
                    if res.id[1] == position:
                        target_residue = res
                        break

    if target_residue is None:
        return None

    # Get partner chain atoms
    partner_atoms = []
    for model in structure:
        for ch in model:
            if ch.id == partner_chain:
                for res in ch:
                    partner_atoms.extend(list(res.get_atoms()))

    if not partner_atoms:
        return None

    # Compute minimum distance
    min_dist = float('inf')
    for atom1 in target_residue.get_atoms():
        for atom2 in partner_atoms:
            dist = atom1 - atom2
            min_dist = min(min_dist, dist)

    return min_dist if min_dist != float('inf') else None


def map_and_build_graphs(
    intersect_df: pd.DataFrame,
    burke_df: pd.DataFrame,
    structure_dir: Path,
    graph_dir: Path,
) -> pd.DataFrame:
    """Map variants to structures and build graph files."""

    logger.info("Mapping variants to structures and building graphs...")
    graph_dir.mkdir(parents=True, exist_ok=True)

    all_variants = []

    for idx, row in intersect_df.iterrows():
        uniprot_id = row["uniprot_id"]
        scoreset_id = row["scoreset_id"]

        # Download DMS scores
        scores_df = download_variant_scores(scoreset_id)
        if scores_df is None:
            continue

        # Find Burke structures
        burke_structures = burke_df[
            ((burke_df["id1"] == uniprot_id) | (burke_df["id2"] == uniprot_id)) &
            (burke_df["is_high_confidence"] == True)
        ]

        if len(burke_structures) == 0:
            logger.warning(f"No high-confidence structures for {uniprot_id}")
            continue

        logger.info(f"Processing {row['gene_name']} ({uniprot_id}): "
                   f"{len(scores_df)} variants, {len(burke_structures)} structures")

        # Process variants
        for _, variant_row in scores_df.iterrows():
            parsed = parse_hgvs_protein(variant_row["hgvs_pro"])
            if parsed is None:
                continue

            position = parsed["position"]
            wt_aa = parsed["wt"]
            mut_aa = parsed["mut"]

            if wt_aa is None or mut_aa is None:
                continue

            # Map to each structure
            for _, struct_row in burke_structures.iterrows():
                pdb_filename = struct_row["structure_file"]
                pdb_path = structure_dir / pdb_filename

                if not pdb_path.exists():
                    continue

                # Determine chains
                is_chain_a = struct_row["id1"] == uniprot_id
                mutant_chain = "A" if is_chain_a else "B"
                partner_chain = "B" if is_chain_a else "A"

                # Check interface distance
                interface_dist = compute_interface_distance(
                    pdb_path, mutant_chain, position, partner_chain
                )

                if interface_dist is None:
                    continue

                is_interface_proximal = interface_dist <= 10.0

                # Build graph
                try:
                    # Build full graph for the structure
                    full_graph = build_full_graph_object(
                        pdb_path,
                        edge_cutoff=8.0,
                        interface_cutoff=10.0,
                    )

                    # Build variant subgraph
                    residue_id = f"{mutant_chain}:{position}"
                    graph_filename = (
                        f"{row['gene_name']}_{uniprot_id}_"
                        f"{position}{wt_aa}{mut_aa}_{struct_row['unique_ID']}.pt"
                    )
                    graph_path = graph_dir / graph_filename

                    build_variant_subgraph(
                        full_graph_bundle=full_graph,
                        residue_id=residue_id,
                        wildtype=wt_aa,
                        mutant=mut_aa,
                        output_path=graph_path,
                        radius_angstrom=15.0,
                        monomer_only=False,
                    )

                    all_variants.append({
                        "gene_name": row["gene_name"],
                        "uniprot_id": uniprot_id,
                        "scoreset_id": scoreset_id,
                        "position": position,
                        "wt": wt_aa,
                        "mut": mut_aa,
                        "dms_score": variant_row["score"],
                        "structure_id": struct_row["unique_ID"],
                        "interface_distance": interface_dist,
                        "is_interface_proximal": is_interface_proximal,
                        "graph_path": str(graph_path),
                    })

                except Exception as e:
                    logger.warning(f"Failed to build graph for {residue_id}: {e}")
                    continue

    variants_df = pd.DataFrame(all_variants)

    if len(variants_df) > 0:
        logger.info(f"Built {len(variants_df)} variant graphs")
        logger.info(f"  Interface-proximal (<10Å): "
                   f"{variants_df['is_interface_proximal'].sum()}")

        output_path = graph_dir.parent / "mapped_variants.csv"
        variants_df.to_csv(output_path, index=False)

    return variants_df


def label_variants_from_dms(
    variants_df: pd.DataFrame,
    percentile_threshold: float = 20.0
) -> pd.DataFrame:
    """Label variants as disruptive (bottom 20% of DMS scores)."""

    logger.info(f"Labeling variants using {percentile_threshold}th percentile...")

    labeled_variants = []

    for scoreset_id in variants_df["scoreset_id"].unique():
        subset = variants_df[variants_df["scoreset_id"] == scoreset_id].copy()

        threshold = np.percentile(subset["dms_score"], percentile_threshold)
        subset["is_disruptive"] = subset["dms_score"] <= threshold

        labeled_variants.append(subset)

    result_df = pd.concat(labeled_variants, ignore_index=True)

    n_disruptive = result_df["is_disruptive"].sum()
    logger.info(f"Labeled {n_disruptive} / {len(result_df)} as disruptive")

    return result_df


def run_model_predictions(
    variants_df: pd.DataFrame,
    model_checkpoint: Path,
) -> pd.DataFrame:
    """Run complex GNN on external validation set."""

    logger.info("Loading model...")

    # Load example graph to get dimensions
    example_path = variants_df.iloc[0]["graph_path"]
    example = torch.load(example_path, map_location="cpu", weights_only=False)

    model = ComplexVarGAT(
        node_dim=int(example.x.shape[1]),
        edge_dim=int(example.edge_attr.shape[1]),
        perturbation_dim=int(example.perturbation.shape[-1]),
    )

    state_dict = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()

    logger.info("Running predictions...")

    # Load all graphs
    samples = []
    for idx, row in variants_df.iterrows():
        try:
            data = torch.load(row["graph_path"], map_location="cpu", weights_only=False)
            data.variant_idx = idx
            samples.append(data)
        except Exception as e:
            logger.warning(f"Failed to load graph {idx}: {e}")

    # Run in batches
    loader = DataLoader(samples, batch_size=32, shuffle=False)
    predictions = {}

    with torch.no_grad():
        for batch in loader:
            outputs = model(batch)
            probs = torch.sigmoid(outputs["classification"]).cpu().numpy()
            ddgs = outputs["regression"].cpu().numpy()

            # Map back to original indices
            for i in range(len(batch.variant_idx)):
                idx = batch.variant_idx[i].item()
                predictions[idx] = {
                    "predicted_prob": float(probs[i]),
                    "predicted_ddg": float(ddgs[i]),
                }

    # Add predictions to dataframe
    result_df = variants_df.copy()
    result_df["predicted_prob"] = result_df.index.map(
        lambda i: predictions.get(i, {}).get("predicted_prob")
    )
    result_df["predicted_ddg"] = result_df.index.map(
        lambda i: predictions.get(i, {}).get("predicted_ddg")
    )

    logger.info(f"Generated {len(predictions)} predictions")

    return result_df


def compute_validation_metrics(
    predictions_df: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Compute AUROC and validation metrics."""

    logger.info("Computing validation metrics...")

    # Filter to interface-proximal with valid predictions
    interface_df = predictions_df[
        predictions_df["is_interface_proximal"] &
        predictions_df["predicted_prob"].notna()
    ].copy()

    if len(interface_df) == 0:
        logger.error("No valid interface-proximal predictions!")
        return {}

    y_true = interface_df["is_disruptive"].astype(int).values
    y_pred = interface_df["predicted_prob"].values

    # Compute AUROC
    auroc = roc_auc_score(y_true, y_pred)

    # Bootstrap 95% CI
    n_bootstrap = 1000
    rng = np.random.default_rng(42)
    aurocs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(len(y_true), size=len(y_true), replace=True)
        try:
            boot_auroc = roc_auc_score(y_true[indices], y_pred[indices])
            aurocs.append(boot_auroc)
        except:
            continue

    ci_lower = np.percentile(aurocs, 2.5)
    ci_upper = np.percentile(aurocs, 97.5)

    # Statistics
    n_genes = predictions_df["gene_name"].nunique()
    n_total = len(predictions_df)
    n_interface = len(interface_df)
    n_disruptive = y_true.sum()

    metrics = {
        "n_genes": int(n_genes),
        "n_total_variants": int(n_total),
        "n_interface_proximal_variants": int(n_interface),
        "n_disruptive_variants": int(n_disruptive),
        "auroc": float(auroc),
        "auroc_ci_lower": float(ci_lower),
        "auroc_ci_upper": float(ci_upper),
        "interpretation": "external_validation_mavedb_huri",
        "validation_dataset": "MaveDB DMS + HuRI binary interactions",
        "interface_threshold_angstroms": 10.0,
        "dms_disruptive_threshold_percentile": 20.0,
    }

    logger.info(f"\nExternal Validation Results:")
    logger.info(f"  Genes: {n_genes}")
    logger.info(f"  Total variants: {n_total}")
    logger.info(f"  Interface-proximal: {n_interface}")
    logger.info(f"  Disruptive: {n_disruptive}")
    logger.info(f"  AUROC: {auroc:.3f} (95% CI: [{ci_lower:.3f}, {ci_upper:.3f}])")

    # Save
    output_path = output_dir / "mavedb_huri_validation.json"
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Saved to {output_path}")

    return metrics


def main():
    """Main external validation workflow."""

    # Paths
    data_dir = Path("data/external_validation")
    results_dir = Path("results/external_validation")
    results_dir.mkdir(parents=True, exist_ok=True)

    structure_dir = Path("data/raw/burke/pdbs/high_confidence")
    burke_csv = Path("data/raw/burke/table_AF2_HURI_HuMap_UNIQUE.csv")
    model_checkpoint = Path("results/skempi/complex_gnn/best_model.pt")

    # Load Burke structures
    logger.info("Loading Burke structure manifest...")
    burke_df = pd.read_csv(burke_csv)

    # Add high-confidence flag
    burke_df["pDockQ"] = pd.to_numeric(burke_df["pDockQ"], errors="coerce")
    burke_df["is_high_confidence"] = burke_df["pDockQ"] > 0.5

    logger.info(f"Loaded {len(burke_df)} Burke structures, "
               f"{burke_df['is_high_confidence'].sum()} high-confidence")

    # Step 1: Download MaveDB
    mavedb_df = download_mavedb_scoresets(data_dir)

    # Step 2: Download HuRI
    huri_df = download_huri_network(data_dir)

    # Step 3: Intersect
    intersect_df = intersect_mavedb_huri(mavedb_df, huri_df, burke_df, data_dir)

    logger.info(f"\nTop genes by structure count:")
    top = intersect_df.nlargest(10, "num_burke_structures")
    for _, row in top.iterrows():
        logger.info(f"  {row['gene_name']}: {row['num_burke_structures']} structures")

    # Step 4: Map variants and build graphs
    graph_dir = results_dir / "graphs"
    variants_df = map_and_build_graphs(
        intersect_df, burke_df, structure_dir, graph_dir
    )

    if len(variants_df) == 0:
        logger.error("No variants mapped! Exiting.")
        return {}

    # Step 5: Label variants
    labeled_df = label_variants_from_dms(variants_df, percentile_threshold=20.0)

    # Step 6: Run model
    predictions_df = run_model_predictions(labeled_df, model_checkpoint)

    # Step 7: Compute metrics
    metrics = compute_validation_metrics(predictions_df, results_dir)

    logger.info("\n" + "="*60)
    logger.info("EXTERNAL VALIDATION COMPLETE")
    logger.info("="*60)

    return metrics


if __name__ == "__main__":
    main()
