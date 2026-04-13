"""Command-line interface for ComplexVar."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from complexvar.analysis.failure_modes import high_confidence_errors
from complexvar.data.burke import build_structure_manifest, write_download_manifest
from complexvar.data.clinvar import write_filtered_clinvar
from complexvar.data.intact import write_normalized_intact
from complexvar.data.skempi import write_normalized_skempi
from complexvar.downloads import Step1Config, run_step1
from complexvar.features import mutation_descriptor
from complexvar.graphs.builder import write_graph_bundle
from complexvar.metrics.classification import (
    compute_classification_metrics,
    grouped_bootstrap,
    macro_average_by_group,
)
from complexvar.metrics.regression import compute_regression_metrics
from complexvar.models.baselines import (
    persist_training_outputs,
    train_regression_baseline,
    train_structural_classifier,
)
from complexvar.text_policy import assert_text_policy
from complexvar.utils.io import read_table, write_json, write_tsv
from complexvar.utils.splits import SplitFractions, leakage_summary, make_group_splits


def make_toy_dataset(output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for index in range(18):
        wildtype = "AILVKR"[index % 6]
        mutant = "GPCEYW"[index % 6]
        mutation = mutation_descriptor(wildtype=wildtype, mutant=mutant)
        inter_chain_contacts = float((index % 4) + (2 if index % 3 == 0 else 0))
        distance_to_interface = float((index % 7) + 0.5)
        local_degree = float(4 + (index % 5))
        solvent_proxy = float(8 - (index % 4))
        is_interface = int(distance_to_interface <= 3.5)
        signal = (
            0.9 * is_interface
            + 0.3 * inter_chain_contacts
            - 0.2 * distance_to_interface
            + 0.1 * mutation["delta_hydrophobicity"]
        )
        label = int(signal > 0.4)
        ddg = signal + 0.2 * (index % 2)
        protein_group = f"protein_{index // 3}"
        family_group = f"family_{index // 3}"
        record = {
            "sample_id": f"toy_{index:03d}",
            "protein_group": protein_group,
            "family_group": family_group,
            "wildtype": wildtype,
            "mutant": mutant,
            "distance_to_interface": distance_to_interface,
            "inter_chain_contacts": inter_chain_contacts,
            "local_degree": local_degree,
            "solvent_proxy": solvent_proxy,
            "burial_proxy": local_degree + inter_chain_contacts - solvent_proxy,
            "pLDDT": 70 + (index % 20),
            "secondary_structure_loop": int(index % 3 == 0),
            "secondary_structure_helix": int(index % 3 == 1),
            "secondary_structure_sheet": int(index % 3 == 2),
            "is_interface": is_interface,
            "label": label,
            "ddg": round(float(ddg), 3),
            **mutation,
        }
        records.append(record)

    classification = pd.DataFrame(records)
    classification = make_group_splits(
        classification,
        group_column="family_group",
        fractions=SplitFractions(train=0.67, val=0.17, test=0.16),
    )
    regression = classification.copy()

    write_tsv(classification, output_dir / "classification_features.tsv")
    write_tsv(regression, output_dir / "regression_features.tsv")
    write_tsv(
        classification[
            ["sample_id", "protein_group", "family_group", "label", "split"]
        ],
        output_dir / "toy_samples.tsv",
    )


def write_toy_manifests(input_dir: str | Path, output_dir: str | Path) -> None:
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    frame = read_table(input_dir / "classification_features.tsv")
    write_tsv(
        frame[["sample_id", "wildtype", "mutant", "protein_group", "family_group"]],
        output_dir / "toy_variant_manifest.tsv",
    )
    write_tsv(
        frame[["sample_id", "label", "is_interface"]],
        output_dir / "toy_label_manifest.tsv",
    )
    split_manifest = frame[
        ["sample_id", "protein_group", "family_group", "split"]
    ].copy()
    write_tsv(split_manifest, output_dir / "toy_split_manifest.tsv")


def build_toy_graphs(input_dir: str | Path, output: str | Path) -> None:
    input_dir = Path(input_dir)
    toy_pdb = input_dir / "toy_complex.pdb"
    pdb_lines = [
        (
            "ATOM      1  CA  ALA A   1       0.000   0.000   0.000"
            "  1.00 80.00           C"
        ),
        (
            "ATOM      2  CA  LEU A   2       2.000   0.500   0.000"
            "  1.00 81.00           C"
        ),
        (
            "ATOM      3  CA  GLY B   1       0.000   0.000   6.000"
            "  1.00 78.00           C"
        ),
        (
            "ATOM      4  CA  SER B   2       2.000   0.500   6.000"
            "  1.00 79.00           C"
        ),
        "TER",
        "END",
    ]
    toy_pdb.write_text("\n".join(pdb_lines) + "\n")
    bundle = write_graph_bundle(toy_pdb, input_dir / "graphs")
    manifest = pd.DataFrame([{"graph_bundle": str(bundle), "pdb_path": str(toy_pdb)}])
    write_tsv(manifest, output)


def _merge_features_and_splits(
    features_path: str | Path, splits_path: str | Path | None
) -> pd.DataFrame:
    frame = read_table(features_path)
    if splits_path is None:
        return frame
    splits = read_table(splits_path)
    return frame.drop(columns=["split"], errors="ignore").merge(
        splits, on=["sample_id", "protein_group", "family_group"]
    )


def train_baseline(args: argparse.Namespace) -> None:
    frame = _merge_features_and_splits(args.features, args.splits)
    model, predictions, feature_columns = train_structural_classifier(
        frame=frame,
        target_column=args.target_column,
        split_column="split",
        model_name=args.model_name,
    )
    persist_training_outputs(
        model=model,
        predictions=predictions,
        feature_columns=feature_columns,
        output_dir=args.output_dir,
        model_name=args.model_name,
    )


def train_regression(args: argparse.Namespace) -> None:
    frame = _merge_features_and_splits(args.features, args.splits)
    model, predictions, feature_columns, metrics = train_regression_baseline(
        frame=frame,
        target_column=args.target_column,
        split_column="split",
    )
    persist_training_outputs(
        model=model,
        predictions=predictions,
        feature_columns=feature_columns,
        output_dir=args.output_dir,
        model_name="regression_baseline",
        extra_metadata=metrics,
    )


def evaluate_classification(args: argparse.Namespace) -> None:
    frame = read_table(args.predictions)
    metrics = compute_classification_metrics(frame[frame["split"] == "test"])
    metrics.update(
        macro_average_by_group(frame[frame["split"] == "test"], "protein_group")
    )
    metrics.update(
        macro_average_by_group(frame[frame["split"] == "test"], "family_group")
    )
    metrics["bootstrap"] = grouped_bootstrap(
        frame[frame["split"] == "test"], "protein_group"
    )
    write_json(metrics, args.output)


def evaluate_regression(args: argparse.Namespace) -> None:
    frame = read_table(args.predictions)
    metrics = compute_regression_metrics(frame[frame["split"] == "test"])
    write_json(metrics, args.output)


def make_summary_figure(args: argparse.Namespace) -> None:
    frame = read_table(args.predictions)
    subset = frame[frame["split"] == "test"].copy()
    subset["predicted_label"] = (subset["score"] >= 0.5).astype(int)
    counts = subset.groupby(["label", "predicted_label"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", figsize=(6, 4))
    plt.title("Toy Classification Summary")
    plt.xlabel("True label")
    plt.ylabel("Count")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    plt.close()


def write_gnn_note(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "GNN training is implemented as an optional path that requires the "
        "`.[gnn]` extra and real graph data.\n",
        encoding="utf-8",
    )


def build_structure_manifest_command(args: argparse.Namespace) -> None:
    build_structure_manifest(
        summary_csv=args.summary_csv,
        output=args.output,
        pdockq_threshold=args.pdockq_threshold,
    )


def build_split_manifest(args: argparse.Namespace) -> None:
    frame = read_table(args.input)
    split_frame = make_group_splits(
        frame,
        group_column=args.group_column,
        fractions=SplitFractions(
            train=args.train_fraction, val=args.val_fraction, test=args.test_fraction
        ),
    )
    write_tsv(split_frame, args.output)


def check_leakage(args: argparse.Namespace) -> None:
    frame = read_table(args.input)
    summary = leakage_summary(
        frame, split_column="split", fields=args.fields.split(",")
    )
    write_json(summary, args.output)


def failure_audit(args: argparse.Namespace) -> None:
    frame = read_table(args.predictions)
    out = high_confidence_errors(frame, threshold=args.threshold)
    write_tsv(out, args.output)


def download_step1_command(args: argparse.Namespace) -> None:
    datasets = [item.strip() for item in args.datasets.split(",") if item.strip()]
    config = Step1Config(
        datasets=datasets,
        root=Path(args.root),
        skip_existing=args.skip_existing,
        force=args.force,
        extract_high_confidence_only=args.extract_high_confidence_only,
        pdb_limit=args.pdb_limit,
        monomer_limit=args.monomer_limit,
        workers=args.workers,
    )
    summary = run_step1(config)
    if args.output:
        write_json(summary, args.output)


def check_text_policy_command(args: argparse.Namespace) -> None:
    assert_text_policy(Path(args.root))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="complexvar")
    subparsers = parser.add_subparsers(dest="command", required=True)

    make_toy = subparsers.add_parser("make-toy-dataset")
    make_toy.add_argument("--output-dir", required=True)
    make_toy.set_defaults(func=lambda args: make_toy_dataset(args.output_dir))

    write_downloads = subparsers.add_parser("write-download-manifest")
    write_downloads.add_argument("--output", required=True)
    write_downloads.set_defaults(func=lambda args: write_download_manifest(args.output))

    step1 = subparsers.add_parser("download-step1")
    step1.add_argument("--datasets", required=True)
    step1.add_argument("--root", default=".")
    step1.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    step1.add_argument("--force", action="store_true")
    step1.add_argument(
        "--extract-high-confidence-only",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    step1.add_argument("--pdb-limit", type=int)
    step1.add_argument("--monomer-limit", type=int)
    step1.add_argument("--workers", type=int, default=4)
    step1.add_argument("--output")
    step1.set_defaults(func=download_step1_command)

    text_policy = subparsers.add_parser("check-text-policy")
    text_policy.add_argument("--root", default=".")
    text_policy.set_defaults(func=check_text_policy_command)

    toy_manifests = subparsers.add_parser("write-toy-manifests")
    toy_manifests.add_argument("--input-dir", required=True)
    toy_manifests.add_argument("--output-dir", required=True)
    toy_manifests.set_defaults(
        func=lambda args: write_toy_manifests(args.input_dir, args.output_dir)
    )

    toy_graphs = subparsers.add_parser("build-toy-graphs")
    toy_graphs.add_argument("--input-dir", required=True)
    toy_graphs.add_argument("--output", required=True)
    toy_graphs.set_defaults(
        func=lambda args: build_toy_graphs(args.input_dir, args.output)
    )

    train = subparsers.add_parser("train-baseline")
    train.add_argument("--features", required=True)
    train.add_argument("--splits")
    train.add_argument("--target-column", default="label")
    train.add_argument("--group-column", default="protein_group")
    train.add_argument("--model-name", default="ddg_proxy_logistic")
    train.add_argument("--output-dir", required=True)
    train.set_defaults(func=train_baseline)

    regression = subparsers.add_parser("train-regression-baseline")
    regression.add_argument("--features", required=True)
    regression.add_argument("--splits")
    regression.add_argument("--target-column", default="ddg")
    regression.add_argument("--output-dir", required=True)
    regression.set_defaults(func=train_regression)

    eval_cls = subparsers.add_parser("evaluate-classification")
    eval_cls.add_argument("--predictions", required=True)
    eval_cls.add_argument("--output", required=True)
    eval_cls.set_defaults(func=evaluate_classification)

    eval_reg = subparsers.add_parser("evaluate-regression")
    eval_reg.add_argument("--predictions", required=True)
    eval_reg.add_argument("--output", required=True)
    eval_reg.set_defaults(func=evaluate_regression)

    figure = subparsers.add_parser("make-summary-figure")
    figure.add_argument("--predictions", required=True)
    figure.add_argument("--output", required=True)
    figure.set_defaults(func=make_summary_figure)

    gnn_note = subparsers.add_parser("write-gnn-note")
    gnn_note.add_argument("--output", required=True)
    gnn_note.set_defaults(func=write_gnn_note)

    structure_manifest = subparsers.add_parser("build-structure-manifest")
    structure_manifest.add_argument("--summary-csv", required=True)
    structure_manifest.add_argument("--output", required=True)
    structure_manifest.add_argument("--pdockq-threshold", type=float, default=0.5)
    structure_manifest.set_defaults(func=build_structure_manifest_command)

    clinvar = subparsers.add_parser("filter-clinvar")
    clinvar.add_argument("--input", required=True)
    clinvar.add_argument("--output", required=True)
    clinvar.set_defaults(
        func=lambda args: write_filtered_clinvar(args.input, args.output)
    )

    skempi = subparsers.add_parser("normalize-skempi")
    skempi.add_argument("--input", required=True)
    skempi.add_argument("--output", required=True)
    skempi.set_defaults(
        func=lambda args: write_normalized_skempi(args.input, args.output)
    )

    intact = subparsers.add_parser("normalize-intact")
    intact.add_argument("--input", required=True)
    intact.add_argument("--output", required=True)
    intact.set_defaults(
        func=lambda args: write_normalized_intact(args.input, args.output)
    )

    splits = subparsers.add_parser("build-split-manifest")
    splits.add_argument("--input", required=True)
    splits.add_argument("--output", required=True)
    splits.add_argument("--group-column", default="family_group")
    splits.add_argument("--train-fraction", type=float, default=0.7)
    splits.add_argument("--val-fraction", type=float, default=0.15)
    splits.add_argument("--test-fraction", type=float, default=0.15)
    splits.set_defaults(func=build_split_manifest)

    leakage = subparsers.add_parser("check-leakage")
    leakage.add_argument("--input", required=True)
    leakage.add_argument("--fields", default="protein_group,family_group")
    leakage.add_argument("--output", required=True)
    leakage.set_defaults(func=check_leakage)

    audit = subparsers.add_parser("failure-audit")
    audit.add_argument("--predictions", required=True)
    audit.add_argument("--threshold", type=float, default=0.8)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=failure_audit)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
