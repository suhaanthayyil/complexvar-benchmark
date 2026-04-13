"""Baseline model training utilities."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error

from complexvar.constants import DEFAULT_RANDOM_SEED
from complexvar.features import mutation_descriptor
from complexvar.utils.io import write_json, write_tsv

try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:  # pragma: no cover - optional dependency
    XGBClassifier = None
    XGBRegressor = None


@dataclass(frozen=True)
class TrainingArtifacts:
    model_path: Path
    predictions_path: Path
    metadata_path: Path


def _feature_columns(
    frame: pd.DataFrame, target_column: str, split_column: str
) -> list[str]:
    preferred = [
        "min_inter_chain_distance",
        "distance_to_interface",
        "inter_chain_contacts",
        "burial_proxy",
        "solvent_proxy",
        "relative_sasa",
        "b_factor_or_plddt",
        "b_factor",
        "local_degree",
        "is_interface",
        "interface_proximal",
        "delta_charge",
        "delta_hydrophobicity",
        "delta_volume",
        "delta_polarity",
        "blosum62_score",
        "changed_to_gly",
        "changed_to_pro",
        "changed_to_cys",
        "mutation_unchanged",
        "secondary_structure_helix",
        "secondary_structure_sheet",
        "secondary_structure_loop",
    ]
    columns = []
    for column in preferred:
        if column in frame.columns and pd.api.types.is_numeric_dtype(frame[column]):
            if not frame[column].isna().all():
                columns.append(column)
    return columns


def _augment_structural_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if {"wildtype", "mutant"}.issubset(out.columns):
        if "delta_charge" not in out.columns:
            descriptors = out.apply(
                lambda row: mutation_descriptor(
                    wildtype=str(row["wildtype"]),
                    mutant=str(row["mutant"]),
                ),
                axis=1,
                result_type="expand",
            )
            out = pd.concat([out, descriptors], axis=1)
    if (
        "distance_to_interface" not in out.columns
        and "min_inter_chain_distance" in out.columns
    ):
        out["distance_to_interface"] = out["min_inter_chain_distance"]
    for column in ["min_inter_chain_distance", "distance_to_interface"]:
        if column in out.columns:
            out[column] = out[column].replace([np.inf, -np.inf], 20.0)
    numeric_columns = out.select_dtypes(include=["number"]).columns
    out.loc[:, numeric_columns] = out.loc[:, numeric_columns].replace(
        [np.inf, -np.inf],
        np.nan,
    )
    if "secondary_structure" in out.columns:
        normalized = out["secondary_structure"].fillna("loop").astype(str).str.lower()
        out["secondary_structure_helix"] = (normalized == "helix").astype(int)
        out["secondary_structure_sheet"] = (normalized == "sheet").astype(int)
        out["secondary_structure_loop"] = (normalized == "loop").astype(int)
    if "interface_proximal" not in out.columns and "is_interface" in out.columns:
        out["interface_proximal"] = out["is_interface"]
    return out


def train_structural_classifier(
    frame: pd.DataFrame,
    target_column: str,
    split_column: str = "split",
    model_name: str = "ddg_proxy_logistic",
    seed: int = DEFAULT_RANDOM_SEED,
):
    frame = _augment_structural_features(frame)
    feature_columns = _feature_columns(
        frame, target_column=target_column, split_column=split_column
    )
    train = frame[frame[split_column] == "train"].copy()
    val = frame[frame[split_column] == "val"].copy()
    test = frame[frame[split_column] == "test"].copy()
    fill_values = train[feature_columns].median(numeric_only=True).fillna(0.0)
    train.loc[:, feature_columns] = train[feature_columns].fillna(fill_values)
    val.loc[:, feature_columns] = val[feature_columns].fillna(fill_values)
    test.loc[:, feature_columns] = test[feature_columns].fillna(fill_values)

    x_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train[target_column].to_numpy(dtype=int)

    if model_name == "ddg_proxy_logistic":
        model = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
    elif (
        model_name == "struct_xgboost"
        and XGBClassifier is not None
        and os.environ.get("COMPLEXVAR_DISABLE_XGBOOST", "0") != "1"
    ):
        model = XGBClassifier(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=4,
            eval_metric="logloss",
            random_state=seed,
        )
    elif model_name == "struct_hgb":
        model = HistGradientBoostingClassifier(random_state=seed)
    else:
        model = HistGradientBoostingClassifier(random_state=seed)

    model.fit(x_train, y_train)

    predictions = []
    metadata_columns = [
        column
        for column in [
            "sample_id",
            target_column,
            "protein_group",
            "family_group",
            "source_dataset",
            "interface_proximal",
            "is_interface",
        ]
        if column in frame.columns
    ]
    for split_name, split_frame in [("train", train), ("val", val), ("test", test)]:
        x_split = split_frame[feature_columns].to_numpy(dtype=float)
        scores = model.predict_proba(x_split)[:, 1]
        local = split_frame[metadata_columns].copy()
        local["split"] = split_name
        local["score"] = scores
        predictions.append(local.rename(columns={target_column: "label"}))

    return model, pd.concat(predictions, ignore_index=True), feature_columns


def train_regression_baseline(
    frame: pd.DataFrame,
    target_column: str,
    split_column: str = "split",
    seed: int = DEFAULT_RANDOM_SEED,
):
    frame = _augment_structural_features(frame)
    feature_columns = _feature_columns(
        frame, target_column=target_column, split_column=split_column
    )
    train = frame[frame[split_column] == "train"].copy()
    val = frame[frame[split_column] == "val"].copy()
    test = frame[frame[split_column] == "test"].copy()
    fill_values = train[feature_columns].median(numeric_only=True).fillna(0.0)
    train.loc[:, feature_columns] = train[feature_columns].fillna(fill_values)
    val.loc[:, feature_columns] = val[feature_columns].fillna(fill_values)
    test.loc[:, feature_columns] = test[feature_columns].fillna(fill_values)

    x_train = train[feature_columns].to_numpy(dtype=float)
    y_train = train[target_column].to_numpy(dtype=float)

    if (
        XGBRegressor is not None
        and os.environ.get("COMPLEXVAR_DISABLE_XGBOOST", "0") != "1"
    ):
        model = XGBRegressor(
            max_depth=4,
            n_estimators=300,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            tree_method="hist",
            n_jobs=4,
            random_state=seed,
        )
    else:
        model = HistGradientBoostingRegressor(random_state=seed)

    model.fit(x_train, y_train)

    predictions = []
    metadata_columns = [
        column
        for column in [
            "sample_id",
            target_column,
            "protein_group",
            "family_group",
            "source_dataset",
            "interface_proximal",
            "is_interface",
        ]
        if column in frame.columns
    ]
    for split_name, split_frame in [("train", train), ("val", val), ("test", test)]:
        x_split = split_frame[feature_columns].to_numpy(dtype=float)
        scores = model.predict(x_split)
        local = split_frame[metadata_columns].copy()
        local["split"] = split_name
        local["prediction"] = scores
        predictions.append(local.rename(columns={target_column: "ddg"}))

    metrics = {
        "test_rmse": float(
            mean_squared_error(
                test[target_column].to_numpy(dtype=float),
                model.predict(test[feature_columns].to_numpy(dtype=float)),
                squared=False,
            )
        )
    }
    return model, pd.concat(predictions, ignore_index=True), feature_columns, metrics


def persist_training_outputs(
    model,
    predictions: pd.DataFrame,
    feature_columns: list[str],
    output_dir: str | Path,
    model_name: str,
    extra_metadata: dict | None = None,
) -> TrainingArtifacts:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    predictions_path = output_dir / "predictions.tsv"
    metadata_path = output_dir / "training_metadata.json"
    joblib.dump(model, model_path)
    write_tsv(predictions, predictions_path)
    metadata = {"model_name": model_name, "feature_columns": feature_columns}
    if extra_metadata:
        metadata.update(extra_metadata)
    write_json(metadata, metadata_path)
    return TrainingArtifacts(
        model_path=model_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
    )
