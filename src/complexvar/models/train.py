"""Torch training utilities for ComplexVar."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from complexvar.utils.io import ensure_parent, write_json, write_tsv

try:
    import torch
    from torch import nn
    from torch.optim import AdamW
    from torch.optim.lr_scheduler import CosineAnnealingLR
    from torch.utils.data import DataLoader as TorchDataLoader
    from torch.utils.data import TensorDataset
except ImportError:  # pragma: no cover
    torch = None
    nn = None
    AdamW = None
    CosineAnnealingLR = None
    TorchDataLoader = None
    TensorDataset = None

try:
    from torch_geometric.loader import DataLoader as GraphDataLoader
except ImportError:  # pragma: no cover
    GraphDataLoader = None

from complexvar.metrics.classification import compute_classification_metrics


@dataclass(frozen=True)
class TrainingOutputs:
    checkpoint_path: Path
    log_path: Path
    predictions_path: Path
    metadata_path: Path


def masked_multitask_loss(
    outputs,
    classification_targets,
    regression_targets,
    classification_mask,
    regression_mask,
    cls_weight: float = 0.5,
    reg_weight: float = 0.5,
):
    if torch is None or nn is None:
        raise RuntimeError("torch is required for multitask loss")
    bce = nn.BCEWithLogitsLoss(reduction="none")
    mse = nn.MSELoss(reduction="none")
    cls_loss = torch.tensor(0.0, device=classification_targets.device)
    reg_loss = torch.tensor(0.0, device=classification_targets.device)
    if classification_mask.any():
        cls_values = bce(outputs["classification"], classification_targets.float())
        cls_loss = cls_values[classification_mask].mean()
    if regression_mask.any():
        reg_values = mse(outputs["regression"], regression_targets.float())
        reg_loss = reg_values[regression_mask].mean()
    return cls_weight * cls_loss + reg_weight * reg_loss, cls_loss, reg_loss


def _device() -> str:
    return "cpu"


def _load_graph_samples(manifest: pd.DataFrame) -> list[Any]:
    if torch is None:
        raise RuntimeError("torch is required to load graph samples")
    samples = []
    for row in manifest.itertuples(index=False):
        data = torch.load(row.graph_path, map_location="cpu", weights_only=False)
        data.sample_id = getattr(row, "sample_id", "")
        data.source_dataset = getattr(row, "source_dataset", "")
        interface_value = getattr(
            row,
            "interface_proximal",
            getattr(row, "is_interface", 0),
        )
        data.interface_proximal = int(interface_value)
        data.protein_group = getattr(row, "protein_group", "")
        data.family_group = getattr(row, "family_group", "")
        binary_label = (
            row.binary_label if hasattr(row, "binary_label") else float("nan")
        )
        ddg_value = row.ddg if hasattr(row, "ddg") else float("nan")
        data.classification_label = (
            float(binary_label) if pd.notna(binary_label) else 0.0
        )
        data.regression_label = float(ddg_value) if pd.notna(ddg_value) else 0.0
        data.classification_mask = int(pd.notna(binary_label))
        data.regression_mask = int(pd.notna(ddg_value))
        samples.append(data)
    return samples


def _evaluate_graph_model(model, loader) -> tuple[dict[str, float], pd.DataFrame]:
    if torch is None:
        raise RuntimeError("torch is required for evaluation")
    model.eval()
    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(_device())
            outputs = model(batch)
            scores = torch.sigmoid(outputs["classification"]).detach().cpu().numpy()
            regression = outputs["regression"].detach().cpu().numpy()
            batch_size = len(scores)
            sample_ids = getattr(batch, "sample_id", [""] * batch_size)
            if isinstance(sample_ids, str):
                sample_ids = [sample_ids]
            source_dataset = getattr(batch, "source_dataset", [""] * batch_size)
            interface_proximal = getattr(batch, "interface_proximal", [0] * batch_size)
            protein_group = getattr(batch, "protein_group", [""] * batch_size)
            family_group = getattr(batch, "family_group", [""] * batch_size)
            labels = batch.classification_label.detach().cpu().numpy()
            ddg = batch.regression_label.detach().cpu().numpy()
            cls_mask = batch.classification_mask.detach().cpu().numpy()
            reg_mask = batch.regression_mask.detach().cpu().numpy()
            for index in range(batch_size):
                rows.append(
                    {
                        "sample_id": sample_ids[index],
                        "source_dataset": source_dataset[index],
                        "interface_proximal": int(interface_proximal[index]),
                        "protein_group": protein_group[index],
                        "family_group": family_group[index],
                        "label": (
                            float(labels[index]) if cls_mask[index] else float("nan")
                        ),
                        "score": float(scores[index]),
                        "prediction": (
                            float(regression[index])
                            if reg_mask[index]
                            else float("nan")
                        ),
                        "ddg": float(ddg[index]) if reg_mask[index] else float("nan"),
                    }
                )
    predictions = pd.DataFrame(rows)
    metrics = {}
    cls_eval = predictions.dropna(subset=["label"]).copy()
    if not cls_eval.empty and cls_eval["label"].nunique() >= 2:
        metrics = compute_classification_metrics(cls_eval)
    return metrics, predictions


def train_graph_model(
    train_manifest: pd.DataFrame,
    val_manifest: pd.DataFrame,
    test_manifest: pd.DataFrame | None,
    model,
    output_dir: str | Path,
    batch_size: int = 64,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
) -> TrainingOutputs:
    if (
        torch is None
        or GraphDataLoader is None
        or AdamW is None
        or CosineAnnealingLR is None
    ):
        raise RuntimeError("torch and torch-geometric are required for graph training")

    train_samples = _load_graph_samples(train_manifest)
    val_samples = _load_graph_samples(val_manifest)
    train_loader = GraphDataLoader(train_samples, batch_size=batch_size, shuffle=True)
    val_loader = GraphDataLoader(val_samples, batch_size=batch_size, shuffle=False)
    test_loader = None
    if test_manifest is not None and not test_manifest.empty:
        test_loader = GraphDataLoader(
            _load_graph_samples(test_manifest),
            batch_size=batch_size,
            shuffle=False,
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(_device())
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    stale_epochs = 0
    epoch_rows: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(_device())
            optimizer.zero_grad()
            outputs = model(batch)
            loss, cls_loss, reg_loss = masked_multitask_loss(
                outputs=outputs,
                classification_targets=batch.classification_label,
                regression_targets=batch.regression_label,
                classification_mask=batch.classification_mask.bool(),
                regression_mask=batch.regression_mask.bool(),
            )
            loss.backward()
            optimizer.step()
            train_losses.append(
                {
                    "loss": float(loss.detach().cpu()),
                    "classification_loss": float(cls_loss.detach().cpu()),
                    "regression_loss": float(reg_loss.detach().cpu()),
                }
            )
        scheduler.step()
        val_metrics, val_predictions = _evaluate_graph_model(model, val_loader)
        val_predictions["split"] = "val"
        val_auroc = float(val_metrics.get("auroc", float("nan")))
        epoch_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(pd.DataFrame(train_losses)["loss"].mean()),
                "train_classification_loss": float(
                    pd.DataFrame(train_losses)["classification_loss"].mean()
                ),
                "train_regression_loss": float(
                    pd.DataFrame(train_losses)["regression_loss"].mean()
                ),
                "val_auroc": val_auroc,
                "val_auprc": float(val_metrics.get("auprc", float("nan"))),
            }
        )
        if pd.notna(val_auroc) and val_auroc > best_score:
            best_score = val_auroc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            write_tsv(val_predictions, output_dir / "val_predictions.tsv")
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    checkpoint_path = ensure_parent(output_dir / "best_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    log_path = write_tsv(pd.DataFrame(epoch_rows), output_dir / "training_log.tsv")
    val_metrics, predictions = _evaluate_graph_model(model, val_loader)
    predictions["split"] = "val"
    if test_loader is not None:
        _, test_predictions = _evaluate_graph_model(model, test_loader)
        test_predictions["split"] = "test"
        predictions = pd.concat([predictions, test_predictions], ignore_index=True)
    predictions_path = write_tsv(predictions, output_dir / "predictions.tsv")
    metadata_path = write_json(
        {
            "best_val_auroc": best_score,
            "epochs_completed": len(epoch_rows),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "val_metrics": val_metrics,
        },
        output_dir / "training_metadata.json",
    )
    return TrainingOutputs(
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
    )


def _evaluate_tabular_model(
    model,
    loader,
    split_name: str,
    metadata: pd.DataFrame | None = None,
) -> tuple[dict[str, float], pd.DataFrame]:
    if torch is None:
        raise RuntimeError("torch is required for evaluation")
    model.eval()
    rows: list[dict[str, Any]] = []
    row_offset = 0
    with torch.no_grad():
        for features, cls_labels, reg_labels, cls_mask, reg_mask in loader:
            features = features.to(_device())
            outputs = model(features)
            scores = torch.sigmoid(outputs["classification"]).detach().cpu().numpy()
            regression = outputs["regression"].detach().cpu().numpy()
            for index in range(len(scores)):
                meta = {}
                if metadata is not None and row_offset + index < len(metadata):
                    meta = metadata.iloc[row_offset + index].to_dict()
                rows.append(
                    {
                        **meta,
                        "split": split_name,
                        "label": (
                            float(cls_labels[index])
                            if cls_mask[index]
                            else float("nan")
                        ),
                        "score": float(scores[index]),
                        "prediction": (
                            float(regression[index])
                            if reg_mask[index]
                            else float("nan")
                        ),
                        "ddg": (
                            float(reg_labels[index])
                            if reg_mask[index]
                            else float("nan")
                        ),
                    }
                )
            row_offset += len(scores)
    predictions = pd.DataFrame(rows)
    metrics = {}
    cls_eval = predictions.dropna(subset=["label"]).copy()
    if not cls_eval.empty and cls_eval["label"].nunique() >= 2:
        metrics = compute_classification_metrics(cls_eval)
    return metrics, predictions


def train_tabular_model(
    train_features,
    val_features,
    test_features,
    train_cls,
    val_cls,
    test_cls,
    train_reg,
    val_reg,
    test_reg,
    train_cls_mask,
    val_cls_mask,
    test_cls_mask,
    train_reg_mask,
    val_reg_mask,
    test_reg_mask,
    model,
    output_dir: str | Path,
    batch_size: int = 64,
    epochs: int = 200,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 20,
    val_metadata: pd.DataFrame | None = None,
    test_metadata: pd.DataFrame | None = None,
) -> TrainingOutputs:
    if torch is None or TensorDataset is None or TorchDataLoader is None:
        raise RuntimeError("torch is required for tabular training")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = TensorDataset(
        train_features, train_cls, train_reg, train_cls_mask, train_reg_mask
    )
    val_dataset = TensorDataset(
        val_features, val_cls, val_reg, val_cls_mask, val_reg_mask
    )
    test_dataset = TensorDataset(
        test_features, test_cls, test_reg, test_cls_mask, test_reg_mask
    )
    train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    model = model.to(_device())
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_state = copy.deepcopy(model.state_dict())
    best_score = float("-inf")
    stale_epochs = 0
    rows = []
    for epoch in range(1, epochs + 1):
        model.train()
        losses = []
        for features, cls_labels, reg_labels, cls_mask, reg_mask in train_loader:
            features = features.to(_device())
            cls_labels = cls_labels.to(_device())
            reg_labels = reg_labels.to(_device())
            cls_mask = cls_mask.to(_device())
            reg_mask = reg_mask.to(_device())
            optimizer.zero_grad()
            outputs = model(features)
            loss, cls_loss, reg_loss = masked_multitask_loss(
                outputs,
                cls_labels,
                reg_labels,
                cls_mask.bool(),
                reg_mask.bool(),
            )
            loss.backward()
            optimizer.step()
            losses.append(
                (
                    float(loss.detach().cpu()),
                    float(cls_loss.detach().cpu()),
                    float(reg_loss.detach().cpu()),
                )
            )
        scheduler.step()
        val_metrics, val_predictions = _evaluate_tabular_model(
            model,
            val_loader,
            "val",
            metadata=val_metadata,
        )
        val_auroc = float(val_metrics.get("auroc", float("nan")))
        rows.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean([item[0] for item in losses])),
                "train_classification_loss": float(
                    np.mean([item[1] for item in losses])
                ),
                "train_regression_loss": float(np.mean([item[2] for item in losses])),
                "val_auroc": val_auroc,
                "val_auprc": float(val_metrics.get("auprc", float("nan"))),
            }
        )
        if pd.notna(val_auroc) and val_auroc > best_score:
            best_score = val_auroc
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
            write_tsv(val_predictions, output_dir / "val_predictions.tsv")
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)
    checkpoint_path = ensure_parent(output_dir / "best_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    log_path = write_tsv(pd.DataFrame(rows), output_dir / "training_log.tsv")
    val_metrics, predictions = _evaluate_tabular_model(
        model,
        val_loader,
        "val",
        metadata=val_metadata,
    )
    _, test_predictions = _evaluate_tabular_model(
        model,
        test_loader,
        "test",
        metadata=test_metadata,
    )
    predictions = pd.concat([predictions, test_predictions], ignore_index=True)
    predictions_path = write_tsv(predictions, output_dir / "predictions.tsv")
    metadata_path = write_json(
        {
            "best_val_auroc": best_score,
            "epochs_completed": len(rows),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "val_metrics": val_metrics,
        },
        output_dir / "training_metadata.json",
    )
    return TrainingOutputs(
        checkpoint_path=checkpoint_path,
        log_path=log_path,
        predictions_path=predictions_path,
        metadata_path=metadata_path,
    )
