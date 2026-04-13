"""Filesystem and table helpers."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def ensure_parent(path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    return target


def write_json(data: Any, path: str | Path) -> Path:
    output = ensure_parent(path)
    output.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    return output


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_tsv(df: pd.DataFrame, path: str | Path) -> Path:
    output = ensure_parent(path)
    df.sort_index(axis=1).to_csv(output, sep="\t", index=False)
    return output


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix == ".gz":
        return pd.read_csv(path, sep="\t", compression="gzip")
    return pd.read_csv(path, sep="\t")


def open_maybe_gzip(path: str | Path, mode: str = "rt"):
    path = Path(path)
    if path.suffix == ".gz":
        return gzip.open(path, mode, encoding="utf-8")
    return path.open(mode, encoding="utf-8")
