import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


def run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "complexvar.cli", *args],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )


def test_cli_smoke_workflow(tmp_path: Path):
    toy_dir = tmp_path / "toy"
    results_dir = tmp_path / "results"
    metrics_path = results_dir / "baseline" / "metrics.json"

    run_cli(tmp_path, "make-toy-dataset", "--output-dir", str(toy_dir))
    run_cli(
        tmp_path,
        "train-baseline",
        "--features",
        str(toy_dir / "classification_features.tsv"),
        "--target-column",
        "label",
        "--output-dir",
        str(results_dir / "baseline"),
    )
    run_cli(
        tmp_path,
        "evaluate-classification",
        "--predictions",
        str(results_dir / "baseline" / "predictions.tsv"),
        "--output",
        str(metrics_path),
    )

    assert metrics_path.exists()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert "auroc" in payload
    predictions = pd.read_csv(results_dir / "baseline" / "predictions.tsv", sep="\t")
    assert {"sample_id", "score", "split"}.issubset(predictions.columns)
