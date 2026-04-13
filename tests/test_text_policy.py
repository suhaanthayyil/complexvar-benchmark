from __future__ import annotations

import subprocess
from pathlib import Path

from complexvar.text_policy import scan_text_policy


def _make_word(values: list[int]) -> str:
    return "".join(chr(value) for value in values)


def test_text_policy_detects_forbidden_dash(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    bad_dash = chr(0x2014)
    path = tmp_path / "README.md"
    path.write_text(f"title {bad_dash} note\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    violations = scan_text_policy(tmp_path)
    assert any(item.reason == "forbidden_dash_character" for item in violations)


def test_text_policy_detects_forbidden_term(tmp_path: Path):
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True)
    path = tmp_path / "README.md"
    bad_text = _make_word([67, 104, 97, 116, 71, 80, 84])
    path.write_text(bad_text + "\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )
    violations = scan_text_policy(tmp_path)
    assert any(item.reason == "forbidden_term" for item in violations)
