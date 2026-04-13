"""Repository text policy checks."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ALLOWED_SUFFIXES = {
    ".cff",
    ".json",
    ".md",
    ".rst",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def _word(parts: list[int]) -> str:
    return "".join(chr(value) for value in parts)


FORBIDDEN_PATTERNS = [
    re.compile(rf"\b{re.escape(_word([97, 105]))}\b", re.IGNORECASE),
    re.compile(
        re.escape(
            _word(
                [
                    97,
                    114,
                    116,
                    105,
                    102,
                    105,
                    99,
                    105,
                    97,
                    108,
                    32,
                    105,
                    110,
                    116,
                    101,
                    108,
                    108,
                    105,
                    103,
                    101,
                    110,
                    99,
                    101,
                ]
            )
        ),
        re.IGNORECASE,
    ),
    re.compile(re.escape(_word([67, 104, 97, 116, 71, 80, 84])), re.IGNORECASE),
    re.compile(re.escape(_word([79, 112, 101, 110, 65, 73])), re.IGNORECASE),
    re.compile(rf"\b{re.escape(_word([76, 76, 77]))}\b", re.IGNORECASE),
    re.compile(
        re.escape(
            _word([108, 97, 110, 103, 117, 97, 103, 101, 32, 109, 111, 100, 101, 108])
        ),
        re.IGNORECASE,
    ),
    re.compile(re.escape(_word([97, 103, 101, 110, 116, 105, 99])), re.IGNORECASE),
    re.compile(
        re.escape(
            _word(
                [
                    109,
                    97,
                    99,
                    104,
                    105,
                    110,
                    101,
                    45,
                    103,
                    101,
                    110,
                    101,
                    114,
                    97,
                    116,
                    101,
                    100,
                ]
            )
        ),
        re.IGNORECASE,
    ),
]

FORBIDDEN_DASHES = {"\u2013", "\u2014"}


@dataclass(slots=True)
class PolicyViolation:
    path: str
    line_number: int
    line: str
    reason: str


def _tracked_files(root: Path) -> list[Path]:
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        paths = [root / line for line in result.stdout.splitlines() if line.strip()]
        return [path for path in paths if path.suffix in ALLOWED_SUFFIXES]
    except Exception:  # noqa: BLE001
        return sorted(
            path
            for path in root.rglob("*")
            if path.is_file() and path.suffix in ALLOWED_SUFFIXES
        )


def scan_text_policy(root: Path) -> list[PolicyViolation]:
    violations: list[PolicyViolation] = []
    for path in _tracked_files(root):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(text.splitlines(), start=1):
            if any(bad_dash in line for bad_dash in FORBIDDEN_DASHES):
                violations.append(
                    PolicyViolation(
                        path=str(path.relative_to(root)),
                        line_number=line_number,
                        line=line,
                        reason="forbidden_dash_character",
                    )
                )
            for pattern in FORBIDDEN_PATTERNS:
                if pattern.search(line):
                    violations.append(
                        PolicyViolation(
                            path=str(path.relative_to(root)),
                            line_number=line_number,
                            line=line,
                            reason="forbidden_term",
                        )
                    )
                    break
    return violations


def assert_text_policy(root: Path) -> None:
    violations = scan_text_policy(root)
    if violations:
        preview = "\n".join(
            f"{item.path}:{item.line_number}:{item.reason}:{item.line}"
            for item in violations[:20]
        )
        raise RuntimeError(preview)
