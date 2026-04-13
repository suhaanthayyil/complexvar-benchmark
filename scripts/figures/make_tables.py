#!/usr/bin/env python3
"""Convert a metrics JSON file into a flat TSV table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from complexvar.utils.io import write_tsv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    flat = {}
    for key, value in payload.items():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, dict):
                    for sub_key, sub_value in inner_value.items():
                        flat[f"{key}.{inner_key}.{sub_key}"] = sub_value
                else:
                    flat[f"{key}.{inner_key}"] = inner_value
        else:
            flat[key] = value
    write_tsv(pd.DataFrame([flat]), args.output)


if __name__ == "__main__":
    main()
