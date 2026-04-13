#!/usr/bin/env python3
"""Build the Burke structure manifest."""

from __future__ import annotations

import argparse

from complexvar.data.burke import build_structure_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-csv", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--pdockq-threshold", type=float, default=0.5)
    parser.add_argument("--structure-root")
    args = parser.parse_args()
    build_structure_manifest(
        args.summary_csv,
        args.output,
        args.pdockq_threshold,
        structure_root=args.structure_root,
    )


if __name__ == "__main__":
    main()
