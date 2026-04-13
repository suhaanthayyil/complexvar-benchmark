#!/usr/bin/env python3
"""Filter ClinVar to benchmark-ready labels."""

from __future__ import annotations

import argparse

from complexvar.data.clinvar import write_filtered_clinvar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_filtered_clinvar(args.input, args.output)


if __name__ == "__main__":
    main()
