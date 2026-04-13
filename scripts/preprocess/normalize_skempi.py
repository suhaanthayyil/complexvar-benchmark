#!/usr/bin/env python3
"""Normalize SKEMPI rows."""

from __future__ import annotations

import argparse

from complexvar.data.skempi import write_normalized_skempi


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_normalized_skempi(args.input, args.output)


if __name__ == "__main__":
    main()
