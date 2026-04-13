#!/usr/bin/env python3
"""Normalize IntAct mutation rows."""

from __future__ import annotations

import argparse

from complexvar.data.intact import write_normalized_intact


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_normalized_intact(args.input, args.output)


if __name__ == "__main__":
    main()
