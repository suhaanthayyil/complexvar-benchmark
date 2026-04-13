#!/usr/bin/env python3
"""Train a regression baseline."""

from __future__ import annotations

import sys

from complexvar.cli import build_parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(["train-regression-baseline", *sys.argv[1:]])
    args.func(args)
