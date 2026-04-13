#!/usr/bin/env python3
"""Make the toy summary figure."""

from __future__ import annotations

import sys

from complexvar.cli import build_parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(["make-summary-figure", *sys.argv[1:]])
    args.func(args)
