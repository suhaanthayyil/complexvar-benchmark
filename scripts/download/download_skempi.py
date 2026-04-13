#!/usr/bin/env python3
"""Download SKEMPI data and matching RCSB structures."""

from __future__ import annotations

import argparse

from complexvar.downloads import Step1Config, run_step1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".")
    parser.add_argument("--pdb-limit", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    run_step1(
        Step1Config(
            datasets=["skempi", "rcsb"],
            root=args.root,
            skip_existing=args.skip_existing,
            force=args.force,
            pdb_limit=args.pdb_limit,
            workers=args.workers,
        )
    )


if __name__ == "__main__":
    main()
