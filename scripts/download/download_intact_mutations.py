#!/usr/bin/env python3
"""Download IntAct / IMEx mutation exports."""

from __future__ import annotations

import argparse

from complexvar.utils.download import stream_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    stream_download(args.url, args.output)


if __name__ == "__main__":
    main()
