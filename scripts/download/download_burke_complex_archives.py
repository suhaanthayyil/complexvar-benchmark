#!/usr/bin/env python3
"""Download Burke high-level complex archives."""

from __future__ import annotations

import argparse
from pathlib import Path

from complexvar.data.burke import burke_file_urls
from complexvar.utils.download import stream_download


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, url in burke_file_urls("complex").items():
        stream_download(url, output_dir / name)


if __name__ == "__main__":
    main()
