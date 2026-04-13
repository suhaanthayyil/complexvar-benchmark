#!/usr/bin/env python3
"""Write the Burke release download manifest."""

from __future__ import annotations

import argparse

from complexvar.data.burke import write_download_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    write_download_manifest(args.output)


if __name__ == "__main__":
    main()
