"""Download helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path

import requests

from complexvar.utils.io import ensure_parent


def stream_download(
    url: str, destination: str | Path, chunk_size: int = 1 << 20
) -> Path:
    destination = ensure_parent(destination)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)
    return destination


def md5sum(path: str | Path) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
