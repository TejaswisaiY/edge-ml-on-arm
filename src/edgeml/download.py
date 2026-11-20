"""Small, dependency-free downloader with SHA-256 pinning.

Every external file the pipeline uses is pinned to an exact version and checksum,
so a run today and a run in six months train on the same bytes.
"""

from __future__ import annotations

import hashlib
import shutil
import tempfile
import urllib.request
from pathlib import Path


class ChecksumError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url: str, dest: Path, sha256: str | None) -> Path:
    """Download ``url`` to ``dest`` unless a file with the right checksum is already there."""
    dest = Path(dest)
    if dest.exists() and (sha256 is None or sha256_of(dest) == sha256):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=dest.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)
        with urllib.request.urlopen(url, timeout=120) as resp:  # noqa: S310 (pinned https URLs only)
            shutil.copyfileobj(resp, tmp)
    if sha256 is not None:
        got = sha256_of(tmp_path)
        if got != sha256:
            tmp_path.unlink(missing_ok=True)
            raise ChecksumError(f"{url}: expected sha256 {sha256}, got {got}")
    tmp_path.replace(dest)
    return dest
