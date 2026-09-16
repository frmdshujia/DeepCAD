#!/usr/bin/env python3
"""Verify every file listed in a sha256sum-compatible manifest."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checksum-file", required=True)
    args = parser.parse_args()
    checksum_file = Path(args.checksum_file).expanduser().resolve()
    failures = []
    checked = 0
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        relative = relative.lstrip("* ")
        path = checksum_file.parent / relative
        observed = sha256(path)
        checked += 1
        if observed != expected:
            failures.append(str(path))
    if failures:
        raise RuntimeError(
            "SHA256 mismatch for: " + ", ".join(failures))
    print(f"manifest_integrity=PASS files={checked}")


if __name__ == "__main__":
    main()
