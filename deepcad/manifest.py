"""Utilities for loading one or more split manifests consistently."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


ManifestInput = str | Path | Sequence[str | Path]


def read_manifest(paths: ManifestInput) -> pd.DataFrame:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = []
    for value in paths:
        path = Path(value).expanduser().resolve()
        frame = pd.read_csv(path)
        if "fundus_path" not in frame and "fundus_image_path" in frame:
            frame["fundus_path"] = frame["fundus_image_path"]
        for column in ("cmr_path", "fundus_path", "fundus_image_path"):
            if column in frame:
                frame[column] = frame[column].map(
                    lambda item: str(
                        Path(str(item)).expanduser()
                        if Path(str(item)).expanduser().is_absolute()
                        else (path.parent / str(item)).resolve()))
        frames.append(frame)
    if not frames:
        raise ValueError("At least one manifest is required.")
    return pd.concat(frames, ignore_index=True)
