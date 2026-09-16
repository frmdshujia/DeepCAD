from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from deepcad.manifest import ManifestInput, read_manifest


def autocast_context(device: torch.device, amp_dtype: str):
    """Use CUDA autocast when requested and a no-op context elsewhere."""
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def parse_columns(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def make_output_dir(path: str, allow_existing: bool = False) -> Path:
    output = Path(path).expanduser().resolve()
    if output.exists() and any(output.iterdir()) and not allow_existing:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use a new versioned "
            "directory or pass --allow-existing explicitly.")
    output.mkdir(parents=True, exist_ok=True)
    return output


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def save_checkpoint(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def regression_statistics(
    manifest: ManifestInput, columns: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    if not columns:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    frame = read_manifest(manifest)
    train = frame.loc[frame["split"] == "train", columns].apply(
        pd.to_numeric, errors="coerce")
    mean = train.mean(axis=0).to_numpy(np.float32)
    std = train.std(axis=0).replace(0, 1).fillna(1).to_numpy(np.float32)
    return mean, std


def classification_pos_weights(
    manifest: ManifestInput, columns: list[str]
) -> np.ndarray:
    """Compute per-task negative/positive ratios from the train split only."""
    if not columns:
        return np.empty(0, np.float32)
    frame = read_manifest(manifest)
    targets = frame.loc[frame["split"] == "train", columns].apply(
        pd.to_numeric, errors="coerce")
    weights = []
    for column in columns:
        values = targets[column].dropna().to_numpy()
        positive = int(np.sum(values == 1))
        negative = int(np.sum(values == 0))
        if positive == 0:
            raise ValueError(
                f"Classification target {column!r} has no positive training cases.")
        weights.append(negative / positive)
    return np.asarray(weights, dtype=np.float32)


def assert_disjoint_participants(
    training_manifest: ManifestInput, external_manifest: ManifestInput | None
) -> None:
    if not external_manifest:
        return
    train_ids = set(read_manifest(training_manifest)["eid"].astype(str))
    external_ids = set(read_manifest(external_manifest)["eid"].astype(str))
    overlap = train_ids & external_ids
    if overlap:
        raise ValueError(
            f"Participant leakage detected: {len(overlap)} identifiers occur in "
            "both the development and external-validation manifests.")


def dump_run_config(output: Path, args, extra: dict | None = None) -> None:
    payload = dict(vars(args))
    if extra:
        payload.update(extra)
    (output / "run_config.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str))


def trainable_parameters(modules: Iterable[torch.nn.Module]):
    for module in modules:
        yield from (parameter for parameter in module.parameters()
                    if parameter.requires_grad)
