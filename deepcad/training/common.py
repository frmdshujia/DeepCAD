from __future__ import annotations

import json
import math
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


def assert_manifest_schema(
    manifest: ManifestInput,
    required_columns: Iterable[str],
    *,
    allow_repeated_within_split: bool,
) -> None:
    """Fail before training when columns or participant splits are invalid."""
    frame = read_manifest(manifest)
    required = {"eid", "split", *required_columns}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Manifest is missing required columns: " + ", ".join(missing))
    invalid_splits = sorted(set(frame["split"].dropna().astype(str))
                            - {"train", "val", "test"})
    if invalid_splits:
        raise ValueError(
            "Unknown manifest split values: " + ", ".join(invalid_splits))
    participant_splits = frame.groupby("eid")["split"].nunique(dropna=False)
    if (participant_splits > 1).any():
        example = participant_splits[participant_splits > 1].index[0]
        raise ValueError(
            f"Participant {example} occurs in more than one data split.")
    if not allow_repeated_within_split:
        duplicated = frame.duplicated(["eid", "split"], keep=False)
        if duplicated.any():
            example = frame.loc[duplicated, "eid"].iloc[0]
            raise ValueError(
                f"Participant {example} has duplicate rows within one split.")


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


def warmup_cosine_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_epochs: int,
    total_epochs: int,
    min_lr_ratio: float = 0.01,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warm-up followed by cosine decay to a fixed LR ratio."""
    if warmup_epochs < 0 or warmup_epochs >= total_epochs:
        raise ValueError("warmup_epochs must be in [0, total_epochs).")
    if not 0 < min_lr_ratio <= 1:
        raise ValueError("min_lr_ratio must be in (0, 1].")

    def multiplier(epoch: int) -> float:
        if warmup_epochs and epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(
            total_epochs - warmup_epochs - 1, 1)
        progress = min(max(progress, 0.0), 1.0)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)
