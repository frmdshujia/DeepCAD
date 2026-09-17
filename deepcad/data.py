"""Manifest-driven datasets. No participant identifiers or paths are bundled."""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler
from torchvision import transforms

from .manifest import ManifestInput, read_manifest


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def select_one_image_per_participant(frame: pd.DataFrame) -> pd.DataFrame:
    """Select one retinal photograph per participant deterministically.

    Selection is label-independent. When available, the photograph closest in
    time to CMR is preferred, followed by the lowest fundus acquisition
    instance. The resolved image path is the final stable tie-breaker.
    """
    if "eid" not in frame or "fundus_path" not in frame:
        raise ValueError("Retinal manifests require eid and fundus_path columns.")
    ranked = frame.copy()
    if "visit_interval_years" in ranked:
        ranked["_visit_gap"] = pd.to_numeric(
            ranked["visit_interval_years"], errors="coerce").abs().fillna(np.inf)
    else:
        ranked["_visit_gap"] = np.inf
    if "fundus_instance" in ranked:
        ranked["_fundus_instance"] = pd.to_numeric(
            ranked["fundus_instance"], errors="coerce").fillna(np.inf)
    else:
        ranked["_fundus_instance"] = np.inf
    ranked["_stable_path"] = ranked["fundus_path"].astype(str)
    ranked = ranked.sort_values(
        ["eid", "_visit_gap", "_fundus_instance", "_stable_path"],
        kind="stable")
    return ranked.drop_duplicates("eid", keep="first").drop(
        columns=["_visit_gap", "_fundus_instance", "_stable_path"]
    ).reset_index(drop=True)


def _as_bool(value: object) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n", ""}:
        return False
    raise ValueError(f"Cannot parse Boolean value: {value!r}")


def fundus_transform(training: bool, image_size: int = 224,
                     protocol: str = "supervised"):
    if training and protocol in {"contrastive", "supervised"}:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(180),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0))
            ], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    if protocol not in {"contrastive", "supervised"}:
        raise ValueError(f"Unknown fundus transform protocol: {protocol}")
    return transforms.Compose([
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CMRDataset(Dataset):
    """CMR teacher dataset driven by a CSV manifest.

    Required columns: ``eid``, ``split``, ``cmr_path`` and ``t1_available``.
    Classification and regression columns are supplied explicitly by the caller.
    """

    def __init__(self, manifest: ManifestInput, split: str,
                 classification_columns: list[str],
                 regression_columns: list[str], training: bool = False):
        frame = read_manifest(manifest)
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        self.classification_columns = classification_columns
        self.regression_columns = regression_columns
        self.training = training

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        cmr_path = row["cmr_path"]
        cmr = np.load(cmr_path).astype(np.float32)
        if cmr.shape != (16, 224, 224):
            raise ValueError(f"Unexpected CMR shape {cmr.shape}: {cmr_path}")
        tensor = torch.from_numpy(cmr).unsqueeze(1)

        cls = pd.to_numeric(row[self.classification_columns], errors="coerce").to_numpy(
            dtype=np.float32) if self.classification_columns else np.empty(0, np.float32)
        reg = pd.to_numeric(row[self.regression_columns], errors="coerce").to_numpy(
            dtype=np.float32) if self.regression_columns else np.empty(0, np.float32)
        cls_mask = np.isfinite(cls)
        reg_mask = np.isfinite(reg)
        return {
            "eid": int(row["eid"]),
            "cmr": tensor,
            "t1_available": torch.tensor(_as_bool(row["t1_available"])),
            "classification_targets": torch.from_numpy(np.nan_to_num(cls)),
            "regression_targets": torch.from_numpy(np.nan_to_num(reg)),
            "classification_mask": torch.from_numpy(cls_mask),
            "regression_mask": torch.from_numpy(reg_mask),
        }


class Stage1ContrastiveDataset(Dataset):
    """Fundus images paired with precomputed frozen CMR teacher embeddings."""

    def __init__(self, manifest: ManifestInput, split: str, embedding_file: str,
                 embedding_eids_file: str, training: bool,
                 require_t1: bool = False):
        frame = read_manifest(manifest)
        frame = frame.loc[frame["split"] == split].copy()
        if require_t1:
            frame = frame.loc[frame["t1_available"].map(_as_bool)]
        embeddings = np.load(embedding_file, mmap_mode="r")
        eids = np.load(embedding_eids_file).astype(np.int64)
        self.eid_to_index = {int(eid): i for i, eid in enumerate(eids)}
        frame = frame[frame["eid"].astype(int).isin(self.eid_to_index)]
        if not training:
            frame = select_one_image_per_participant(frame)
        self.frame = frame.reset_index(drop=True)
        self.embeddings = embeddings
        self.transform = fundus_transform(training, protocol="contrastive")
        self.eids = self.frame["eid"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        eid = int(row["eid"])
        image_path = row["fundus_path"]
        image = self.transform(Image.open(image_path).convert("RGB"))
        embedding = np.asarray(
            self.embeddings[self.eid_to_index[eid]], dtype=np.float32).copy()
        return {
            "eid": eid,
            "image": image,
            "cmr_embedding": torch.from_numpy(embedding),
            "t1_available": torch.tensor(_as_bool(row["t1_available"])),
        }


class UniqueParticipantSampler(Sampler[int]):
    """Select one image per participant per epoch."""

    def __init__(self, dataset: Dataset, seed: int = 0):
        if not hasattr(dataset, "eids"):
            raise TypeError("UniqueParticipantSampler requires dataset.eids.")
        groups: dict[int, list[int]] = defaultdict(list)
        for index, eid in enumerate(dataset.eids):
            groups[eid].append(index)
        self.groups = list(groups.values())
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.groups)

    def __iter__(self):
        generator = np.random.default_rng(self.seed + self.epoch)
        indices = [int(generator.choice(group)) for group in self.groups]
        generator.shuffle(indices)
        return iter(indices)


class FundusBinaryDataset(Dataset):
    """Stage II binary CAD dataset.

    Required columns: ``eid``, ``split``, ``fundus_path`` and ``label``.
    """

    def __init__(self, manifest: ManifestInput, split: str, training: bool):
        frame = read_manifest(manifest)
        frame = frame.loc[frame["split"] == split].copy()
        conflicting = frame.groupby("eid")["label"].nunique(dropna=False)
        if (conflicting > 1).any():
            eid = conflicting[conflicting > 1].index[0]
            raise ValueError(f"Conflicting labels for EID {eid}")
        if not training:
            frame = select_one_image_per_participant(frame)
        self.frame = frame.reset_index(drop=True)
        self.transform = fundus_transform(training, protocol="supervised")
        self.eids = self.frame["eid"].astype(int).tolist()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        image_path = row["fundus_path"]
        image = self.transform(Image.open(image_path).convert("RGB"))
        return {"eid": int(row["eid"]), "image": image,
                "label": torch.tensor(float(row["label"]), dtype=torch.float32)}


class ClinicalFusionDataset(Dataset):
    """Stage III retinal-score and clinical-risk-factor dataset."""

    def __init__(self, manifest: ManifestInput, split: str,
                 feature_columns: list[str]):
        frame = read_manifest(manifest)
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        self.feature_columns = feature_columns

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        features = row[self.feature_columns].to_numpy(dtype=np.float32)
        return {
            "eid": int(row["eid"]),
            "features": torch.from_numpy(features),
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
        }
