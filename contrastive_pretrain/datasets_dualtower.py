"""
datasets_dualtower.py

DataLoaders for end-to-end dual-tower training.
  - CMRDataset     : reads {eid}.npy (4,224,224) + labels from CMR sampled CSV
  - FundusDataset  : finds {eid}_2101[56]_{instance}_*.png + labels from fundus sampled CSV
  - PairedDataset  : EIDs with both modalities available (cmr npy + fundus png)
"""
from __future__ import annotations

import pathlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]
REG_COLS = [
    'LV ejection fraction',
    'LV end diastolic volume',
    'LV end systolic volume',
]

FUNDUS_IMG_DIR = pathlib.Path('/data/home/home6/fundus_data/UKB/fundus_images')
CMR_NPY_DIR    = pathlib.Path('/data/home/shujia/UKB/CMRI/preprocessed_lax_sax')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _fundus_transform(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 3.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


def _build_fundus_index() -> dict:
    """Build (eid, instance) -> [path, ...] index once at import time.
    Each key maps to a list of all available eye paths (left 21015 and/or right 21016),
    sorted so right eye (21016) comes first."""
    index: dict = {}
    try:
        for fname in FUNDUS_IMG_DIR.iterdir():
            if not fname.suffix == '.png':
                continue
            parts = fname.stem.split('_')
            if len(parts) < 3:
                continue
            eid, field, inst = int(parts[0]), int(parts[1]), int(parts[2])
            key = (eid, inst)
            index.setdefault(key, []).append((field, fname))
    except Exception as e:
        print(f'[datasets_dualtower] warning: could not build fundus index: {e}')
    # sort each list: right eye (21016) first, then left (21015)
    for key in index:
        index[key] = [p for _, p in sorted(index[key], key=lambda x: x[0], reverse=True)]
    return index


_FUNDUS_INDEX: dict | None = None


def _get_fundus_index() -> dict:
    global _FUNDUS_INDEX
    if _FUNDUS_INDEX is None:
        print('[datasets_dualtower] building fundus image index ...')
        _FUNDUS_INDEX = _build_fundus_index()
        n_keys = len(_FUNDUS_INDEX)
        n_imgs = sum(len(v) for v in _FUNDUS_INDEX.values())
        print(f'[datasets_dualtower] fundus index: {n_keys:,} (eid,instance) keys, {n_imgs:,} images')
    return _FUNDUS_INDEX


def _find_fundus_path(eid: int, instance: int) -> pathlib.Path | None:
    """Return single best path (right eye preferred). Used by PairedDataset."""
    idx = _get_fundus_index()
    paths = idx.get((eid, instance))
    return paths[0] if paths else None


def _find_fundus_paths(eid: int, instance: int) -> list:
    """Return all available eye paths for this (eid, instance). Used by FundusDataset."""
    idx = _get_fundus_index()
    return idx.get((eid, instance), [])


def _load_labels(row: pd.Series):
    """Returns (cls_labels float32 (4,), reg_labels float32 (3,), reg_mask bool (3,))."""
    cls = np.array([
        float(row[c]) if pd.notna(row[c]) else -1.0
        for c in CLS_COLS
    ], dtype=np.float32)
    reg = np.array([
        float(row[c]) if pd.notna(row.get(c, np.nan)) else np.nan
        for c in REG_COLS
    ], dtype=np.float32)
    mask = ~np.isnan(reg)
    reg = np.where(mask, reg, 0.0)
    return cls, reg, mask


class CMRDataset(Dataset):
    """Reads preprocessed CMR npy files. Skips EIDs without a .npy file.
    split=None means use all rows (for pre-split CSV files)."""

    def __init__(self, csv_path: str, split: str | None = None, max_samples: int = 0):
        df = pd.read_csv(csv_path, low_memory=False)
        if split is not None and 'split' in df.columns:
            df = df[df['split'] == split].reset_index(drop=True)

        rows = []
        for _, row in df.iterrows():
            eid = int(row['eid'])
            npy = CMR_NPY_DIR / f'{eid}.npy'
            if npy.exists():
                rows.append({'eid': eid, 'npy': npy, 'row': row})

        if max_samples and max_samples < len(rows):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(rows), size=max_samples, replace=False)
            rows = [rows[i] for i in idx]

        self.rows = rows
        print(f'[CMRDataset] split={split}: {len(self.rows):,} samples with npy')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        npy = np.load(item['npy']).astype(np.float32)  # (4, 224, 224)
        cmr = torch.from_numpy(npy).unsqueeze(1)        # (4, 1, 224, 224)
        cls, reg, mask = _load_labels(item['row'])
        return {
            'cmr':        cmr,
            'eid':        item['eid'],
            'cls_labels': torch.from_numpy(cls),
            'reg_labels': torch.from_numpy(reg),
            'reg_mask':   torch.from_numpy(mask),
        }


class FundusDataset(Dataset):
    """Reads fundus PNG images. Each person contributes up to 2 entries (both eyes).
    split=None means use all rows (for pre-split CSV files)."""

    def __init__(self, csv_path: str, split: str | None = None,
                 max_samples: int = 0, is_train: bool = True):
        df = pd.read_csv(csv_path, low_memory=False)
        if split is not None and 'split' in df.columns:
            df = df[df['split'] == split].reset_index(drop=True)

        rows = []
        for _, row in df.iterrows():
            eid = int(row['eid'])
            inst = int(row['instance']) if 'instance' in row else 0
            paths = _find_fundus_paths(eid, inst)
            for p in paths:
                rows.append({'eid': eid, 'path': str(p), 'row': row})

        if max_samples and max_samples < len(rows):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(rows), size=max_samples, replace=False)
            rows = [rows[i] for i in idx]

        self.rows = rows
        self.transform = _fundus_transform(is_train)
        n_eids = len({r['eid'] for r in self.rows})
        print(f'[FundusDataset] split={split}: {len(self.rows):,} images from {n_eids:,} people')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]
        img = Image.open(item['path']).convert('RGB')
        fundus = self.transform(img)
        cls, reg, mask = _load_labels(item['row'])
        return {
            'fundus':     fundus,
            'eid':        item['eid'],
            'cls_labels': torch.from_numpy(cls),
            'reg_labels': torch.from_numpy(reg),
            'reg_mask':   torch.from_numpy(mask),
        }


class PairedDataset(Dataset):
    """
    EIDs that have BOTH a CMR npy AND a fundus PNG.
    Merges labels from CMR CSV (authoritative for LV) with fundus CSV (for visit/instance).
    """

    def __init__(self, cmr_csv: str, fundus_csv: str, split: str | None = None,
                 max_samples: int = 0, is_train: bool = True):
        df_c = pd.read_csv(cmr_csv,    low_memory=False)
        df_f = pd.read_csv(fundus_csv, low_memory=False)
        if split is not None and 'split' in df_c.columns:
            df_c = df_c[df_c['split'] == split].reset_index(drop=True)
        if split is not None and 'split' in df_f.columns:
            df_f = df_f[df_f['split'] == split].reset_index(drop=True)

        cmr_eids = {int(r['eid']): r for _, r in df_c.iterrows()
                    if (CMR_NPY_DIR / f'{int(r["eid"])}.npy').exists()}

        rows = []
        for _, row_f in df_f.iterrows():
            eid = int(row_f['eid'])
            if eid not in cmr_eids:
                continue
            inst = int(row_f['instance']) if 'instance' in row_f else 0
            paths = _find_fundus_paths(eid, inst)
            for p in paths:
                rows.append({
                    'eid':   eid,
                    'path':  str(p),
                    'npy':   CMR_NPY_DIR / f'{eid}.npy',
                    'row_c': cmr_eids[eid],
                    'row_f': row_f,
                })

        if max_samples and max_samples < len(rows):
            rng = np.random.default_rng(42)
            idx = rng.choice(len(rows), size=max_samples, replace=False)
            rows = [rows[i] for i in idx]

        self.rows = rows
        self.transform = _fundus_transform(is_train)
        n_eids = len({r['eid'] for r in self.rows})
        print(f'[PairedDataset] split={split}: {len(self.rows):,} paired images from {n_eids:,} people')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        item = self.rows[idx]

        npy = np.load(item['npy']).astype(np.float32)
        cmr = torch.from_numpy(npy).unsqueeze(1)  # (4, 1, 224, 224)

        img = Image.open(item['path']).convert('RGB')
        fundus = self.transform(img)

        # Use CMR row for labels (has LV regression; CLS from CMR row)
        cls, reg, mask = _load_labels(item['row_c'])
        return {
            'fundus':     fundus,
            'cmr':        cmr,
            'eid':        item['eid'],
            'cls_labels': torch.from_numpy(cls),
            'reg_labels': torch.from_numpy(reg),
            'reg_mask':   torch.from_numpy(mask),
        }
