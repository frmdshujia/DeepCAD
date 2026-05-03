"""
datasets_paired_max.py

Dataset for Mutual Contrastive + Downstream Task joint training.
Source: task_reports/paired_max_{train,val,test}.csv
  - Both fundus PNG and CMR npy are verified to exist.
  - Both-eye expansion: each EID contributes up to 2 rows (left + right eye),
    each paired with the same CMR npy.

UniqueEIDSampler:
  - Each epoch, randomly picks ONE eye per EID → guaranteed no duplicate EID
    in any batch window of any size.
  - Over multiple epochs, both eyes get used (different random picks per epoch).
"""
from __future__ import annotations

import fcntl
import json
import pathlib
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image

FUNDUS_IMG_DIR = pathlib.Path('/data/home/home6/fundus_data/UKB/fundus_images')
CMR_NPY_DIR    = pathlib.Path('/data/home/shujia/UKB/CMRI/preprocessed_lax_sax')

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


_EYE_INDEX: dict | None = None


_INDEX_CACHE = pathlib.Path('/tmp/fundus_eye_index.json')

def _get_eye_index() -> dict:
    """(eid, instance) -> [path, ...] sorted right eye (21016) first.
    Cached to /tmp so 4 DDP processes only scan the filesystem once.
    """
    global _EYE_INDEX
    if _EYE_INDEX is not None:
        return _EYE_INDEX

    # File-lock so only rank-0 builds the cache; others wait and read.
    lock_path = pathlib.Path('/tmp/fundus_eye_index.lock')
    with open(lock_path, 'w') as lf:
        fcntl.flock(lf, fcntl.LOCK_EX)
        try:
            if _INDEX_CACHE.exists():
                print('[datasets_paired_max] loading fundus eye index from cache ...')
                raw = json.loads(_INDEX_CACHE.read_text())
                index = {(int(k.split(',')[0]), int(k.split(',')[1])): [pathlib.Path(p) for p in v]
                         for k, v in raw.items()}
            else:
                print('[datasets_paired_max] building fundus eye index ...')
                index: dict = {}
                try:
                    for fname in FUNDUS_IMG_DIR.iterdir():
                        if fname.suffix != '.png':
                            continue
                        parts = fname.stem.split('_')
                        if len(parts) < 3:
                            continue
                        eid, field, inst = int(parts[0]), int(parts[1]), int(parts[2])
                        index.setdefault((eid, inst), []).append((field, fname))
                except Exception as e:
                    print(f'[datasets_paired_max] warning: could not build index: {e}')
                for key in index:
                    index[key] = [p for _, p in sorted(index[key], key=lambda x: x[0], reverse=True)]
                # Save cache
                serial = {f'{k[0]},{k[1]}': [str(p) for p in v] for k, v in index.items()}
                _INDEX_CACHE.write_text(json.dumps(serial))
        finally:
            fcntl.flock(lf, fcntl.LOCK_UN)

    n_imgs = sum(len(v) for v in index.values())
    print(f'[datasets_paired_max] {len(index):,} (eid,inst) keys, {n_imgs:,} images')
    _EYE_INDEX = index
    return _EYE_INDEX


class PairedMaxDataset(Dataset):
    """
    Each person in paired_max CSV contributes up to 2 rows (both eyes),
    each paired with the same CMR npy. Labels are shared across eyes.

    Use UniqueEIDSampler with this dataset to prevent same-EID duplicates
    within a batch during InfoNCE training.
    """

    def __init__(self, csv_path: str, is_train: bool = True):
        df = pd.read_csv(csv_path, low_memory=False)
        eye_idx = _get_eye_index()

        rows = []
        for _, row in df.iterrows():
            eid = int(row['eid'])
            npy_path = pathlib.Path(str(row['cmr_npy_path']))
            if not npy_path.exists():
                continue
            inst = int(row.get('fundus_instance', 0))
            paths = eye_idx.get((eid, inst), [])
            if not paths:
                # fallback to the path stored in CSV
                p = pathlib.Path(str(row['fundus_image_path']))
                if p.exists():
                    paths = [p]
            for p in paths:
                rows.append({
                    'eid':       eid,
                    'fundus_path': str(p),
                    'npy_path':  str(npy_path),
                    'row':       row,
                })

        self.rows = rows
        self.transform = _fundus_transform(is_train)
        n_eids = len({r['eid'] for r in rows})
        split = 'train' if is_train else 'val/test'
        print(f'[PairedMaxDataset] {split}: {len(rows):,} images from {n_eids:,} people')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        item = self.rows[idx]
        row  = item['row']

        img    = Image.open(item['fundus_path']).convert('RGB')
        fundus = self.transform(img)

        npy = np.load(item['npy_path']).astype(np.float32)
        cmr = torch.from_numpy(npy).unsqueeze(1)  # (4, 1, 224, 224)

        cls = np.array([
            float(row[c]) if pd.notna(row.get(c, np.nan)) else -1.0
            for c in CLS_COLS
        ], dtype=np.float32)
        reg = np.array([
            float(row[c]) if pd.notna(row.get(c, np.nan)) else np.nan
            for c in REG_COLS
        ], dtype=np.float32)
        mask = ~np.isnan(reg)
        reg  = np.where(mask, reg, 0.0)

        return {
            'eid':        item['eid'],
            'fundus':     fundus,
            'cmr':        cmr,
            'cls_labels': torch.from_numpy(cls),
            'reg_labels': torch.from_numpy(reg),
            'reg_mask':   torch.from_numpy(mask),
        }


class UniqueEIDSampler(Sampler):
    """
    Guarantees no duplicate EIDs within any batch, DDP-compatible.

    Each epoch: for every EID, randomly select ONE eye index (left or right).
    The resulting one-per-EID list is shuffled, then divided across DDP ranks.
    This ensures:
      - No duplicate EIDs globally (UniqueEID selection)
      - Each rank gets a disjoint subset (DDP shard)
      - No padding with duplicate indices (drop_last handled by DataLoader)
    """

    def __init__(self, dataset: PairedMaxDataset, seed: int = 0,
                 num_replicas: int = 1, rank: int = 0):
        eid_to_idx: dict = defaultdict(list)
        for i, r in enumerate(dataset.rows):
            eid_to_idx[r['eid']].append(i)
        self.eid_groups = list(eid_to_idx.values())
        self.seed = seed
        self._epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __len__(self):
        # each rank gets floor(total / num_replicas) samples
        return len(self.eid_groups) // self.num_replicas

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        # 1. pick one eye per EID (same choice on all ranks — same seed)
        indices = [int(rng.choice(g)) for g in self.eid_groups]
        rng.shuffle(indices)
        # 2. trim to multiple of num_replicas, then take this rank's shard
        total = (len(indices) // self.num_replicas) * self.num_replicas
        indices = indices[:total]
        indices = indices[self.rank::self.num_replicas]
        return iter(indices)
