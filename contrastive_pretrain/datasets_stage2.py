"""datasets_stage2.py -- fundus + CMR image-level dataset for Stage 2.

Each sample: (fundus_image [3,224,224], cmr_frames [4,1,224,224], eid).
- fundus: PNG loaded from `fundus_image_path` column in fundus_table.csv.
- cmr: preprocessed 4-frame keyframes stored at {cmr_npy_dir}/{eid}.npy with
  shape (4, 224, 224) float16 (LAX_4Ch ED/ES + SAX_mid ED/ES).

EIDs whose CMR npy file is missing are silently skipped (useful for Tier 0/1
subsets that only preprocessed a small cohort).
"""
from __future__ import annotations
import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FundusCMRImageDataset(Dataset):
    """Return (fundus_img_tensor, cmr_tensor, eid) for Stage 2 contrastive
    learning. CMR tensor shape = (num_frames, 1, 224, 224), grayscale.
    """

    def __init__(
        self,
        csv_path: str,
        cmr_npy_dir: str,
        split: str = 'train',
        num_frames: int = 4,
        use_eval_transform: bool = False,
        eid_allowlist: Optional[List[int]] = None,
        train_max_samples: int = 0,
        subset_seed: int = 0,
    ):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)
        if len(df) == 0:
            raise ValueError(f'No samples for split={split} in {csv_path}')

        n_total = len(df)
        exist_mask = df['fundus_image_path'].apply(os.path.exists)
        df = df[exist_mask].reset_index(drop=True)

        # filter by CMR npy availability
        def _cmr_exists(eid):
            return os.path.exists(os.path.join(cmr_npy_dir, f'{int(eid)}.npy'))
        df = df[df['eid'].apply(_cmr_exists)].reset_index(drop=True)

        if eid_allowlist is not None:
            allow = set(int(e) for e in eid_allowlist)
            df = df[df['eid'].isin(allow)].reset_index(drop=True)

        if split == 'train' and train_max_samples and train_max_samples > 0 and len(df) > train_max_samples:
            rng = np.random.default_rng(subset_seed)
            idx = rng.choice(len(df), size=train_max_samples, replace=False)
            df = df.iloc[idx].reset_index(drop=True)

        self.image_paths = df['fundus_image_path'].tolist()
        self.eids = df['eid'].astype(int).tolist()
        self.cmr_npy_dir = cmr_npy_dir
        self.num_frames = num_frames
        self.is_train = (split == 'train') and (not use_eval_transform)
        self.transform = self._build_fundus_transform()

        print(f'[FundusCMRImageDataset] split={split}: '
              f'usable_samples={len(df)} / {n_total}, unique_eids={len(set(self.eids))}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        eid = self.eids[idx]

        img = Image.open(path).convert('RGB')
        img = self.transform(img)  # (3, 224, 224)

        cmr_path = os.path.join(self.cmr_npy_dir, f'{eid}.npy')
        cmr = np.load(cmr_path)               # (4, 224, 224) float16, values in [0,1]
        if cmr.shape[0] != self.num_frames:
            raise RuntimeError(f'CMR npy for eid={eid} has {cmr.shape[0]} frames, expected {self.num_frames}')
        cmr = torch.from_numpy(cmr.astype(np.float32))   # (F, 224, 224)
        cmr = cmr.unsqueeze(1)                            # (F, 1, 224, 224)
        cmr = (cmr - 0.5) / 0.5                           # normalize to [-1,1]

        return img, cmr, eid

    def _build_fundus_transform(self):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if self.is_train:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3/4, 4/3)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str,
                    default='contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table.csv')
    ap.add_argument('--cmr_dir', type=str,
                    default='/data/home/shujia/UKB/CMRI/preprocessed_lax_sax')
    ap.add_argument('--split', type=str, default='train')
    args = ap.parse_args()

    ds = FundusCMRImageDataset(args.csv, args.cmr_dir, split=args.split, num_frames=4)
    print(f'[check] len={len(ds)}')
    if len(ds) > 0:
        img, cmr, eid = ds[0]
        print(f'[check] sample 0: eid={eid}')
        print(f'[check]   fundus img: {tuple(img.shape)} dtype={img.dtype} '
              f'min={img.min():.3f} max={img.max():.3f}')
        print(f'[check]   cmr frames: {tuple(cmr.shape)} dtype={cmr.dtype} '
              f'min={cmr.min():.3f} max={cmr.max():.3f}')
