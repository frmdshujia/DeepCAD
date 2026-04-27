from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def build_transforms(input_size: int, is_train: bool):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    if is_train:
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.RandomGrayscale(p=0.2),
                transforms.ColorJitter(),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0))],
                    p=0.2,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )
    if input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(input_size / crop_pct)
    return transforms.Compose(
        [
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


class Stage1BinaryDataset(Dataset):
    """返回 (path_str, image_tensor, int_label)，与 engine_finetune.train_one_epoch 解包兼容。"""

    def __init__(
        self,
        frame: pd.DataFrame,
        target_col: str,
        is_train: bool,
        input_size: int = 224,
        balance_train: bool = False,
        seed: int = 0,
    ):
        self.frame = frame.reset_index(drop=True)
        self.target_col = target_col
        self.balance_train = balance_train and is_train
        self.seed = seed

        paths = self.frame["fundus_image_path"].tolist()
        labels = self.frame[target_col].astype(np.int64).values

        if self.balance_train:
            pos_idx = np.where(labels == 1)[0]
            neg_idx = np.where(labels == 0)[0]
            rng = np.random.default_rng(seed)
            if len(pos_idx) == 0 or len(neg_idx) == 0:
                pass
            elif len(neg_idx) > len(pos_idx):
                neg_idx = rng.choice(neg_idx, size=len(pos_idx), replace=False)
                sel = np.sort(np.concatenate([pos_idx, neg_idx]))
                paths = [paths[i] for i in sel]
                labels = labels[sel]
            elif len(pos_idx) > len(neg_idx):
                pos_idx = rng.choice(pos_idx, size=len(neg_idx), replace=False)
                sel = np.sort(np.concatenate([pos_idx, neg_idx]))
                paths = [paths[i] for i in sel]
                labels = labels[sel]

        self.paths = paths
        self.labels = labels.astype(np.int64)
        self.transform = build_transforms(input_size, is_train)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        y = int(self.labels[idx])
        image = Image.open(p).convert("RGB")
        image = self.transform(image)
        return p, image, y
