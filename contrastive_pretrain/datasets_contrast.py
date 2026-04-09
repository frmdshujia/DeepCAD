"""
datasets_contrast.py
数据集与采样器：
  - FundusContrastDataset  : 从 fundus_table.csv 读取，返回 (image, eid, pc_vector)
  - UniqueEIDSampler       : 保证同一 batch 内无重复 EID
  - CMRBank                : 全量 CMR PC scores 加载到 GPU，按需采样
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import transforms
from PIL import Image, ImageFile
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FundusContrastDataset(Dataset):
    """
    读取 fundus_table.csv，按 split 过滤，返回 (image_tensor, eid, pc_vector)。
    训练时使用大幅数据增强（含旋转/翻转，保证部署时任意方向的鲁棒性）；
    验证/测试时只做 Resize + CenterCrop。
    """

    def __init__(self, csv_path: str, pc_cols: list, split: str = 'train'):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f'No samples found for split={split} in {csv_path}')

        missing_cols = [c for c in pc_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f'PC columns missing from fundus CSV: {missing_cols}')

        self.image_paths = df['fundus_image_path'].tolist()
        self.eids = df['eid'].tolist()
        self.pc_vectors = df[pc_cols].values.astype(np.float32)  # (N, 14)
        self.is_train = (split == 'train')
        self.transform = self._build_transform()

        print(f'[FundusContrastDataset] split={split}, samples={len(df)}, '
              f'unique_eids={len(set(self.eids))}')

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        eid = self.eids[idx]
        pc = torch.tensor(self.pc_vectors[idx], dtype=torch.float32)

        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return image, eid, pc

    def _build_transform(self):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD

        if self.is_train:
            # 强增强：含 ±180° 旋转 + 双向翻转，让 encoder 对图像方向不敏感
            # （上传测试时用户可能旋转图像，需保证预测稳定）
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3/4, 4/3)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                       saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0))
                ], p=0.5),
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


class UniqueEIDSampler(Sampler):
    """
    自定义 Sampler：每个 epoch 内每个 EID 只出现一次（随机选择其中一张图），
    从而保证同一 batch 内不会有同人的左右眼同时出现。

    epoch 结束后调用 set_epoch(epoch) 更新随机种子，保证每 epoch 采样的图不同。
    """

    def __init__(self, dataset: FundusContrastDataset, seed: int = 0):
        self.seed = seed
        self.epoch = 0

        # 构建 EID -> 图像索引列表 的映射
        eid_to_indices: dict = {}
        for idx, eid in enumerate(dataset.eids):
            eid_to_indices.setdefault(eid, []).append(idx)

        self.eid_to_indices = eid_to_indices
        self.unique_eids = list(eid_to_indices.keys())

    def __len__(self):
        return len(self.unique_eids)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_eids = rng.permutation(self.unique_eids)

        indices = []
        for eid in shuffled_eids:
            candidates = self.eid_to_indices[eid]
            chosen = int(rng.choice(candidates))
            indices.append(chosen)

        return iter(indices)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class DistributedUniqueEIDSampler(Sampler):
    """
    分布式版本的 UniqueEIDSampler：
    先全局排列所有 EID，再按 rank 切分，保证不同进程覆盖不同 EID。
    """

    def __init__(self, dataset: FundusContrastDataset, num_replicas: int,
                 rank: int, seed: int = 0):
        self.seed = seed
        self.epoch = 0
        self.num_replicas = num_replicas
        self.rank = rank

        eid_to_indices: dict = {}
        for idx, eid in enumerate(dataset.eids):
            eid_to_indices.setdefault(eid, []).append(idx)

        self.eid_to_indices = eid_to_indices
        self.unique_eids = list(eid_to_indices.keys())

        # 每个 rank 分到的 EID 数量（向上取整，末尾补 pad）
        self.num_samples = int(np.ceil(len(self.unique_eids) / num_replicas))
        self.total_size = self.num_samples * num_replicas

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled_eids = list(rng.permutation(self.unique_eids))

        # padding to total_size
        while len(shuffled_eids) < self.total_size:
            shuffled_eids += shuffled_eids
        shuffled_eids = shuffled_eids[:self.total_size]

        # 本 rank 的 EID 切片
        my_eids = shuffled_eids[self.rank:self.total_size:self.num_replicas]

        indices = []
        for eid in my_eids:
            candidates = self.eid_to_indices[eid]
            chosen = int(rng.choice(candidates))
            indices.append(chosen)

        return iter(indices)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


class CMRBank:
    """
    将全量 CMR PC score 矩阵加载到指定设备（GPU/CPU）。
    每次调用 sample(k) 随机采样 K 个，返回其 PC score 向量。
    CMR MLP 极小（14→128→d），每 batch 用 fresh forward 计算梯度，无需 epoch 级缓存。
    """

    def __init__(self, csv_path: str, pc_cols: list, split: str = 'train',
                 device: str = 'cuda'):
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f'No CMR samples found for split={split} in {csv_path}')

        missing_cols = [c for c in pc_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f'PC columns missing from CMR CSV: {missing_cols}')

        self.pc_matrix = torch.tensor(
            df[pc_cols].values.astype(np.float32)
        ).to(device)                    # (N_cmr, n_pc)
        self.eids = df['eid'].tolist()
        self.n_cmr = len(df)
        self.device = device

        print(f'[CMRBank] split={split}, n_cmr={self.n_cmr}, '
              f'shape={self.pc_matrix.shape}, device={device}')

    def sample(self, k: int):
        """随机采样 K 个 CMR，返回 (indices, pc_scores)。
           indices : LongTensor (K,)
           pc_scores: FloatTensor (K, n_pc)  — 留在 GPU 上
        """
        indices = torch.randint(0, self.n_cmr, (k,), device=self.device)
        return indices, self.pc_matrix[indices]

    def get_all(self):
        """返回全量 PC score 矩阵，用于 epoch 结束后的评估。"""
        return self.pc_matrix
