from __future__ import annotations

"""
datasets_contrast.py
数据集与采样器：
  - FundusContrastDataset  : 从 fundus_table.csv 读取，返回 (image, eid, pc_vector)
  - UniqueEIDSampler       : 保证同一 batch 内无重复 EID
  - CMRBank                : 全量 CMR PC scores 加载到 GPU，按需采样
"""

import os
import pickle
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

    def __init__(
        self,
        csv_path: str,
        pc_cols: list,
        split: str = 'train',
        train_subset_ratio: float = 1.0,
        train_max_samples: int = 0,
        subset_seed: int = 0,
        use_eval_transform: bool = False,
    ):
        """
        use_eval_transform: True 时即使 split=='train' 也用验证集同款确定性预处理
        （线性探针等需在 train 上训练头但希望与 val 一致的增广策略时使用）。

        train_subset_ratio / train_max_samples / subset_seed 仅当 split=='train' 时生效：
          - train_max_samples > 0 时优先：随机最多保留这么多行；
          - 否则若 train_subset_ratio < 1.0：按比例随机子采样。
        验证/测试集默认始终全量（便于稳定监控）。
        """
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f'No samples found for split={split} in {csv_path}')

        missing_cols = [c for c in pc_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f'PC columns missing from fundus CSV: {missing_cols}')

        # 过滤掉图像文件尚未上传到服务器的行（等上传完毕后自动生效，无需修改代码）
        n_total = len(df)
        exist_mask = df['fundus_image_path'].apply(os.path.exists)
        n_missing = int((~exist_mask).sum())
        if n_missing > 0:
            df = df[exist_mask].reset_index(drop=True)
            print(f'[FundusContrastDataset] split={split}: '
                  f'{n_missing}/{n_total} 行图像尚未上传，已跳过，等上传完毕后自动纳入。')

        if len(df) == 0:
            raise ValueError(f'split={split} 中所有图像路径均不存在，无法构建 dataset。')

        # 训练集子采样（保守探索 / 烟雾测）
        if split == 'train':
            n_before = len(df)
            n_take = n_before
            if train_max_samples and train_max_samples > 0:
                n_take = min(int(train_max_samples), n_before)
            elif train_subset_ratio < 1.0:
                n_take = max(1, int(n_before * train_subset_ratio))
            if n_take < n_before:
                rng = np.random.default_rng(subset_seed)
                idx = rng.choice(n_before, size=n_take, replace=False)
                df = df.iloc[idx].reset_index(drop=True)
                print(f'[FundusContrastDataset] train 子采样: {n_take}/{n_before} 行 '
                      f'(ratio={train_subset_ratio}, max_samples={train_max_samples}, seed={subset_seed})')

        self.image_paths = df['fundus_image_path'].tolist()
        self.eids = df['eid'].tolist()
        self.pc_vectors = df[pc_cols].values.astype(np.float32)  # (N, 14)
        self.is_train = (split == 'train') and (not use_eval_transform)
        self.transform = self._build_transform()

        print(f'[FundusContrastDataset] split={split}: '
              f'可用样本={len(df)}, 唯一EID={len(set(self.eids))}')

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

    def __init__(
        self,
        csv_path: str,
        pc_cols: list,
        split: str = 'train',
        device: str = 'cuda',
        max_rows: int = 0,
        subset_seed: int = 0,
        stratified_buckets_path: str | None = None,
        strat_neg_frac_high: float = 1.0 / 3.0,
        strat_neg_frac_low: float = 1.0 / 3.0,
        stratified_rng_rank: int = 0,
    ):
        """
        max_rows > 0 时：从该 split 中随机最多保留 max_rows 条（节省显存、加快烟雾测）。

        stratified_buckets_path: precompute_stratified_buckets.py 生成的 pickle。
          与 max_rows>0 互斥（分层索引按全表行号）；分层实验请 cmr_train_max_rows=0。
        """
        df = pd.read_csv(csv_path)
        df = df[df['split'] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(f'No CMR samples found for split={split} in {csv_path}')

        missing_cols = [c for c in pc_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f'PC columns missing from CMR CSV: {missing_cols}')

        # 子采样 bank 前：为每个 EID 保留「全表」中首次出现的 PC 向量，供 batch 正样本槽位使用
        # （否则 max_rows 随机子集可能不含某 fundus 对应 EID，导致无法对齐）
        arr_full = df[pc_cols].values.astype(np.float32)
        eid_full = df['eid'].tolist()
        self._eid_to_paired_pc = {}
        for i, eid in enumerate(eid_full):
            k_e = CMRBank._normalize_eid_key(eid)
            if k_e not in self._eid_to_paired_pc:
                self._eid_to_paired_pc[k_e] = torch.tensor(
                    arr_full[i], dtype=torch.float32, device=device)

        n_before = len(df)
        subset_choice = None
        if max_rows and max_rows > 0 and n_before > max_rows:
            if stratified_buckets_path:
                raise ValueError(
                    'stratified_buckets_path 不能与 cmr_train_max_rows>0 同时使用；'
                    '分层采样需全量 CMR bank（cmr_train_max_rows=0）。'
                )
            rng = np.random.default_rng(int(subset_seed) + 910001)
            subset_choice = rng.choice(n_before, size=max_rows, replace=False)
            df = df.iloc[subset_choice].reset_index(drop=True)
            print(f'[CMRBank] split={split} 子采样: {len(df)}/{n_before} 行 (max_rows={max_rows})')

        self.pc_matrix = torch.tensor(
            df[pc_cols].values.astype(np.float32)
        ).to(device)                    # (N_cmr, n_pc)
        self.eids = df['eid'].tolist()
        self.n_cmr = len(df)
        self.device = device
        self._split = split

        self._full_to_bank_row = np.arange(n_before, dtype=np.int64)
        if subset_choice is not None:
            full_to_bank = np.full(n_before, -1, dtype=np.int64)
            for bank_row, full_idx in enumerate(subset_choice.tolist()):
                full_to_bank[int(full_idx)] = bank_row
            self._full_to_bank_row = full_to_bank

        self.strat_neg_frac_high = float(strat_neg_frac_high)
        self.strat_neg_frac_low = float(strat_neg_frac_low)
        self._stratified_buckets: dict | None = None
        self._strat_rng = np.random.default_rng(
            int(subset_seed) + 777 + int(stratified_rng_rank) * 100003
        )
        if stratified_buckets_path:
            if split != 'train':
                raise ValueError('stratified_buckets_path 仅用于 train CMRBank')
            with open(stratified_buckets_path, 'rb') as f:
                payload = pickle.load(f)
            meta = payload['meta']
            if int(meta['n_cmr_train']) != n_before:
                raise ValueError(
                    f"分桶 n_cmr_train={meta['n_cmr_train']} != 当前 train CMR 行数 {n_before}"
                )
            self._stratified_buckets = payload['buckets']
            print(f'[CMRBank] 分层分桶已加载: {stratified_buckets_path} ({len(self._stratified_buckets)} EIDs)')

        # EID -> bank 行号（同人多条 CMR 时取首行，与正样本定义一致）
        self._eid_to_row = {}
        for row, eid in enumerate(self.eids):
            k_e = self._normalize_eid_key(eid)
            if k_e not in self._eid_to_row:
                self._eid_to_row[k_e] = row

        print(f'[CMRBank] split={split}, n_cmr={self.n_cmr}, '
              f'shape={self.pc_matrix.shape}, device={device}, '
              f'unique_eid_in_bank={len(self._eid_to_row)}, '
              f'paired_eid_table={len(self._eid_to_paired_pc)}')

    @staticmethod
    def _normalize_eid_key(eid):
        """与 batch 中 eid 比较时统一键类型（避免 int/np.int64/str 不一致）。"""
        if isinstance(eid, torch.Tensor):
            eid = eid.detach().cpu().item()
        if isinstance(eid, (bytes, str)):
            return str(eid).strip()
        if isinstance(eid, (np.floating, float)) and float(eid).is_integer():
            return int(eid)
        if isinstance(eid, np.integer):
            return int(eid)
        if isinstance(eid, int):
            return eid
        return int(eid)

    def sample(self, k: int):
        """随机采样 K 个 CMR，返回 (indices, pc_scores)。
           indices : LongTensor (K,)
           pc_scores: FloatTensor (K, n_pc)  — 留在 GPU 上
        """
        indices = torch.randint(0, self.n_cmr, (k,), device=self.device)
        return indices, self.pc_matrix[indices]

    def _sample_negative_indices(self, exclude_rows: torch.Tensor, n: int) -> torch.Tensor:
        """从 bank 中采 n 个行号，尽量不含 exclude_rows（正样本行），无放回。
        使用 CPU/numpy 采样以避免旧版 PyTorch CUDA randperm 的潜在数值问题。
        """
        if exclude_rows.numel() > 0:
            exc_arr = np.unique(exclude_rows.cpu().numpy().astype(np.int64))
            exc_arr = exc_arr[(exc_arr >= 0) & (exc_arr < self.n_cmr)]
            avail = np.setdiff1d(np.arange(self.n_cmr, dtype=np.int64), exc_arr)
        else:
            avail = np.arange(self.n_cmr, dtype=np.int64)

        if len(avail) < n:
            return torch.randint(0, self.n_cmr, (n,), device=self.device)

        idx = np.random.choice(avail, size=n, replace=False)
        return torch.tensor(idx, device=self.device, dtype=torch.long)

    def sample_with_batch_positives(self, batch_eids, k: int):
        """
        构造长度为 K 的 CMR 集合，保证与 batch 顺序对齐：
          - 第 0..B-1 行为各 fundus 对应 EID 在 CMR bank 中的正样本 PC；
          - 第 B..K-1 行为随机负样本（默认排除正样本行，避免重复占位）。

        这样 soft-label InfoNCE 的 S_GT 在「同人」位置必有真实 CMR，
        且 logits 矩阵第 i 行与第 i 列对齐为跨模态正样本对。

        Args:
            batch_eids: 长度 B 的可迭代对象（与 DataLoader 中 eid 列一致）
            k: 总 CMR 条数，必须满足 k > B（B = len(batch_eids)）

        Returns:
            indices : LongTensor (K,)
            pc_scores: FloatTensor (K, n_pc)
        """
        keys = [self._normalize_eid_key(e) for e in batch_eids]
        B = len(keys)
        if k <= B:
            raise ValueError(
                f'cmr_sample_k ({k}) must be > batch size ({B}) '
                f'when using paired positives in the first B slots.'
            )

        pos_rows = []
        for key in keys:
            if key not in self._eid_to_paired_pc:
                raise KeyError(
                    f'EID {key!r} not found in CMR paired lookup '
                    f'(split={self._split!r}). '
                    f'Ensure fundus EIDs exist in the same CMR CSV split.'
                )
            pos_rows.append(self._eid_to_paired_pc[key])
        pos_pc = torch.stack(pos_rows, dim=0)  # (B, n_pc)

        neg_n = k - B
        # 负样本从子采样 bank 抽取；若某 EID 仍在 bank 中有行，则排除该行以免与正样本重复占位
        exclude_list = []
        for key in keys:
            if key in self._eid_to_row:
                exclude_list.append(self._eid_to_row[key])
        if exclude_list:
            exclude_rows = torch.tensor(
                exclude_list, device=self.device, dtype=torch.long).unique()
        else:
            exclude_rows = torch.empty(0, dtype=torch.long, device=self.device)

        neg_indices = self._sample_negative_indices(exclude_rows, neg_n)
        pc_neg = self.pc_matrix[neg_indices]
        pc_cmr = torch.cat([pos_pc, pc_neg], dim=0)

        # 前 B 个为「全表配对」正样本，无 bank 行号，记为 -1
        indices = torch.cat([
            torch.full((B,), -1, device=self.device, dtype=torch.long),
            neg_indices,
        ], dim=0)
        return indices, pc_cmr

    def _pool_to_bank_rows(self, keys: list, stratum: str, exclude_set: set) -> list:
        """将 batch 内各 EID 在 stratum 下的全表索引映射为 bank 行号，去重并过滤 exclude。"""
        out = []
        if not self._stratified_buckets:
            return out
        seen = set()
        for key in keys:
            b = self._stratified_buckets.get(key)
            if not b:
                continue
            for full_idx in b.get(stratum, np.array([], dtype=np.int64)):
                br = int(self._full_to_bank_row[int(full_idx)])
                if br < 0 or br in exclude_set or br in seen:
                    continue
                seen.add(br)
                out.append(br)
        return out

    def sample_with_batch_positives_stratified(self, batch_eids, k: int):
        """
        与 sample_with_batch_positives 相同的前 B 正样本；负样本按高/中/低 S_GT 分层配额采样
        （配额比例 strat_neg_frac_high / strat_neg_frac_low，余下为 mid），不足时从 mid→全库随机补齐。
        """
        if not self._stratified_buckets:
            raise RuntimeError('未加载 stratified_buckets_path，无法分层采样')

        keys = [self._normalize_eid_key(e) for e in batch_eids]
        B = len(keys)
        if k <= B:
            raise ValueError(f'cmr_sample_k ({k}) must be > batch size ({B})')

        pos_rows = []
        for key in keys:
            if key not in self._eid_to_paired_pc:
                raise KeyError(
                    f'EID {key!r} not found in CMR paired lookup '
                    f'(split={self._split!r}).'
                )
            pos_rows.append(self._eid_to_paired_pc[key])
        pos_pc = torch.stack(pos_rows, dim=0)

        neg_n = k - B
        exclude_list = []
        for key in keys:
            if key in self._eid_to_row:
                exclude_list.append(self._eid_to_row[key])
        exclude_set = set(exclude_list)

        n_h = int(neg_n * self.strat_neg_frac_high)
        n_l = int(neg_n * self.strat_neg_frac_low)
        n_m = max(0, neg_n - n_h - n_l)

        picked: list = []
        picked_set: set = set()
        rng = self._strat_rng

        def take_from_stratum(name: str, need: int) -> None:
            if need <= 0:
                return
            pool = self._pool_to_bank_rows(keys, name, exclude_set)
            rng.shuffle(pool)
            got = 0
            for br in pool:
                if got >= need:
                    break
                if br in picked_set:
                    continue
                picked.append(br)
                picked_set.add(br)
                got += 1

        take_from_stratum('high', n_h)
        take_from_stratum('low', n_l)
        take_from_stratum('mid', n_m)

        if len(picked) < neg_n:
            banned = np.array(list(exclude_set | picked_set), dtype=np.int64)
            avail = np.setdiff1d(np.arange(self.n_cmr, dtype=np.int64), banned)
            rng.shuffle(avail)
            for br in avail.tolist():
                if len(picked) >= neg_n:
                    break
                bi = int(br)
                if bi in picked_set:
                    continue
                picked.append(bi)
                picked_set.add(bi)

        if len(picked) < neg_n:
            exc_tensor = torch.tensor(
                list(exclude_set | picked_set), device=self.device, dtype=torch.long
            )
            rest = self._sample_negative_indices(exc_tensor, neg_n - len(picked))
            for r in rest.cpu().tolist():
                if len(picked) >= neg_n:
                    break
                if int(r) in picked_set:
                    continue
                picked.append(int(r))
                picked_set.add(int(r))

        picked = picked[:neg_n]
        neg_indices = torch.tensor(picked, device=self.device, dtype=torch.long)
        pc_neg = self.pc_matrix[neg_indices]
        pc_cmr = torch.cat([pos_pc, pc_neg], dim=0)
        indices = torch.cat([
            torch.full((B,), -1, device=self.device, dtype=torch.long),
            neg_indices,
        ], dim=0)
        return indices, pc_cmr

    def get_all(self):
        """返回全量 PC score 矩阵，用于 epoch 结束后的评估。"""
        return self.pc_matrix
