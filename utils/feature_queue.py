"""
简单的特征队列（memory bank）实现

用于在对比学习中提供额外负样本，而不增加 encoder 前向的显存开销。
"""

from typing import Optional

import torch


class FeatureQueue:
    """
    循环队列形式的 memory bank，**同时**存储投影后的 embedding 向量和对应的正样本 key。
    
    设计目的：
    - queue 中的每一条记录都是 (feat, key)
    - key 用于在 loss 中识别正样本（例如 subject_id / label / grade 等）
    - 这样可以在使用队列扩展负样本时，避免把“过去的自己”或“同标签样本”当成错误的负样本
    """
    
    def __init__(self, dim: int, K: int, device: Optional[str] = None, key_dtype: torch.dtype = torch.long):
        """
        Args:
            dim: 特征维度（例如 latent_dim）
            K: 队列长度（memory bank 容量）
            device: 存放队列的设备（默认与当前默认设备一致）
            key_dtype: key 的 dtype（默认 long，一般用于整数索引）
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.K = int(K)
        self.dim = int(dim)
        self.device = device
        
        # 存特征 (K, dim)
        self.buffer = torch.zeros(self.K, self.dim, device=self.device)
        # 存对应的 key (K,)
        self.keys = torch.full((self.K,), fill_value=-1, dtype=key_dtype, device=self.device)
        
        self.ptr = 0
        self.full = False
    
    @torch.no_grad()
    def enqueue(self, feats: torch.Tensor, keys: torch.Tensor):
        """
        将当前 batch 的特征和对应 key 入队（FIFO）。
        
        Args:
            feats: (B, dim) 的张量
            keys:  (B,)     的张量，元素类型通常为 long，表示正样本 key
        """
        if feats.ndim != 2 or feats.shape[1] != self.dim:
            raise ValueError(f"FeatureQueue.enqueue 期望 feats 形状为 (B, {self.dim})，但得到 {tuple(feats.shape)}")
        if keys.ndim != 1 or keys.shape[0] != feats.shape[0]:
            raise ValueError(f"FeatureQueue.enqueue 期望 keys 形状为 (B,)，但得到 {tuple(keys.shape)}")
        
        feats = feats.detach().to(self.device)
        keys = keys.detach().to(self.device)
        B = feats.shape[0]
        
        if B >= self.K:
            # batch 太大时，只保留最后 K 条
            self.buffer.copy_(feats[-self.K :])
            self.keys.copy_(keys[-self.K :])
            self.ptr = 0
            self.full = True
            return
        
        end = self.ptr + B
        if end <= self.K:
            self.buffer[self.ptr : end] = feats
            self.keys[self.ptr : end] = keys
        else:
            first = self.K - self.ptr
            self.buffer[self.ptr :] = feats[:first]
            self.keys[self.ptr :] = keys[:first]
            self.buffer[: B - first] = feats[first:]
            self.keys[: B - first] = keys[first:]
        
        self.ptr = (self.ptr + B) % self.K
        if self.ptr == 0:
            self.full = True
    
    def get(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        返回当前队列中的所有有效特征和对应 key。
        
        Returns:
            feats: (N, dim) 张量，N <= K
            keys:  (N,)     张量
        """
        if self.full or self.ptr == 0:
            return self.buffer, self.keys
        else:
            return self.buffer[: self.ptr], self.keys[: self.ptr]


