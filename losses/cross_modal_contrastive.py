"""
监督跨模态对比损失
实现 DeepCAD Stage I 的监督跨模态对比学习损失函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def cross_modal_contrastive_loss(
    z_R: torch.Tensor,
    z_C: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
    tau: float = 0.1,
    eps: float = 1e-8,
    positive_mask: Optional[torch.Tensor] = None,
    training_mode: str = "grade",
    subject_ids: Optional[torch.Tensor] = None,
    # 统一的正样本 key（batch 内）
    pos_keys: Optional[torch.Tensor] = None,
    # 队列中的历史特征与对应 key（可选，用于扩展负样本，并避免 false negative）
    queue_R_feats: Optional[torch.Tensor] = None,
    queue_R_keys: Optional[torch.Tensor] = None,
    queue_C_feats: Optional[torch.Tensor] = None,
    queue_C_keys: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    监督跨模态对比损失
    
    严格按照数学定义实现：
    - 心脏→视网膜损失 L_C
    - 视网膜→心脏损失 L_R
    - 总损失 L = L_C + L_R
    
    Args:
        z_R: 视网膜投影，形状为 (batch_size, latent_dim)，已L2归一化
             其中 latent_dim 默认值为 128 (共享潜在空间维度)
        z_C: 心脏MRI投影，形状为 (batch_size, latent_dim)，已L2归一化
             维度与 z_R 相同，确保可以计算跨模态相似度
        labels: CAD标签，形状为 (batch_size,)，值为0或1
        tau: 温度参数（默认0.1）
             来源: DeepCAD_README_with_prompts.md 中的数学定义
             说明: 温度参数用于缩放相似度分数，较小的值（0.1）使模型更关注困难样本
                   这是对比学习中常用的设置，参考了 SimCLR、CLIP 等工作的经验
                   可调整范围: 0.05 (更关注困难样本) 到 0.5 (更平滑的分布)
        eps: 数值稳定性的小常数（默认1e-8）
             用于避免 log(0) 导致的数值问题
    
    Returns:
        Tuple[总损失, L_C, L_R]
    """
    batch_size = z_R.shape[0]
    device = z_R.device
    
    # 确保输入已归一化
    z_R = F.normalize(z_R, p=2, dim=1)
    z_C = F.normalize(z_C, p=2, dim=1)
    
    # 计算余弦相似度矩阵 (C -> R 方向)
    cosine_sim = torch.matmul(z_C, z_R.T)  # (batch_size, batch_size)
    sim_scaled = cosine_sim / tau  # (batch_size, batch_size)

    # 根据 training_mode 或统一的 pos_keys 构造正样本掩码（如果未显式提供）
    if positive_mask is None:
        if pos_keys is not None:
            # ✅ 新推荐路径：使用统一的 pos_keys（批内谁 key 相同，谁就是正样本）
            pos_keys = pos_keys.to(device)
            positive_mask = (pos_keys.unsqueeze(0) == pos_keys.unsqueeze(1)).float()
        else:
            # ⚠️ 兼容旧路径：根据 training_mode + labels/subject_ids 构造
            if training_mode == "subject":
                if subject_ids is None:
                    raise ValueError("subject_ids must be provided when training_mode='subject'.")
                unique_map = {}
                encoded = []
                for sid in subject_ids:
                    sid_str = str(sid)
                    if sid_str not in unique_map:
                        unique_map[sid_str] = len(unique_map)
                    encoded.append(unique_map[sid_str])
                subj_tensor = torch.tensor(encoded, device=device)
                positive_mask = (subj_tensor.unsqueeze(0) == subj_tensor.unsqueeze(1)).float()
            else:
                if labels is None:
                    raise ValueError(
                        "labels must be provided when positive_mask is None and "
                        "training_mode is not 'subject' and pos_keys is None."
                    )
                positive_mask = compute_positive_mask(labels).to(device)
    else:
        positive_mask = positive_mask.to(device)
    
    # 计算心脏→视网膜损失 L_C
    # L_C = -1/N * sum_{j∈B} log(
    #   (sum_{k∈P(j)} exp(cos(z_j^C, z_k^R)/τ)) /
    #   (sum_{k∈B} exp(cos(z_j^C, z_k^R)/τ))
    # )
    
    # 计算 exp(cos(z_j^C, z_k^R)/τ)
    exp_sim = torch.exp(sim_scaled)  # (batch_size, batch_size)

    # 如果提供了视网膜队列 queue_R_feats，则将其作为额外负样本（只出现在分母中）
    # 同时利用 queue_R_keys 与 pos_keys，剔除“正样本”（避免 false negatives）
    exp_sim_queue = None
    if queue_R_feats is not None and queue_R_feats.numel() > 0:
        queue_R_feats = F.normalize(queue_R_feats.to(device), p=2, dim=1)  # (K_R, D)
        cosine_sim_queue = torch.matmul(z_C, queue_R_feats.T)  # (B, K_R)
        sim_scaled_queue = cosine_sim_queue / tau
        exp_sim_queue_all = torch.exp(sim_scaled_queue)  # (B, K_R)

        if pos_keys is not None and queue_R_keys is not None:
            queue_R_keys = queue_R_keys.to(device)
            # queue_pos_mask[i, k] = 1 表示：第 i 个 anchor 与队列中第 k 个元素是正样本（key 相同）
            queue_pos_mask = (pos_keys.unsqueeze(1) == queue_R_keys.unsqueeze(0)).float()  # (B, K_R)
            queue_neg_mask = 1.0 - queue_pos_mask
            exp_sim_queue = exp_sim_queue_all * queue_neg_mask
        else:
            # 如果缺少 key 信息，则退化为“全部当负样本”
            exp_sim_queue = exp_sim_queue_all
    
    # 对于每个样本j，计算正样本的exp和
    positive_exp_sum = torch.sum(positive_mask * exp_sim, dim=1, keepdim=True)  # (batch_size, 1)
    
    # 对于每个样本j，计算所有样本的exp和（分母）
    all_exp_sum = torch.sum(exp_sim, dim=1, keepdim=True)  # (batch_size, 1)
    if exp_sim_queue is not None:
        all_exp_sum = all_exp_sum + torch.sum(exp_sim_queue, dim=1, keepdim=True)  # (B, 1)
    
    # 计算 log(positive_exp_sum / all_exp_sum)
    # 添加eps避免数值不稳定
    log_prob_C = torch.log(positive_exp_sum + eps) - torch.log(all_exp_sum + eps)  # (batch_size, 1)
    
    # 平均损失
    L_C = -torch.mean(log_prob_C)
    
    # 计算视网膜→心脏损失 L_R（对称）
    # L_R = -1/N * sum_{j∈B} log(
    #   (sum_{k∈P(j)} exp(cos(z_j^R, z_k^C)/τ)) /
    #   (sum_{k∈B} exp(cos(z_j^R, z_k^C)/τ))
    # )
    
    # 计算余弦相似度矩阵（R -> C 方向）
    cosine_sim_R = torch.matmul(z_R, z_C.T)  # (batch_size, batch_size)
    sim_scaled_R = cosine_sim_R / tau

    # 计算 exp(cos(z_j^R, z_k^C)/τ)
    exp_sim_R = torch.exp(sim_scaled_R)

    # 如果提供了心脏队列 queue_C_feats，则将其作为额外负样本（只出现在分母中）
    exp_sim_queue_R = None
    if queue_C_feats is not None and queue_C_feats.numel() > 0:
        queue_C_feats = F.normalize(queue_C_feats.to(device), p=2, dim=1)  # (K_C, D)
        cosine_sim_queue_R_all = torch.matmul(z_R, queue_C_feats.T)  # (B, K_C)
        sim_scaled_queue_R_all = cosine_sim_queue_R_all / tau
        exp_sim_queue_R_all = torch.exp(sim_scaled_queue_R_all)  # (B, K_C)

        if pos_keys is not None and queue_C_keys is not None:
            queue_C_keys = queue_C_keys.to(device)
            queue_pos_mask_R = (pos_keys.unsqueeze(1) == queue_C_keys.unsqueeze(0)).float()  # (B, K_C)
            queue_neg_mask_R = 1.0 - queue_pos_mask_R
            exp_sim_queue_R = exp_sim_queue_R_all * queue_neg_mask_R
        else:
            exp_sim_queue_R = exp_sim_queue_R_all
    
    # 正样本的exp和
    positive_exp_sum_R = torch.sum(positive_mask * exp_sim_R, dim=1, keepdim=True)
    
    # 所有样本的exp和
    all_exp_sum_R = torch.sum(exp_sim_R, dim=1, keepdim=True)
    if exp_sim_queue_R is not None:
        all_exp_sum_R = all_exp_sum_R + torch.sum(exp_sim_queue_R, dim=1, keepdim=True)
    
    # 计算 log(positive_exp_sum_R / all_exp_sum_R)
    log_prob_R = torch.log(positive_exp_sum_R + eps) - torch.log(all_exp_sum_R + eps)
    
    # 平均损失
    L_R = -torch.mean(log_prob_R)
    
    # 总损失
    L = L_C + L_R
    
    return L, L_C, L_R


class CrossModalContrastiveLoss(nn.Module):
    """
    监督跨模态对比损失（PyTorch Module版本）
    """
    
    def __init__(self, tau: float = 0.1, eps: float = 1e-8, training_mode: str = "grade"):
        """
        Args:
            tau: 温度参数（默认0.1）
            eps: 数值稳定性的小常数
        """
        super(CrossModalContrastiveLoss, self).__init__()
        self.tau = tau
        self.eps = eps
        self.training_mode = training_mode
    
    def forward(
        self,
        z_R: torch.Tensor,
        z_C: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        pos_keys: Optional[torch.Tensor] = None,
        queue_R_feats: Optional[torch.Tensor] = None,
        queue_R_keys: Optional[torch.Tensor] = None,
        queue_C_feats: Optional[torch.Tensor] = None,
        queue_C_keys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            z_R: 视网膜投影，形状为 (batch_size, latent_dim)
            z_C: 心脏MRI投影，形状为 (batch_size, latent_dim)
            labels: CAD标签，形状为 (batch_size,)
        
        Returns:
            Tuple[总损失, L_C, L_R]
        """
        return cross_modal_contrastive_loss(
            z_R,
            z_C,
            labels,
            tau=self.tau,
            eps=self.eps,
            positive_mask=positive_mask,
            training_mode=self.training_mode,
            subject_ids=subject_ids,
            pos_keys=pos_keys,
            queue_R_feats=queue_R_feats,
            queue_R_keys=queue_R_keys,
            queue_C_feats=queue_C_feats,
            queue_C_keys=queue_C_keys,
        )
    
    def __call__(
        self,
        z_R: torch.Tensor,
        z_C: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        positive_mask: Optional[torch.Tensor] = None,
        subject_ids: Optional[torch.Tensor] = None,
        pos_keys: Optional[torch.Tensor] = None,
        queue_R_feats: Optional[torch.Tensor] = None,
        queue_R_keys: Optional[torch.Tensor] = None,
        queue_C_feats: Optional[torch.Tensor] = None,
        queue_C_keys: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(
            z_R,
            z_C,
            labels,
            positive_mask,
            subject_ids,
            pos_keys,
            queue_R_feats,
            queue_R_keys,
            queue_C_feats,
            queue_C_keys,
        )


def compute_positive_mask(labels: torch.Tensor) -> torch.Tensor:
    """
    计算正样本掩码
    
    Args:
        labels: 标签，形状为 (batch_size,)
    
    Returns:
        正样本掩码，形状为 (batch_size, batch_size)
        mask[i, j] = 1 如果 labels[i] == labels[j]，否则为 0
    """
    labels_expanded = labels.unsqueeze(0)  # (1, batch_size)
    positive_mask = (labels_expanded == labels_expanded.T).float()  # (batch_size, batch_size)
    return positive_mask


def cosine_similarity_matrix(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """
    计算两个嵌入集合之间的余弦相似度矩阵
    
    Args:
        z1: 嵌入1，形状为 (N, D)
        z2: 嵌入2，形状为 (M, D)
    
    Returns:
        余弦相似度矩阵，形状为 (N, M)
    """
    # 确保已归一化
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    
    # 计算余弦相似度
    sim = torch.matmul(z1, z2.T)
    return sim
