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
    labels: torch.Tensor,
    tau: float = 0.1,
    eps: float = 1e-8
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
    
    # 计算余弦相似度矩阵
    # cos(z_j^C, z_k^R) for all j, k in batch
    cosine_sim = torch.matmul(z_C, z_R.T)  # (batch_size, batch_size)
    
    # 计算温度缩放后的相似度
    sim_scaled = cosine_sim / tau  # (batch_size, batch_size)
    
    # 构建正样本掩码：P(j) = {k ∈ B : y_k = y_j}
    # labels: (batch_size,)
    labels_expanded = labels.unsqueeze(0)  # (1, batch_size)
    positive_mask = (labels_expanded == labels_expanded.T).float()  # (batch_size, batch_size)
    
    # 计算心脏→视网膜损失 L_C
    # L_C = -1/N * sum_{j∈B} log(
    #   (sum_{k∈P(j)} exp(cos(z_j^C, z_k^R)/τ)) /
    #   (sum_{k∈B} exp(cos(z_j^C, z_k^R)/τ))
    # )
    
    # 计算 exp(cos(z_j^C, z_k^R)/τ)
    exp_sim = torch.exp(sim_scaled)  # (batch_size, batch_size)
    
    # 对于每个样本j，计算正样本的exp和
    positive_exp_sum = torch.sum(positive_mask * exp_sim, dim=1, keepdim=True)  # (batch_size, 1)
    
    # 对于每个样本j，计算所有样本的exp和（分母）
    all_exp_sum = torch.sum(exp_sim, dim=1, keepdim=True)  # (batch_size, 1)
    
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
    
    # 计算余弦相似度矩阵（转置）
    cosine_sim_R = torch.matmul(z_R, z_C.T)  # (batch_size, batch_size)
    sim_scaled_R = cosine_sim_R / tau
    
    # 计算 exp(cos(z_j^R, z_k^C)/τ)
    exp_sim_R = torch.exp(sim_scaled_R)
    
    # 正样本的exp和
    positive_exp_sum_R = torch.sum(positive_mask * exp_sim_R, dim=1, keepdim=True)
    
    # 所有样本的exp和
    all_exp_sum_R = torch.sum(exp_sim_R, dim=1, keepdim=True)
    
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
    
    def __init__(self, tau: float = 0.1, eps: float = 1e-8):
        """
        Args:
            tau: 温度参数（默认0.1）
            eps: 数值稳定性的小常数
        """
        super(CrossModalContrastiveLoss, self).__init__()
        self.tau = tau
        self.eps = eps
    
    def forward(
        self,
        z_R: torch.Tensor,
        z_C: torch.Tensor,
        labels: torch.Tensor
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
        return cross_modal_contrastive_loss(z_R, z_C, labels, tau=self.tau, eps=self.eps)
    
    def __call__(
        self,
        z_R: torch.Tensor,
        z_C: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.forward(z_R, z_C, labels)


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

