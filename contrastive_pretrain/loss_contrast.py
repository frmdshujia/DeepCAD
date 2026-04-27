"""
loss_contrast.py
对比损失函数库：
  - hard_infonce_loss          : 标准 InfoNCE，正样本固定在前 B 位（位置 i → 行 i）
  - soft_infonce_loss          : Soft-label InfoNCE，target = softmax(S_GT / τ_g)
  - soft_infonce_loss_with_sgt : 一步完成在线 S_GT 计算 + soft InfoNCE
  - compute_sgt                : 高斯核 GT 相似度矩阵
"""

import torch
import torch.nn.functional as F


def compute_sgt(pc_fundus: torch.Tensor, pc_cmr: torch.Tensor,
                sigma: float) -> torch.Tensor:
    """
    在线计算 Ground-Truth Similarity 矩阵（高斯核）。

    Returns:
        S_GT : (B, K)  ∈ [0, 1]，值越大表示两人心脏表型越相似
        公式：S(A,B) = exp( -‖p_A − p_B‖² / (2σ²) )
    """
    dists = torch.cdist(pc_fundus, pc_cmr, p=2)   # (B, K)
    s_gt = torch.exp(-dists.pow(2) / (2.0 * sigma ** 2))
    return s_gt                                     # (B, K)


def hard_infonce_loss(z_fundus: torch.Tensor,
                      z_cmr: torch.Tensor,
                      temperature: float = 0.07) -> torch.Tensor:
    """
    标准（硬标签）InfoNCE：对 fundus_i，正样本是 cmr_i（前 B 位），其余 K-1 个为负样本。

    前提：调用方通过 sample_with_batch_positives 保证 z_cmr[:B] 与 z_fundus[:B] 一一配对。

    Args:
        z_fundus    : (B, d) L2-normalized fundus embeddings
        z_cmr       : (K, d) L2-normalized CMR embeddings，前 B 行为配对正样本
        temperature : InfoNCE 温度（默认 0.07）

    Returns:
        loss : scalar
    """
    B = z_fundus.size(0)
    logits = z_fundus @ z_cmr.T / temperature    # (B, K)
    labels = torch.arange(B, device=z_fundus.device)  # 正样本位置 = 0, 1, ..., B-1
    return F.cross_entropy(logits, labels)


def soft_infonce_loss(z_fundus: torch.Tensor,
                      z_cmr: torch.Tensor,
                      s_gt: torch.Tensor,
                      temperature: float = 0.07,
                      sgt_temp: float = 1.0) -> torch.Tensor:
    """
    Soft-label InfoNCE（参考 SCE, WACV 2023）。

    核心思路：
      - 传统 InfoNCE 以 one-hot label 做 cross-entropy（只有正样本贡献梯度）
      - 这里改用 softmax(S_GT / sgt_temp) 作为连续目标分布，
        所有 K 个 CMR 都以其与 fundus 的心脏表型相似度按比例贡献

    Args:
        z_fundus    : (B, d) L2-normalized fundus embeddings
        z_cmr       : (K, d) L2-normalized CMR embeddings
        s_gt        : (B, K) 高斯核 GT 相似度（未 softmax）
        temperature : 嵌入 logits 的温度（默认 0.07）
        sgt_temp    : 目标锐化温度 τ_g：target = softmax(S_GT/τ_g)。1.0 与旧版一致；小于 1 则目标更尖。

    Returns:
        loss : scalar，逐行 cross-entropy 的 batch 均值
    """
    # ① 余弦相似度矩阵，除以温度
    logits = z_fundus @ z_cmr.T / temperature    # (B, K)

    # ② 将 S_GT 归一化为目标概率分布（soft label）；τ_g<1 时锐化
    st = max(float(sgt_temp), 1e-6)
    target = F.softmax(s_gt / st, dim=-1)  # (B, K)，每行和为 1

    # ③ 逐行交叉熵 = -∑ target * log_softmax(logits)
    log_probs = F.log_softmax(logits, dim=-1)    # (B, K)
    loss = -(target * log_probs).sum(dim=-1)     # (B,) 每行的 CE
    return loss.mean()                           # scalar


def soft_infonce_loss_with_sgt(z_fundus: torch.Tensor,
                                z_cmr: torch.Tensor,
                                pc_fundus: torch.Tensor,
                                pc_cmr: torch.Tensor,
                                sigma: float,
                                temperature: float = 0.07,
                                sgt_temp: float = 1.0):
    """
    一步完成：在线计算 S_GT 并求 soft InfoNCE loss。
    训练循环中直接调用此函数，不需要在外部单独计算 S_GT。

    Returns:
        loss  : scalar
        s_gt  : (B, K) 供调试/监控用
    """
    s_gt = compute_sgt(pc_fundus, pc_cmr, sigma)
    loss = soft_infonce_loss(
        z_fundus, z_cmr, s_gt, temperature, sgt_temp=sgt_temp)
    return loss, s_gt
