"""
loss_contrast.py
Soft-label InfoNCE 损失及在线 S_GT 计算。
全部矩阵化操作，无 for 循环。
"""

import torch
import torch.nn.functional as F


def compute_sgt(pc_fundus: torch.Tensor, pc_cmr: torch.Tensor,
                sigma: float) -> torch.Tensor:
    """
    在线计算 Ground-Truth Similarity 矩阵（高斯核）。

    Args:
        pc_fundus : (B, n_pc)  —— 当前 batch 眼底样本的 PC score
        pc_cmr    : (K, n_pc)  —— 采样到的 K 个 CMR 的 PC score
        sigma     : 高斯核带宽（由数据 Agent 在训练集 CMR 上用 median heuristic 估计）

    Returns:
        S_GT : (B, K)  ∈ [0, 1]，值越大表示两人心脏表型越相似

    公式：
        S(A,B) = exp( -‖p_A − p_B‖² / (2σ²) )
    """
    # torch.cdist 高效计算 B×K 欧氏距离，不展开中间张量
    dists = torch.cdist(pc_fundus, pc_cmr, p=2)   # (B, K)
    s_gt = torch.exp(-dists.pow(2) / (2.0 * sigma ** 2))
    return s_gt                                     # (B, K)


def soft_infonce_loss(z_fundus: torch.Tensor,
                      z_cmr: torch.Tensor,
                      s_gt: torch.Tensor,
                      temperature: float = 0.07) -> torch.Tensor:
    """
    Soft-label InfoNCE（参考 SCE, WACV 2023）。

    核心思路：
      - 传统 InfoNCE 以 one-hot label 做 cross-entropy（只有正样本贡献梯度）
      - 这里改用 softmax(S_GT) 作为连续目标分布，
        所有 K 个 CMR 都以其与 fundus 的心脏表型相似度按比例贡献

    Args:
        z_fundus    : (B, d) L2-normalized fundus embeddings
        z_cmr       : (K, d) L2-normalized CMR embeddings
        s_gt        : (B, K) 高斯核 GT 相似度（未 softmax）
        temperature : InfoNCE 温度参数（默认 0.07）

    Returns:
        loss : scalar，逐行 cross-entropy 的 batch 均值
    """
    # ① 余弦相似度矩阵，除以温度
    logits = z_fundus @ z_cmr.T / temperature    # (B, K)

    # ② 将 S_GT 归一化为目标概率分布（soft label）
    target = F.softmax(s_gt, dim=-1)             # (B, K)，每行和为 1

    # ③ 逐行交叉熵 = -∑ target * log_softmax(logits)
    log_probs = F.log_softmax(logits, dim=-1)    # (B, K)
    loss = -(target * log_probs).sum(dim=-1)     # (B,) 每行的 CE
    return loss.mean()                           # scalar


def soft_infonce_loss_with_sgt(z_fundus: torch.Tensor,
                                z_cmr: torch.Tensor,
                                pc_fundus: torch.Tensor,
                                pc_cmr: torch.Tensor,
                                sigma: float,
                                temperature: float = 0.07):
    """
    一步完成：在线计算 S_GT 并求 soft InfoNCE loss。
    训练循环中直接调用此函数，不需要在外部单独计算 S_GT。

    Returns:
        loss  : scalar
        s_gt  : (B, K) 供调试/监控用
    """
    s_gt = compute_sgt(pc_fundus, pc_cmr, sigma)
    loss = soft_infonce_loss(z_fundus, z_cmr, s_gt, temperature)
    return loss, s_gt
