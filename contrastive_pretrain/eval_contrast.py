"""
eval_contrast.py
对比学习嵌入质量评估：
  - alignment     : 配对样本平均距离（越小越好）
  - uniformity    : 嵌入在超球面上的均匀性（越小越好，但不能和 alignment 同时崩）
  - retrieval_recall : 跨模态检索 Recall@K（同人命中率）
  - cross_modal_spearman : fundus embedding 与 CMR PC score 的 Spearman ρ
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr


@torch.no_grad()
def alignment(z_fundus: torch.Tensor, z_cmr_matched: torch.Tensor) -> float:
    """
    Wang & Isola (ICML 2020) alignment 指标：
      alignment = E[ ‖z_fundus_i − z_cmr_i‖² ]
    要求输入为 L2 归一化后的配对嵌入（同一受试者的眼底 & CMR 嵌入）。

    Args:
        z_fundus     : (N, d) L2-normalized
        z_cmr_matched: (N, d) L2-normalized，与 z_fundus 一一对应
    """
    return (z_fundus - z_cmr_matched).norm(dim=1).pow(2).mean().item()


@torch.no_grad()
def uniformity(z: torch.Tensor, t: float = 2.0) -> float:
    """
    Wang & Isola (ICML 2020) uniformity 指标：
      uniformity = log( E[ exp(-t × ‖z_i − z_j‖²) ] )
    值越小（越负）表示嵌入越均匀地分布在超球面上。
    若 uniformity 趋近 0 → 嵌入坍缩（collapse）。

    Args:
        z : (N, d) L2-normalized embeddings
        t : kernel 参数（默认 2.0）
    """
    sq_dists = torch.cdist(z, z, p=2).pow(2)           # (N, N)
    return sq_dists.mul(-t).exp().mean().log().item()


@torch.no_grad()
def retrieval_recall(
    fundus_embeddings: torch.Tensor,   # (N_f, d)  L2-normalized
    fundus_eids: list,
    cmr_embeddings: torch.Tensor,      # (N_c, d)  L2-normalized
    cmr_eids: list,
    topk: tuple = (1, 5, 10),
) -> dict:
    """
    跨模态检索 Recall@K：
    对每一个 fundus embedding，在全量 CMR embedding 空间中检索余弦相似度最高的 K 个，
    统计命中（retrieved CMR 中有同 EID 的）比例。

    验证/测试阶段：同一受试者可能有多张眼底图，结果取所有眼底图的平均。

    Returns:
        dict: {'R@1': float, 'R@5': float, 'R@10': float, ...}
    """
    fundus_embeddings = F.normalize(fundus_embeddings, dim=-1)
    cmr_embeddings = F.normalize(cmr_embeddings, dim=-1)

    # 相似度矩阵 (N_f, N_c)
    sim = fundus_embeddings @ cmr_embeddings.T

    cmr_eids_arr = np.array(cmr_eids)
    results = {}

    for k in topk:
        hits = 0
        top_k_indices = sim.topk(k, dim=1).indices.cpu().numpy()  # (N_f, k)
        for i, row_idx in enumerate(top_k_indices):
            target_eid = fundus_eids[i]
            retrieved_eids = cmr_eids_arr[row_idx]
            if target_eid in retrieved_eids:
                hits += 1
        results[f'R@{k}'] = hits / len(fundus_eids)

    return results


@torch.no_grad()
def embed_all_fundus(model, loader, device: str = 'cuda'):
    """
    遍历 fundus DataLoader，收集所有样本的嵌入、EID、PC score。

    用于：
      - 计算验证集 retrieval recall
      - 计算 alignment / uniformity

    Returns:
        embeddings : (N, d) FloatTensor
        eids       : list of EID
        pc_vecs    : (N, n_pc) FloatTensor
    """
    model.eval()
    all_z, all_eids, all_pc = [], [], []

    for images, eids, pc_vecs in loader:
        images = images.to(device, non_blocking=True)
        z = model(images)                              # (B, d)
        all_z.append(z.cpu())
        all_eids.extend(list(eids))
        all_pc.append(pc_vecs)

    return (
        torch.cat(all_z, dim=0),
        all_eids,
        torch.cat(all_pc, dim=0),
    )


@torch.no_grad()
def embed_all_cmr(cmr_encoder, cmr_bank, device: str = 'cuda',
                  batch_size: int = 2048):
    """
    对全量 CMR PC scores 分批次过 CMR MLP，返回嵌入矩阵。

    Returns:
        embeddings : (N_cmr, d) FloatTensor（CPU）
        eids       : list of EID
    """
    cmr_encoder.eval()
    all_z = []
    pc_all = cmr_bank.get_all()      # (N_cmr, n_pc) on device

    for start in range(0, cmr_bank.n_cmr, batch_size):
        pc_batch = pc_all[start: start + batch_size]
        z = cmr_encoder(pc_batch)
        all_z.append(z.cpu())

    return torch.cat(all_z, dim=0), cmr_bank.eids


@torch.no_grad()
def cross_modal_spearman(
    fundus_embeddings: torch.Tensor,   # (N_f, d)
    fundus_eids: list,
    cmr_embeddings: torch.Tensor,      # (N_c, d)
    cmr_eids: list,
) -> float:
    """
    计算 fundus embedding 空间与 CMR embedding 空间的跨模态对齐程度。
    方法：对每个 fundus 样本，取其在 CMR 空间的最近邻余弦相似度，
    与该 fundus 对应的 CMR PC 向量之间的直接余弦相似度对比，
    计算 Spearman ρ。

    更简化的实现：对有配对的 fundus-CMR（同 EID），
    计算 z_fundus ⋅ z_cmr_matched 的分布 Spearman ρ with 1.0（完美对齐期望）
    实际作为 early stopping 指标时用 retrieval_recall 更直观，此函数仅作补充。
    """
    fundus_embeddings = F.normalize(fundus_embeddings, dim=-1)
    cmr_embeddings = F.normalize(cmr_embeddings, dim=-1)

    cmr_eid2idx = {}
    for idx, eid in enumerate(cmr_eids):
        cmr_eid2idx.setdefault(eid, []).append(idx)

    cos_sims = []
    for i, feid in enumerate(fundus_eids):
        if feid not in cmr_eid2idx:
            continue
        # 同人配对：取所有 CMR 嵌入的均值
        matched_idx = cmr_eid2idx[feid]
        z_cmr_mean = cmr_embeddings[matched_idx].mean(dim=0)
        z_cmr_mean = F.normalize(z_cmr_mean.unsqueeze(0), dim=-1).squeeze(0)
        cos_sim = (fundus_embeddings[i] * z_cmr_mean).sum().item()
        cos_sims.append(cos_sim)

    if len(cos_sims) < 2:
        return float('nan')

    # cos_sims 与完美分数 1.0 的 Spearman ρ 等价于 cos_sims 内部的秩次稳定性
    # 实际意义：配对相似度越高越好；这里直接返回均值作为近似指标
    return float(np.mean(cos_sims))


def run_full_eval(fundus_model, cmr_encoder, val_loader, cmr_bank,
                  device='cuda', topk=(1, 5, 10)):
    """
    完整评估流程，每 N 个 epoch 调用一次：
      1. 嵌入所有验证集 fundus 样本
      2. 嵌入所有 CMR 样本
      3. 计算 retrieval recall, alignment, uniformity, spearman

    Returns: dict of metrics
    """
    print('[eval] Embedding all fundus (val)...')
    z_fundus, f_eids, _ = embed_all_fundus(fundus_model, val_loader, device)

    print('[eval] Embedding all CMR...')
    z_cmr, c_eids = embed_all_cmr(cmr_encoder, cmr_bank, device)

    print('[eval] Computing retrieval recall...')
    recall = retrieval_recall(z_fundus, f_eids, z_cmr, c_eids, topk)

    print('[eval] Computing alignment & uniformity...')
    # alignment 需要配对嵌入；仅对有配对 CMR 的 fundus 计算
    cmr_eid2idx = {}
    for idx, eid in enumerate(c_eids):
        cmr_eid2idx.setdefault(eid, []).append(idx)

    paired_f, paired_c = [], []
    for i, feid in enumerate(f_eids):
        if feid in cmr_eid2idx:
            matched = cmr_eid2idx[feid]
            paired_f.append(z_fundus[i])
            paired_c.append(z_cmr[matched].mean(dim=0))

    metrics = dict(recall)
    if paired_f:
        pf = F.normalize(torch.stack(paired_f), dim=-1)
        pc = F.normalize(torch.stack(paired_c), dim=-1)
        metrics['alignment'] = alignment(pf, pc)

    metrics['uniformity_fundus'] = uniformity(F.normalize(z_fundus, dim=-1))
    metrics['uniformity_cmr'] = uniformity(F.normalize(z_cmr, dim=-1))
    metrics['paired_cosine'] = cross_modal_spearman(z_fundus, f_eids, z_cmr, c_eids)

    return metrics
