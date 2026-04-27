"""
eval_contrast.py
对比学习嵌入质量评估：
  - alignment     : 配对样本平均距离（越小越好）
  - uniformity    : 嵌入在超球面上的均匀性（越小越好，但不能和 alignment 同时崩）
  - retrieval_recall : 跨模态检索 Recall@K（同人命中率）
  - mean_paired_cosine : 同人 fundus·CMR(均值) 余弦的均值（旧名 cross_modal_spearman 保留兼容）
  - gt_pred_sim_spearman : 跨人 (fundus_i · CMR_j) 与 PC 空间高斯 GT 的 Spearman/Pearson
"""

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr, pearsonr


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


def _eid_key(e):
    from contrastive_pretrain.datasets_contrast import CMRBank as _CB
    return _CB._normalize_eid_key(e)


@torch.no_grad()
def retrieval_recall_subsampled(
    fundus_embeddings: torch.Tensor,
    fundus_eids: list,
    cmr_embeddings: torch.Tensor,
    cmr_eids: list,
    pool_size: int,
    seed: int,
    topk: tuple = (1, 5, 10),
) -> dict:
    """
    小候选集检索：对每个 fundus 查询，构造大小为 pool_size 的 CMR 候选池——
    必含「同人」至少一条（随机选一条正样本索引），其余为不同 EID 的负样本随机无放回抽样；
    在该池内按余弦相似度排名，计算 R@k（仅当正样本排名 < k 算命中）。

    pool_size >= 全库 CMR 条数时，退化为在全库上检索（与 retrieval_recall 一致的思想）。
    """
    fundus_embeddings = F.normalize(fundus_embeddings.float(), dim=-1)
    cmr_embeddings = F.normalize(cmr_embeddings.float(), dim=-1)
    n_c = cmr_embeddings.size(0)
    rng = np.random.RandomState(int(seed))

    f_keys = [_eid_key(e) for e in fundus_eids]
    c_keys = [_eid_key(e) for e in cmr_eids]

    pos_by_fkey = {}
    for j, ek in enumerate(c_keys):
        pos_by_fkey.setdefault(ek, []).append(j)

    hits = {k: 0 for k in topk}
    n_ok = 0

    for i in range(len(fundus_eids)):
        fk = f_keys[i]
        if fk not in pos_by_fkey or not pos_by_fkey[fk]:
            continue
        pos_candidates = pos_by_fkey[fk]
        pos_j = int(rng.choice(pos_candidates))
        neg_candidates = [j for j in range(n_c) if c_keys[j] != fk]
        ps = min(int(pool_size), n_c)
        if ps < 2:
            continue
        need_neg = ps - 1
        if len(neg_candidates) < need_neg:
            if len(neg_candidates) == 0:
                continue
            neg_pick = np.array(
                rng.choice(neg_candidates, size=need_neg, replace=True), dtype=np.int64
            )
        else:
            neg_pick = rng.choice(neg_candidates, size=need_neg, replace=False)

        pool_idx = np.concatenate([[pos_j], neg_pick])
        zf = fundus_embeddings[i : i + 1]
        zc = cmr_embeddings[torch.as_tensor(pool_idx, device=fundus_embeddings.device)]
        sims = (zf @ zc.T).squeeze(0)
        order = torch.argsort(sims, descending=True)
        rank = int((order == 0).nonzero(as_tuple=False)[0].item())

        n_ok += 1
        for kk in topk:
            if rank < kk:
                hits[kk] += 1

    if n_ok < 1:
        return {f'R@{k}': float('nan') for k in topk}

    return {f'R@{k}': hits[k] / n_ok for k in topk}


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
        # 确保 EID 统一为 Python 原生类型（避免 LongTensor 的 hash 与 int 不兼容）
        if isinstance(eids, torch.Tensor):
            all_eids.extend(eids.tolist())
        else:
            all_eids.extend([int(e) if hasattr(e, 'item') else e for e in eids])
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

    return torch.cat(all_z, dim=0), [int(e) if hasattr(e, 'item') else e for e in cmr_bank.eids]


@torch.no_grad()
def mean_paired_cosine(
    fundus_embeddings: torch.Tensor,   # (N_f, d)
    fundus_eids: list,
    cmr_embeddings: torch.Tensor,      # (N_c, d)
    cmr_eids: list,
) -> float:
    """
    同人配对：z_fundus[i] 与同 EID 的 CMR 嵌入（多条取均值后再 L2）的余弦相似度，再对样本取平均。
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
        matched_idx = cmr_eid2idx[feid]
        z_cmr_mean = cmr_embeddings[matched_idx].mean(dim=0)
        z_cmr_mean = F.normalize(z_cmr_mean.unsqueeze(0), dim=-1).squeeze(0)
        cos_sim = (fundus_embeddings[i] * z_cmr_mean).sum().item()
        cos_sims.append(cos_sim)

    if len(cos_sims) < 1:
        return float('nan')
    return float(np.mean(cos_sims))


def cross_modal_spearman(*args, **kwargs):
    """Deprecated: 实为 mean_paired_cosine，请改用 mean_paired_cosine。"""
    return mean_paired_cosine(*args, **kwargs)


@torch.no_grad()
def gt_pred_cross_modal_similarity_correlation(
    fundus_embeddings: torch.Tensor,
    fundus_eids: list,
    fundus_pc: torch.Tensor,            # (N_f, P) 与训练一致的 PC
    cmr_embeddings: torch.Tensor,
    cmr_eids: list,
    sigma: float,
    max_pairs: int = 8000,
    seed: int = 0,
) -> dict:
    """
    采样跨人 (e1, e2)，比较：
      - GT：exp(-||pc_e1 - pc_e2||^2 / (2σ^2))，pc 来自各 fundus 行（同 EID 多行取均值）
      - Pred：z_f(e1) · z_c(e2)，fundus 侧 e1 的嵌入 × 人物 e2 的 CMR 嵌入（多条取均值后 L2）

    与 soft-label / 表型几何一致；不要求「检索到同一人」。
    返回 Spearman ρ、Pearson r；样本过少时可能为 nan。
    """
    fundus_embeddings = F.normalize(fundus_embeddings.float(), dim=-1).cpu()
    cmr_embeddings = F.normalize(cmr_embeddings.float(), dim=-1).cpu()
    pc_np = fundus_pc.float().cpu().numpy()

    # EID -> mean z_f, mean pc（键与 CMRBank 一致）
    from collections import defaultdict
    from contrastive_pretrain.datasets_contrast import CMRBank as _CB
    zf_sum = defaultdict(lambda: None)
    pc_sum = defaultdict(lambda: None)
    cnt = defaultdict(int)
    for i, e in enumerate(fundus_eids):
        e = _CB._normalize_eid_key(e)
        z = fundus_embeddings[i]
        if zf_sum[e] is None:
            zf_sum[e] = z.clone()
            pc_sum[e] = pc_np[i].copy()
        else:
            zf_sum[e] = zf_sum[e] + z
            pc_sum[e] = pc_sum[e] + pc_np[i]
        cnt[e] += 1
    eids_f = [e for e in zf_sum if cnt[e] > 0]
    for e in eids_f:
        zf_sum[e] = F.normalize((zf_sum[e] / cnt[e]).unsqueeze(0), dim=-1).squeeze(0)
        pc_sum[e] = pc_sum[e] / cnt[e]

    cmr_eid2idx = defaultdict(list)
    for idx, eid in enumerate(cmr_eids):
        eid = _CB._normalize_eid_key(eid)
        cmr_eid2idx[eid].append(idx)

    zc_mean = {}
    for e, idxs in cmr_eid2idx.items():
        zc_mean[e] = F.normalize(cmr_embeddings[idxs].mean(dim=0).unsqueeze(0), dim=-1).squeeze(0)

    common = [e for e in eids_f if e in zc_mean and e in pc_sum]
    if len(common) < 3:
        return {'gt_pred_spearman': float('nan'), 'gt_pred_pearson': float('nan'), 'n_pairs': 0}

    rng = np.random.RandomState(seed)
    pairs = []
    n_try = 0
    while len(pairs) < max_pairs and n_try < max_pairs * 20:
        n_try += 1
        e1, e2 = rng.choice(common, size=2, replace=False)
        if e1 == e2:
            continue
        pairs.append((int(e1), int(e2)))
    if len(pairs) < 5:
        # 穷举小集合
        pairs = []
        for i, e1 in enumerate(common):
            for e2 in common[i + 1 :]:
                pairs.append((e1, e2))
                if len(pairs) >= max_pairs:
                    break
            if len(pairs) >= max_pairs:
                break

    sig = max(float(sigma), 1e-6)
    gts, preds = [], []
    pc_eids = {e: pc_sum[e] for e in common}
    for e1, e2 in pairs:
        d = pc_eids[e1] - pc_eids[e2]
        gt = float(np.exp(-(d * d).sum() / (2.0 * sig * sig)))
        pr = float((zf_sum[e1] * zc_mean[e2]).sum().item())
        gts.append(gt)
        preds.append(pr)

    gts = np.asarray(gts, dtype=np.float64)
    preds = np.asarray(preds, dtype=np.float64)
    if len(gts) < 3 or np.std(gts) < 1e-12 or np.std(preds) < 1e-12:
        return {
            'gt_pred_spearman': float('nan'),
            'gt_pred_pearson': float('nan'),
            'n_pairs': int(len(gts)),
        }
    rho, _ = spearmanr(gts, preds)
    r, _ = pearsonr(gts, preds)
    return {
        'gt_pred_spearman': float(rho),
        'gt_pred_pearson': float(r),
        'n_pairs': int(len(gts)),
    }


def run_full_eval(fundus_model, cmr_encoder, val_loader, cmr_bank,
                  device='cuda', topk=(1, 5, 10), sigma: float = 6.5893):
    """
    完整评估流程，每 N 个 epoch 调用一次：
      1. 嵌入所有验证集 fundus 样本
      2. 嵌入所有 CMR 样本
      3. 计算 retrieval recall, alignment, uniformity, mean_paired_cosine, GT–Pred 相似度相关

    Returns: dict of metrics
    """
    print('[eval] Embedding all fundus (val)...')
    z_fundus, f_eids, f_pc = embed_all_fundus(fundus_model, val_loader, device)

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
    metrics['paired_cosine'] = mean_paired_cosine(z_fundus, f_eids, z_cmr, c_eids)

    gtcorr = gt_pred_cross_modal_similarity_correlation(
        z_fundus, f_eids, f_pc, z_cmr, c_eids, sigma=sigma)
    metrics.update(gtcorr)

    return metrics
