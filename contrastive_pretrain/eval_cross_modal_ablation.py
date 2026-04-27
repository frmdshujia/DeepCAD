"""
跨模态对齐消融：测试集上对比
  组 A：RETFound 仅初始化眼底 backbone + 随机映射头（冻结）+ 随机初始化 CMR 编码器（冻结）——代表未做对比学习联合训练
  组 B：对比学习 checkpoint_best（fundus_model + cmr_encoder）

指标：gt_pred Spearman/Pearson；全库 R@k；小候选集 R@k（100/1000）各 5 次种子取均值±标准差；
      alignment、uniformity_fundus、uniformity_cmr

用法：
  conda activate retfound
  python contrastive_pretrain/eval_cross_modal_ablation.py \\
    --retfound_ckpt RETFound_cfp_weights.pth \\
    --contrastive_ckpt output_dir/preserved_weights/e2e_run_20260412_174935/checkpoint_best.pth \\
    --split test --gpu 0
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import collections.abc
if 'torch._six' not in sys.modules:
    class _TorchSix:
        container_abcs = collections.abc
        inf = float('inf')
    sys.modules['torch._six'] = _TorchSix()

from contrastive_pretrain.datasets_contrast import FundusContrastDataset, CMRBank
from contrastive_pretrain.models_contrast import FundusContrastModel, CMREncoder
from contrastive_pretrain.eval_contrast import (
    embed_all_fundus,
    embed_all_cmr,
    retrieval_recall,
    retrieval_recall_subsampled,
    alignment,
    uniformity,
    mean_paired_cosine,
    gt_pred_cross_modal_similarity_correlation,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--retfound_ckpt', required=True, type=str)
    p.add_argument('--contrastive_ckpt', required=True, type=str, help='checkpoint_best.pth（含 fundus_model + cmr_encoder）')
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--fundus_csv', default='contrastive_pretrain/preprocessed_data/fundus_table.csv')
    p.add_argument('--cmr_csv', default='contrastive_pretrain/preprocessed_data/cmr_table.csv')
    p.add_argument('--pc_cols', default='M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3')
    p.add_argument('--sigma', type=float, default=None)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--n_repeat_subsample', type=int, default=5, help='小候选集 R@k 重复抽样次数')
    p.add_argument('--subsample_seed_base', type=int, default=42)
    p.add_argument('--pool_sizes', default='100,1000', type=str, help='逗号分隔；另自动跑全库')
    p.add_argument('--output_json', default='', type=str)
    return p.parse_args()


def _collect_metrics(
    name: str,
    fundus_model: FundusContrastModel,
    cmr_enc: CMREncoder,
    loader,
    cmr_bank: CMRBank,
    device: str,
    sigma: float,
    n_repeat: int,
    seed_base: int,
    pool_sizes: list,
) -> dict:
    print(f'\n========== {name} ==========')
    z_f, f_eids, f_pc = embed_all_fundus(fundus_model, loader, device)
    z_c, c_eids = embed_all_cmr(cmr_enc, cmr_bank, device=str(device))
    print(f'  embedded fundus {z_f.shape[0]}, cmr {z_c.shape[0]}')

    out = {}

    gtc = gt_pred_cross_modal_similarity_correlation(
        z_f, f_eids, f_pc, z_c, c_eids, sigma=sigma, max_pairs=8000, seed=0,
    )
    out['gt_pred_spearman'] = gtc.get('gt_pred_spearman', float('nan'))
    out['gt_pred_pearson'] = gtc.get('gt_pred_pearson', float('nan'))
    out['gt_pred_n_pairs'] = gtc.get('n_pairs', 0)

    # 全库检索
    recall_full = retrieval_recall(z_f, f_eids, z_c, c_eids, topk=(1, 5, 10))
    out['retrieval_full_gallery'] = {k: float(recall_full[k]) for k in recall_full}

    # 小候选集
    out['retrieval_subsampled'] = {}
    for ps in pool_sizes:
        if ps >= z_c.size(0):
            m = retrieval_recall(z_f, f_eids, z_c, c_eids, topk=(1, 5, 10))
            out['retrieval_subsampled'][f'N={ps}'] = {
                'note': 'pool >= gallery size, 使用全库',
                **{k: float(m[k]) for k in m},
            }
            continue
        r1, r5, r10 = [], [], []
        for r in range(n_repeat):
            sd = seed_base + r * 1000 + ps
            m = retrieval_recall_subsampled(
                z_f, f_eids, z_c, c_eids, pool_size=ps, seed=sd, topk=(1, 5, 10),
            )
            r1.append(m['R@1'])
            r5.append(m['R@5'])
            r10.append(m['R@10'])
        out['retrieval_subsampled'][f'N={ps}'] = {
            'R@1_mean': float(np.mean(r1)),
            'R@1_std': float(np.std(r1)),
            'R@5_mean': float(np.mean(r5)),
            'R@5_std': float(np.std(r5)),
            'R@10_mean': float(np.mean(r10)),
            'R@10_std': float(np.std(r10)),
            'repeats': n_repeat,
        }

    # alignment / uniformity / paired cosine
    cmr_eid2idx = {}
    for idx, eid in enumerate(c_eids):
        cmr_eid2idx.setdefault(eid, []).append(idx)

    paired_f, paired_c = [], []
    for i, feid in enumerate(f_eids):
        if feid in cmr_eid2idx:
            matched = cmr_eid2idx[feid]
            paired_f.append(z_f[i])
            paired_c.append(z_c[matched].mean(dim=0))

    z_fn = F.normalize(z_f, dim=-1)
    z_cn = F.normalize(z_c, dim=-1)
    out['uniformity_fundus'] = uniformity(z_fn)
    out['uniformity_cmr'] = uniformity(z_cn)
    out['paired_cosine'] = mean_paired_cosine(z_f, f_eids, z_c, c_eids)

    if paired_f:
        pf = F.normalize(torch.stack(paired_f), dim=-1)
        pc = F.normalize(torch.stack(paired_c), dim=-1)
        out['alignment_paired_l2_sq'] = alignment(pf, pc)
    else:
        out['alignment_paired_l2_sq'] = float('nan')

    return out


def main():
    args = parse_args()
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    ckpt_b = torch.load(args.contrastive_ckpt, map_location='cpu')
    ca = ckpt_b.get('args') or {}
    sigma = float(args.sigma if args.sigma is not None else ca.get('sigma', 6.5893))
    proj_dim = int(ca.get('proj_dim', 256))
    drop_path = float(ca.get('drop_path', 0.1))
    n_pc = int(ca.get('n_pc', 14))

    pc_cols = [c.strip() for c in args.pc_cols.split(',')]
    if len(pc_cols) != n_pc:
        n_pc = len(pc_cols)

    fundus_csv = args.fundus_csv if os.path.isabs(args.fundus_csv) else os.path.join(ROOT, args.fundus_csv)
    cmr_csv = args.cmr_csv if os.path.isabs(args.cmr_csv) else os.path.join(ROOT, args.cmr_csv)

    dataset = FundusContrastDataset(fundus_csv, pc_cols, split=args.split)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    cmr_bank = CMRBank(cmr_csv, pc_cols, split=args.split, device=str(device))

    pool_sizes = [int(x.strip()) for x in args.pool_sizes.split(',') if x.strip()]

    report = {
        'split': args.split,
        'n_fundus': len(dataset),
        'n_cmr_bank': cmr_bank.n_cmr,
        'sigma': sigma,
        'group_a_description': 'RETFound backbone + random proj head (frozen) + random CMR encoder (frozen)',
        'group_b_description': args.contrastive_ckpt,
    }

    # —— 组 A —— #
    torch.manual_seed(0)
    np.random.seed(0)
    model_a = FundusContrastModel(proj_dim=proj_dim, drop_path_rate=drop_path)
    retfound_path = args.retfound_ckpt if os.path.isabs(args.retfound_ckpt) else os.path.join(ROOT, args.retfound_ckpt)
    model_a.load_pretrained(retfound_path)
    model_a.to(device)
    cmr_a = CMREncoder(in_dim=n_pc, hidden_dim=128, out_dim=proj_dim).to(device)
    for p in list(model_a.parameters()) + list(cmr_a.parameters()):
        p.requires_grad_(False)

    report['group_a_retfound_random_cmr'] = _collect_metrics(
        'A_RETFound_random_CMR',
        model_a, cmr_a, loader, cmr_bank, str(device), sigma,
        args.n_repeat_subsample, args.subsample_seed_base, pool_sizes,
    )

    del model_a, cmr_a
    torch.cuda.empty_cache()

    # —— 组 B —— #
    model_b = FundusContrastModel(proj_dim=proj_dim, drop_path_rate=drop_path)
    cmr_b = CMREncoder(in_dim=n_pc, hidden_dim=128, out_dim=proj_dim)
    model_b.load_state_dict(ckpt_b['fundus_model'])
    cmr_b.load_state_dict(ckpt_b['cmr_encoder'])
    model_b.to(device)
    cmr_b.to(device)
    for p in list(model_b.parameters()) + list(cmr_b.parameters()):
        p.requires_grad_(False)

    report['group_b_contrastive_best'] = _collect_metrics(
        'B_contrastive_checkpoint',
        model_b, cmr_b, loader, cmr_bank, str(device), sigma,
        args.n_repeat_subsample, args.subsample_seed_base, pool_sizes,
    )

    print('\n================ SUMMARY ================')
    for label, key in [('A 基线', 'group_a_retfound_random_cmr'), ('B 对比学习', 'group_b_contrastive_best')]:
        m = report[key]
        print(f"\n[{label}] gt_pred  Spearman={m['gt_pred_spearman']:.4f}  Pearson={m['gt_pred_pearson']:.4f}")
        print(f"  alignment(L2^2)={m['alignment_paired_l2_sq']:.6f}  paired_cosine={m['paired_cosine']:.4f}")
        print(f"  uniformity_fundus={m['uniformity_fundus']:.4f}  uniformity_cmr={m['uniformity_cmr']:.4f}")
        print(f"  R@k full: {m['retrieval_full_gallery']}")
        for nk, nv in m.get('retrieval_subsampled', {}).items():
            print(f"  {nk}: {nv}")

    out_path = args.output_json
    if not out_path:
        out_path = os.path.join(
            ROOT, 'output_dir', f'cross_modal_ablation_{args.split}.json',
        )
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    def _json_safe(obj):
        if isinstance(obj, dict):
            return {k: _json_safe(v) for k, v in obj.items()}
        if isinstance(obj, float):
            return obj if np.isfinite(obj) else None
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj) if isinstance(obj, np.floating) else int(obj)
        return obj

    with open(out_path, 'w') as f:
        json.dump(_json_safe(report), f, indent=2)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
