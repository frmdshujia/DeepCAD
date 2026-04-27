"""
fast_contrast_train.py
基于预计算特征的极速对比训练：
  - 不加载图像，直接读缓存的 1024-d fundus CLS token
  - 只训练 ProjectionHead (1024→256) 和 CMREncoder (14→256)
  - 每 epoch 秒级完成，可运行 1000+ epoch
  - 完整 retrieval R@k eval

用法：
  python contrastive_pretrain/fast_contrast_train.py \
    --train_feat output_dir/feature_cache/train_feats_full.pt \
    --val_feat output_dir/feature_cache/val_feats_full.pt \
    --cmr_csv contrastive_pretrain/preprocessed_data/cmr_table.csv \
    --output_dir output_dir/exp_fast \
    --epochs 500 --batch_size 128 --lr 1e-3 --temperature 0.07
"""

import sys, os, time, math, json, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn

import collections.abc
if 'torch._six' not in sys.modules:
    class _TorchSix:
        container_abcs = collections.abc
        inf = float('inf')
    sys.modules['torch._six'] = _TorchSix()
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader

from contrastive_pretrain.models_contrast import ProjectionHead, CMREncoder
from contrastive_pretrain.datasets_contrast import CMRBank
from contrastive_pretrain.loss_contrast import (
    hard_infonce_loss,
    soft_infonce_loss_with_sgt,
)
from contrastive_pretrain.eval_contrast import (
    retrieval_recall,
    gt_pred_cross_modal_similarity_correlation,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--train_feat', default='output_dir/feature_cache/train_feats_full.pt')
    p.add_argument('--val_feat',   default='output_dir/feature_cache/val_feats_full.pt')
    p.add_argument('--cmr_csv', default='contrastive_pretrain/preprocessed_data/cmr_table.csv')
    p.add_argument('--pc_cols', default='M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3')
    p.add_argument('--output_dir', default='output_dir/exp_fast')
    p.add_argument('--proj_dim', default=256, type=int)
    p.add_argument('--temperature', default=0.07, type=float)
    p.add_argument('--cmr_sample_k', default=256, type=int)
    p.add_argument('--batch_size', default=128, type=int)
    p.add_argument('--epochs', default=500, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--eval_freq', default=20, type=int)
    p.add_argument('--gpu', default=0, type=int)
    # New: alignment loss weight (0 = pure InfoNCE, >0 adds cosine alignment term)
    p.add_argument('--align_weight', default=0.0, type=float,
                   help='Weight for explicit positive-pair cosine alignment loss')
    # New: symmetric InfoNCE (also compute CMR->fundus direction)
    p.add_argument('--symmetric', action='store_true',
                   help='Use symmetric InfoNCE (both F->C and C->F)')
    # New: projection head without hidden layer (linear probe style)
    p.add_argument('--linear_proj', action='store_true',
                   help='Use linear projection head (no hidden layer)')
    # New: dropout in projection head
    p.add_argument('--proj_dropout', default=0.0, type=float)
    # New: also eval on train set every eval_freq epochs
    p.add_argument('--train_eval', action='store_true',
                   help='Also evaluate paired cosine on training set')
    p.add_argument('--train_eid_frac', default=1.0, type=float,
                   help='仅使用训练集中随机采样的比例 unique EID（0~1，用于学习曲线）')
    p.add_argument('--sigma', default=6.5893, type=float,
                   help='GT 高斯核 σ（soft 训练时用于 S_GT；eval 的 gt_pred 也用同一 σ）')
    p.add_argument('--loss_type', default='hard', choices=['hard', 'soft'],
                   help='hard=标准 InfoNCE；soft=softmax(S_GT/τ_g) 目标，需配合 sgt_temp')
    p.add_argument('--sgt_temp', default=1.0, type=float,
                   help='Soft 目标锐化：target=softmax(S_GT/τ_g)，越小越尖（建议 0.2~1.0）')
    return p.parse_args()


def make_cmr_bank(csv_path, pc_cols, split, device, max_rows=0):
    return CMRBank(csv_path, pc_cols, split=split, device=device, max_rows=max_rows)


@torch.no_grad()
def compute_paired_cosine_train(proj_head, cmr_enc, train_feats, train_pc, device, batch_size=512):
    """计算训练集上的 paired cosine（用 fundus 的配对 PC 直接走 CMR encoder）。
    这是训练集过拟合诊断指标：如果训练高而验证低，说明过拟合。
    """
    proj_head.eval(); cmr_enc.eval()
    sims = []
    N = len(train_feats)
    for s in range(0, N, batch_size):
        f_b = train_feats[s:s+batch_size].to(device)
        c_b = train_pc[s:s+batch_size].to(device)
        z_f = F.normalize(proj_head(f_b), dim=-1)
        z_c = F.normalize(cmr_enc(c_b), dim=-1)
        sims.append((z_f * z_c).sum(dim=-1).cpu())
    return torch.cat(sims).mean().item()


@torch.no_grad()
def run_full_eval_fast(proj_head, cmr_enc, val_feats, val_eids, val_pc, cmr_bank_val, device,
                       sigma: float, topk=(1, 5, 10)):
    proj_head.eval(); cmr_enc.eval()
    # Embed all val fundus
    z_fundus = F.normalize(proj_head(val_feats.to(device).float()), dim=-1).cpu()
    # Embed all val CMR
    pc_all = cmr_bank_val.get_all().to(device)
    z_cmr_list = []
    for s in range(0, cmr_bank_val.n_cmr, 1024):
        z_cmr_list.append(F.normalize(cmr_enc(pc_all[s:s+1024]), dim=-1).cpu())
    z_cmr = torch.cat(z_cmr_list)
    c_eids = [int(e) if hasattr(e, 'item') else e for e in cmr_bank_val.eids]
    # paired cosine
    paired_cos = float('nan')
    eid2idx = {}
    for i, e in enumerate(c_eids):
        eid2idx.setdefault(e, []).append(i)
    sims = []
    for i, fe in enumerate(val_eids):
        if fe in eid2idx:
            matched = z_cmr[eid2idx[fe]].mean(dim=0)
            matched = F.normalize(matched.unsqueeze(0), dim=-1).squeeze()
            sims.append((z_fundus[i] * matched).sum().item())
    if sims:
        paired_cos = float(np.mean(sims))
    recall = retrieval_recall(z_fundus, val_eids, z_cmr, c_eids, topk)
    recall['paired_cosine'] = paired_cos
    recall['n_matched'] = len(sims)
    # alignment (L2 distance between paired embeddings; 2*(1-cos), should decrease)
    pf, pc_matched = [], []
    for i, fe in enumerate(val_eids):
        if fe in eid2idx:
            mc = z_cmr[eid2idx[fe]].mean(dim=0)
            mc = F.normalize(mc.unsqueeze(0), dim=-1).squeeze()
            pf.append(z_fundus[i]); pc_matched.append(mc)
    if pf:
        pf_t = torch.stack(pf); pc_t = torch.stack(pc_matched)
        recall['alignment'] = (pf_t - pc_t).pow(2).sum(dim=-1).mean().item()

    gtc = gt_pred_cross_modal_similarity_correlation(
        z_fundus, val_eids, val_pc, z_cmr, c_eids, sigma=sigma)
    recall.update(gtc)
    return recall


def main():
    args = parse_args()
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    pc_cols = [c.strip() for c in args.pc_cols.split(',')]
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)

    print(f'[fast] Config: loss_type={args.loss_type}, sigma={args.sigma}, sgt_temp={args.sgt_temp}, '
          f'temperature={args.temperature}, align_weight={args.align_weight}, '
          f'symmetric={args.symmetric}, linear_proj={args.linear_proj}, '
          f'proj_dropout={args.proj_dropout}, lr={args.lr}, wd={args.weight_decay}')

    # Load cached features
    print('[fast] Loading cached features...')
    train_cache = torch.load(args.train_feat)
    val_cache   = torch.load(args.val_feat)
    train_feats = train_cache['feats'].float()   # (N, 1024)
    train_eids  = train_cache['eids']
    train_pc    = train_cache['pc'].float()      # (N, 14)
    val_feats   = val_cache['feats'].float()
    val_eids    = [int(e) if hasattr(e, 'item') else e for e in val_cache['eids']]
    val_pc      = val_cache['pc'].float()
    print(f'[fast] Train: {train_feats.shape}, Val: {val_feats.shape}')

    # Build CMR banks
    cmr_bank     = make_cmr_bank(args.cmr_csv, pc_cols, 'train', device, max_rows=0)
    cmr_bank_val = make_cmr_bank(args.cmr_csv, pc_cols, 'val',   device, max_rows=0)

    # Models (no backbone)
    if args.linear_proj:
        # Linear projection: 1024 -> proj_dim directly, minimal overfitting
        import torch.nn as nn
        proj_head = nn.Sequential(
            nn.Linear(1024, args.proj_dim, bias=False)
        ).to(device)
    else:
        proj_head = ProjectionHead(in_dim=1024, hidden_dim=512, out_dim=args.proj_dim,
                                   dropout=args.proj_dropout).to(device)
    cmr_enc = CMREncoder(in_dim=len(pc_cols), hidden_dim=128, out_dim=args.proj_dim).to(device)

    n_params = sum(p.numel() for p in list(proj_head.parameters()) + list(cmr_enc.parameters()))
    print(f'[fast] Total trainable params: {n_params:,}')

    optimizer = torch.optim.AdamW(
        list(proj_head.parameters()) + list(cmr_enc.parameters()),
        lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Build dataset: (feat_idx, eid, pc)
    eid_to_indices = {}
    for idx, e in enumerate(train_eids):
        key = int(e) if hasattr(e, 'item') else e
        eid_to_indices.setdefault(key, []).append(idx)
    unique_eids = list(eid_to_indices.keys())
    if args.train_eid_frac < 1.0:
        rng = np.random.RandomState(42)
        n_keep = max(2, int(len(unique_eids) * args.train_eid_frac))
        unique_eids = list(rng.choice(unique_eids, size=n_keep, replace=False))
        print(f'[fast] train_eid_frac={args.train_eid_frac} → using {len(unique_eids)} unique EIDs')

    best_recall5 = 0.0
    log_rows = []

    for epoch in range(args.epochs):
        proj_head.train(); cmr_enc.train()
        # Sample one image per unique EID, shuffle
        np.random.shuffle(unique_eids)
        chosen_indices = [eid_to_indices[e][np.random.randint(len(eid_to_indices[e]))] for e in unique_eids]
        np.random.shuffle(chosen_indices)

        total_loss = 0.0
        n_batches = 0
        sum_tgt_entropy = 0.0
        n_ent_batches = 0
        for b_start in range(0, len(chosen_indices) - args.batch_size + 1, args.batch_size):
            batch_idx = chosen_indices[b_start: b_start + args.batch_size]
            feats_b = train_feats[batch_idx].to(device)    # (B, 1024)
            pc_fundus_b = train_pc[batch_idx].to(device)   # (B, P) 与 fundus 行一致
            eids_b  = [int(train_eids[i]) if hasattr(train_eids[i], 'item') else train_eids[i]
                       for i in batch_idx]

            # Sample CMR (first B entries are positives matching fundus)
            _, pc_cmr = cmr_bank.sample_with_batch_positives(eids_b, args.cmr_sample_k)
            B = feats_b.size(0)

            # Forward
            z_f = F.normalize(proj_head(feats_b), dim=-1)      # (B, d)
            z_c = F.normalize(cmr_enc(pc_cmr), dim=-1)         # (K, d)

            if args.loss_type == 'soft':
                loss, s_gt = soft_infonce_loss_with_sgt(
                    z_f, z_c, pc_fundus_b, pc_cmr,
                    sigma=args.sigma,
                    temperature=args.temperature,
                    sgt_temp=args.sgt_temp,
                )
                with torch.no_grad():
                    st = max(float(args.sgt_temp), 1e-6)
                    pi = F.softmax(s_gt / st, dim=-1)
                    ent = (-pi * (pi.clamp_min(1e-12).log())).sum(dim=-1).mean()
                    sum_tgt_entropy += ent.item()
                    n_ent_batches += 1
            else:
                loss = hard_infonce_loss(z_f, z_c, temperature=args.temperature)
                # Optional symmetric InfoNCE (仅 hard)
                if args.symmetric:
                    z_f_for_cmr = F.normalize(proj_head(train_feats[batch_idx[:B]].to(device)), dim=-1)
                    logits_cf = z_c[:B] @ z_f_for_cmr.T / args.temperature
                    labels_cf = torch.arange(B, device=device)
                    loss_cf = F.cross_entropy(logits_cf, labels_cf)
                    loss = (loss + loss_cf) / 2

            if args.align_weight > 0:
                pos_cosine = (z_f * z_c[:B]).sum(dim=-1)
                align_loss = (1 - pos_cosine).mean()
                loss = loss + args.align_weight * align_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        avg_tgt_ent = sum_tgt_entropy / max(n_ent_batches, 1) if args.loss_type == 'soft' else None

        # Full eval
        recall_metrics = {}
        if (epoch + 1) % args.eval_freq == 0 or epoch == args.epochs - 1:
            recall_metrics = run_full_eval_fast(
                proj_head, cmr_enc, val_feats, val_eids, val_pc, cmr_bank_val, device,
                sigma=args.sigma)

            # Train set paired cosine (overfitting diagnosis)
            if args.train_eval:
                train_pc_cos = compute_paired_cosine_train(
                    proj_head, cmr_enc, train_feats, train_pc, device)
                recall_metrics['train_paired_cosine'] = train_pc_cos

            r5 = recall_metrics.get('R@5', 0.0)
            if r5 > best_recall5:
                best_recall5 = r5
                torch.save({'proj_head': proj_head.state_dict(), 'cmr_enc': cmr_enc.state_dict()},
                           os.path.join(args.output_dir, 'best.pt'))
            th = f'tgt_H={avg_tgt_ent:.3f} ' if avg_tgt_ent is not None else ''
            print(f'[Epoch {epoch:3d}] loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} '
                  f'{th}eval={recall_metrics}')
        else:
            if epoch % 10 == 0:
                extra = f' tgt_H={avg_tgt_ent:.3f}' if avg_tgt_ent is not None else ''
                print(f'[Epoch {epoch:3d}] loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}{extra}')

        row = {'epoch': epoch, 'loss': avg_loss, **recall_metrics}
        if avg_tgt_ent is not None:
            row['train_target_entropy'] = avg_tgt_ent
        log_rows.append(row)

    # Always save last epoch model
    torch.save({'proj_head': proj_head.state_dict(), 'cmr_enc': cmr_enc.state_dict()},
               os.path.join(args.output_dir, 'last.pt'))

    # Save log
    with open(os.path.join(args.output_dir, 'log.json'), 'w') as f:
        json.dump(log_rows, f, indent=2)
    print(f'\n[fast] Best R@5 = {best_recall5:.4f}')
    print(f'[fast] Training complete.')


if __name__ == '__main__':
    main()
