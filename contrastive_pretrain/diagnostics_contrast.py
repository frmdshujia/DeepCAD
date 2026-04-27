#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
diagnostics_contrast.py — 对比预训练逐项排查（实现 / 数据 / 目标尺度 / 优化）

建议与烟雾测相同的数据与超参，例如：
  conda run -n retfound --no-capture-output python contrastive_pretrain/diagnostics_contrast.py \\
    --fundus_csv contrastive_pretrain/preprocessed_data/fundus_table.csv \\
    --cmr_csv contrastive_pretrain/preprocessed_data/cmr_table.csv \\
    --pc_cols "M1_PC1,..." --sigma 6.5893 --finetune RETFound_cfp_weights.pth

检查项：
  eid      — Fundus(train) 的 EID 是否在 CMR 全表配对字典中
  sgt      — 单 batch 的 S_GT：对角线 vs 非对角、softmax 熵（目标是否过平）
  sigma    — σ 缩放对对角/非对角对比度的影响（σ 是否离谱）
  grad     — 单步反传梯度范数（是否全零、是否极端小）
  overfit  — 固定单 batch 重复优化若干步，loss 能否明显下降（管道可学性）
  all      — 以上全部
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import collections.abc
if 'torch._six' not in sys.modules:
    class _TorchSix:
        container_abcs = collections.abc
        inf = float('inf')
    sys.modules['torch._six'] = _TorchSix()

from contrastive_pretrain.datasets_contrast import FundusContrastDataset, CMRBank
from contrastive_pretrain.loss_contrast import compute_sgt, soft_infonce_loss_with_sgt
from contrastive_pretrain.models_contrast import FundusContrastModel, CMREncoder


def check_eid(args, cmr_bank: CMRBank):
    """因素 A：EID 对齐 / 数据覆盖。"""
    print('\n========== [A] EID 覆盖（Fundus train ↔ CMR 全表配对）==========')
    df = pd.read_csv(args.fundus_csv)
    df = df[df['split'] == 'train'].reset_index(drop=True)
    if len(df) == 0:
        print('  错误: fundus CSV 中无 split=train 行')
        return
    keys = [CMRBank._normalize_eid_key(e) for e in df['eid'].tolist()]
    uniq = list(dict.fromkeys(keys))
    paired = cmr_bank._eid_to_paired_pc
    missing = [k for k in uniq if k not in paired]
    print(f'  Fundus train 行数: {len(df)}，唯一 EID: {len(uniq)}')
    print(f'  CMR 配对表 unique EID: {len(paired)}')
    print(f'  Fundus 中无法在 CMR 配对表查到的 EID 数: {len(missing)}')
    if missing[:5]:
        print(f'  示例缺失 EID（最多 5 个）: {missing[:5]}')
    ratio = 1.0 - (len(missing) / max(len(uniq), 1))
    print(f'  覆盖率: {ratio:.4f}')
    if ratio < 0.999:
        print('  → 判定: 存在大量 EID 缺失时，训练信号与评估会系统性错位（数据/拆分问题）。')
    else:
        print('  → 判定: EID 覆盖良好。')


def check_sgt(args, cmr_bank: CMRBank, pc_cols, device):
    """因素 B：S_GT 形状（σ、是否过平）。"""
    print('\n========== [B] S_GT 统计（单 batch，含正样本在前 B 维）==========')
    ds = FundusContrastDataset(
        args.fundus_csv, pc_cols, split='train',
        train_subset_ratio=1.0,
        train_max_samples=max(args.diag_batch_size * 4, 64),
        subset_seed=args.subset_seed,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.diag_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    images, eids, pc_fundus = next(iter(loader))
    eids_list = eids.detach().cpu().tolist() if isinstance(eids, torch.Tensor) else list(eids)
    B = len(eids_list)
    _, pc_cmr = cmr_bank.sample_with_batch_positives(eids_list, args.cmr_sample_k)
    pc_f = pc_fundus.to(device)
    pc_c = pc_cmr.to(device)

    s_gt = compute_sgt(pc_f, pc_c, args.sigma)
    # 对角线：第 i 个 fundus 与第 i 个 CMR 槽（正样本）
    diag = torch.diagonal(s_gt).detach().cpu().numpy()
    # 非对角：每行去掉第 i 列
    mask = torch.ones_like(s_gt, dtype=torch.bool)
    mask[torch.arange(B, device=device), torch.arange(B, device=device)] = False
    off = s_gt[mask].view(B, -1).mean(dim=1).detach().cpu().numpy()

    tgt = F.softmax(s_gt, dim=-1)
    ent = (-tgt * (tgt.clamp_min(1e-12).log())).sum(dim=-1).detach().cpu().numpy()
    k = args.cmr_sample_k
    logk = float(np.log(k))

    print(f'  batch_size B={B}, K={args.cmr_sample_k}, σ={args.sigma}')
    print(f'  S_GT 对角（正样本槽）: mean={diag.mean():.6f}, min={diag.min():.6f}, max={diag.max():.6f}')
    print(f'  每行非对角均值（粗看「背景」）: mean={off.mean():.6f}')
    print(f'  对角 − 非对角（越大越好）: mean={(diag - off).mean():.6f}')
    print(f'  softmax(S_GT) 行熵: mean={ent.mean():.4f}（均匀分布约 log(K)={logk:.4f}）')
    if ent.mean() > logk - 0.15:
        print('  → 判定: 目标分布接近均匀，soft 标签很「平」，loss 会贴近 log(K)、下降慢。')
    else:
        print('  → 判定: 目标有一定尖峰，可学信号相对集中。')


def check_sigma(args, cmr_bank: CMRBank, pc_cols, device):
    """因素 B2：σ 敏感度。"""
    print('\n========== [B2] σ 敏感度（同 batch，σ × {0.5, 1, 2}）==========')
    ds = FundusContrastDataset(
        args.fundus_csv, pc_cols, split='train',
        train_max_samples=max(args.diag_batch_size * 4, 64),
        subset_seed=args.subset_seed,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.diag_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    images, eids, pc_fundus = next(iter(loader))
    eids_list = eids.detach().cpu().tolist() if isinstance(eids, torch.Tensor) else list(eids)
    B = len(eids_list)
    _, pc_cmr = cmr_bank.sample_with_batch_positives(eids_list, args.cmr_sample_k)
    pc_f = pc_fundus.to(device)
    pc_c = pc_cmr.to(device)

    for fac in (0.5, 1.0, 2.0):
        sig = args.sigma * fac
        s_gt = compute_sgt(pc_f, pc_c, sig)
        diag = torch.diagonal(s_gt).mean().item()
        mask = torch.ones_like(s_gt, dtype=torch.bool)
        mask[torch.arange(B, device=device), torch.arange(B, device=device)] = False
        off = s_gt[mask].view(B, -1).mean().item()
        print(f'  σ={sig:.6f} (×{fac}): diag_mean={diag:.6f}, off_mean={off:.6f}, diag−off={diag-off:.6f}')
    print('  → 若随 σ 变化对角优势消失很快，说明 PC 距离相对 σ 的尺度需复查。')


def check_grad(args, cmr_bank: CMRBank, pc_cols, device):
    """因素 C：梯度是否非零、量级是否合理。"""
    print('\n========== [C] 单步梯度范数（混合精度与主训练一致）==========')
    ds = FundusContrastDataset(
        args.fundus_csv, pc_cols, split='train',
        train_max_samples=max(args.diag_batch_size * 4, 64),
        subset_seed=args.subset_seed,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.diag_batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    images, eids, pc_fundus = next(iter(loader))
    eids_list = eids.detach().cpu().tolist() if isinstance(eids, torch.Tensor) else list(eids)

    fundus_model = FundusContrastModel(proj_dim=args.proj_dim, drop_path_rate=args.drop_path)
    fundus_model.load_pretrained(args.finetune)
    fundus_model.to(device)
    cmr_enc = CMREncoder(in_dim=len(pc_cols), hidden_dim=128, out_dim=args.proj_dim)
    cmr_enc.to(device)
    fundus_model.train()
    cmr_enc.train()

    opt = torch.optim.AdamW(
        list(fundus_model.parameters()) + list(cmr_enc.parameters()),
        lr=1e-4,
        weight_decay=0.05,
    )

    images = images.cuda(non_blocking=True)
    pc_f = pc_fundus.cuda(non_blocking=True)
    _, pc_cmr = cmr_bank.sample_with_batch_positives(eids_list, args.cmr_sample_k)

    opt.zero_grad()
    with torch.cuda.amp.autocast():
        z_f = fundus_model(images)
        z_c = cmr_enc(pc_cmr)
        loss, _ = soft_infonce_loss_with_sgt(
            z_f, z_c, pc_f, pc_cmr, sigma=args.sigma, temperature=args.temperature,
        )
    loss.backward()

    def _gn(m):
        s = 0.0
        n = 0
        for p in m.parameters():
            if p.grad is not None:
                s += p.grad.data.norm(2).item() ** 2
                n += 1
        return (s ** 0.5) if n else 0.0, n

    g_f, n_f = _gn(fundus_model)
    g_c, n_c = _gn(cmr_enc)
    print(f'  loss={loss.item():.6f}')
    print(f'  FundusContrastModel grad L2 范数: {g_f:.6e}（有梯度的参数张量数相关）')
    print(f'  CMREncoder grad L2 范数: {g_c:.6e}')
    if g_f < 1e-12 and g_c < 1e-12:
        print('  → 判定: 梯度几乎为零，检查 loss、冻结项、AMP。')
    elif g_c > g_f * 100:
        print('  → 提示: CMR 塔梯度常远大于 ViT（参数少、lr 分组也常更大），属常见现象。')
    else:
        print('  → 判定: 反传非零，优化通路基本畅通。')


def check_overfit(args, cmr_bank: CMRBank, pc_cols, device):
    """因素 D：单 batch 记忆（实现与可学性）。"""
    print('\n========== [D] 单 batch 过拟合（重复同一步，看 loss 能否明显下降）==========')
    ds = FundusContrastDataset(
        args.fundus_csv, pc_cols, split='train',
        train_max_samples=max(args.diag_batch_size * 4, 64),
        subset_seed=args.subset_seed,
    )
    loader = torch.utils.data.DataLoader(
        ds, batch_size=args.diag_batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    images, eids, pc_fundus = next(iter(loader))
    eids_list = eids.detach().cpu().tolist() if isinstance(eids, torch.Tensor) else list(eids)

    fundus_model = FundusContrastModel(proj_dim=args.proj_dim, drop_path_rate=args.drop_path)
    fundus_model.load_pretrained(args.finetune)
    fundus_model.to(device)
    cmr_enc = CMREncoder(in_dim=len(pc_cols), hidden_dim=128, out_dim=args.proj_dim)
    cmr_enc.to(device)
    fundus_model.train()
    cmr_enc.train()

    # 诊断用略大学习率，便于在数百步内看到趋势（不代表正式训练最优）
    opt = torch.optim.AdamW(
        [
            {'params': fundus_model.parameters(), 'lr': args.overfit_lr_backbone},
            {'params': cmr_enc.parameters(), 'lr': args.overfit_lr_cmr},
        ],
        weight_decay=0.05,
    )

    images = images.cuda(non_blocking=True)
    pc_f = pc_fundus.cuda(non_blocking=True)

    losses = []
    scaler = torch.cuda.amp.GradScaler()

    for step in range(args.overfit_steps):
        opt.zero_grad()
        _, pc_cmr = cmr_bank.sample_with_batch_positives(eids_list, args.cmr_sample_k)
        with torch.cuda.amp.autocast():
            z_f = fundus_model(images)
            z_c = cmr_enc(pc_cmr)
            loss, _ = soft_infonce_loss_with_sgt(
                z_f, z_c, pc_f, pc_cmr, sigma=args.sigma, temperature=args.temperature,
            )
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        losses.append(loss.item())
        if step == 0 or step == args.overfit_steps - 1 or (step + 1) % max(1, args.overfit_steps // 5) == 0:
            print(f'  step {step:4d}  loss={loss.item():.6f}')

    lo, hi = min(losses), max(losses)
    drop = losses[0] - losses[-1]
    logk = float(np.log(args.cmr_sample_k))
    print(f'  首 loss={losses[0]:.6f}, 末 loss={losses[-1]:.6f}, 下降 Δ={drop:.6f}')
    print(f'  参考: log(K)≈{logk:.4f}（均匀目标近似下界）')
    if drop < 0.05:
        print('  → 判定: 几乎不降：优先怀疑目标过平/σ/K、或 LR 仍过小、或需检查实现。')
    elif losses[-1] < logk - 0.3:
        print('  → 判定: 明显下降且低于 log(K) 参考，管道「可学」，慢更可能来自正式训练的 LR/数据规模。')
    else:
        print('  → 判定: 有下降但仍贴近 log(K) 区间，与 [B] 目标熵、「任务难」一致，可并列考虑。')


def main():
    p = argparse.ArgumentParser('Contrastive diagnostics')
    p.add_argument('--fundus_csv', required=True)
    p.add_argument('--cmr_csv', required=True)
    p.add_argument('--pc_cols', required=True)
    p.add_argument('--sigma', type=float, required=True)
    p.add_argument('--finetune', required=True)
    p.add_argument('--cmr_train_max_rows', type=int, default=12000)
    p.add_argument('--subset_seed', type=int, default=42)
    p.add_argument('--temperature', type=float, default=0.07)
    p.add_argument('--cmr_sample_k', type=int, default=1024)
    p.add_argument('--proj_dim', type=int, default=256)
    p.add_argument('--drop_path', type=float, default=0.1)
    p.add_argument('--diag_batch_size', type=int, default=16)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--overfit_steps', type=int, default=300)
    p.add_argument('--overfit_lr_backbone', type=float, default=3e-5)
    p.add_argument('--overfit_lr_cmr', type=float, default=3e-4)
    p.add_argument('--checks', nargs='+', default=['all'],
                   choices=['eid', 'sgt', 'sigma', 'grad', 'overfit', 'all'])
    args = p.parse_args()

    os.environ.setdefault('CUDA_VISIBLE_DEVICES', os.environ.get('CUDA_VISIBLE_DEVICES', '0'))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print('警告: 无 CUDA，[grad]/[overfit] 会很慢或 OOM。')

    pc_cols = [c.strip() for c in args.pc_cols.split(',')]

    cmr_bank = CMRBank(
        args.cmr_csv, pc_cols, split='train', device=str(device),
        max_rows=args.cmr_train_max_rows, subset_seed=args.subset_seed,
    )

    want = set(args.checks)
    if 'all' in want:
        want = {'eid', 'sgt', 'sigma', 'grad', 'overfit'}

    if 'eid' in want:
        check_eid(args, cmr_bank)
    if 'sgt' in want:
        check_sgt(args, cmr_bank, pc_cols, device)
    if 'sigma' in want:
        check_sigma(args, cmr_bank, pc_cols, device)
    if 'grad' in want:
        check_grad(args, cmr_bank, pc_cols, device)
    if 'overfit' in want:
        check_overfit(args, cmr_bank, pc_cols, device)

    print('\n========== 汇总阅读顺序建议 ==========')
    print('  1) EID 覆盖差 → 先修表/拆分，再谈模型。')
    print('  2) S_GT 熵≈log(K) → 目标过平：σ、K、或任务本身信息弱。')
    print('  3) 梯度≈0 → 实现/冻结/数值；梯度正常但 loss 不降 → 看 2) 与 LR。')
    print('  4) 单 batch 过拟合仍不降 → 实现或目标定义；能降 → 正式训练慢多半来自 LR/数据量/K。')
    print('完成。\n')


if __name__ == '__main__':
    main()
