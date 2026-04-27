"""
train_stage1_mini.py

Stage 1 快速消融实验 — 端到端双塔训练（使用真实图像，全参数微调）。

实验设计（对应 dual_tower_framework.md §12.3）：
  Exp A: CMR 单塔         (cmr_only)
  Exp B: 眼底单塔         (fundus_only)
  Exp D: 双塔 + CrossAttn (paired mode)

数据：
  - CMR:    stage1_cmr_sampled_train.csv   + npy files
  - Fundus: stage1_fundus_sampled_train.csv + PNG files
  - Paired: 两者交集

运行（单机4卡）：
  cd /data/home/shujia/CHD/model_train/RETFound_MAE-main
  torchrun --nproc_per_node=4 contrastive_pretrain/train_stage1_mini.py \
      --exp A --epochs 15 --batch_size 8
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import os
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

HERE  = pathlib.Path(__file__).parent
ROOT  = HERE.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.datasets_dualtower import (
    CMRDataset, FundusDataset, PairedDataset
)
from contrastive_pretrain.models_dualtower import DualTowerModel, build_loss

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]
CMR_CSV    = str(HERE / 'stage1_cmr_sampled_train.csv')
FUNDUS_CSV = str(HERE / 'stage1_fundus_sampled_train.csv')
CMR_VAL    = str(HERE / 'stage1_cmr_sampled_val.csv')
FUNDUS_VAL = str(HERE / 'stage1_fundus_sampled_val.csv')


# ── helpers ──────────────────────────────────────────────────────────────────
def is_dist():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_dist() else 0


def is_main():
    return get_rank() == 0


def log(msg: str):
    if is_main():
        print(msg, flush=True)


def setup_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        dist.init_process_group('nccl')
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        return True
    return False


def collate_fn_cmr(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == 'eid':
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    out['mode'] = 'cmr_only'
    return out


def collate_fn_fundus(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == 'eid':
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    out['mode'] = 'fundus_only'
    return out


def collate_fn_paired(batch):
    keys = batch[0].keys()
    out = {}
    for k in keys:
        if k == 'eid':
            out[k] = [b[k] for b in batch]
        else:
            out[k] = torch.stack([b[k] for b in batch])
    out['mode'] = 'paired'
    return out


def to_device(batch: dict, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# ── AUC evaluation ───────────────────────────────────────────────────────────
@torch.no_grad()
def eval_auc(model_base, loader, device, mode: str, criterion):
    """Evaluate AUC on val set.  Returns dict {cls_name: auc}."""
    model_base.eval()
    all_logits = [[] for _ in range(4)]
    all_labels = [[] for _ in range(4)]
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        batch = to_device(batch, device)
        batch['mode'] = mode
        results = model_base(batch)

        # collect cls predictions
        if mode in ('cmr_only', 'fundus_only'):
            preds = results['cls']
        else:
            preds = results['cmr_cls_base']

        for i, pred in enumerate(preds):
            all_logits[i].append(pred.cpu())
            all_labels[i].append(batch['cls_labels'][:, i].cpu())

        try:
            loss = criterion.compute(results, batch)
            total_loss += loss.item()
            n_batches  += 1
        except Exception:
            pass

    aucs = {}
    for i, name in enumerate(CLS_COLS):
        logits = torch.cat(all_logits[i]).numpy()
        labels = torch.cat(all_labels[i]).numpy()
        valid  = labels >= 0
        if valid.sum() < 2 or labels[valid].max() < 1:
            aucs[name] = float('nan')
            continue
        try:
            aucs[name] = roc_auc_score(labels[valid], logits[valid])
        except Exception:
            aucs[name] = float('nan')

    avg_auc = np.nanmean(list(aucs.values()))
    avg_loss = total_loss / max(n_batches, 1)
    aucs['mean_auc'] = avg_auc
    aucs['val_loss'] = avg_loss
    return aucs


# ── training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, criterion, device, mode):
    model.train()
    total_loss = 0.0
    n = 0
    for batch in loader:
        batch = to_device(batch, device)
        batch['mode'] = mode
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            results = model(batch) if not isinstance(model, DDP) else model(batch)
            loss    = criterion.compute(
                results if not isinstance(model, DDP) else results,
                batch
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def run_experiment(args, exp_name: str, mode: str):
    log(f'\n{"="*60}')
    log(f'Exp {exp_name}  mode={mode}  epochs={args.epochs}  bs={args.batch_size}')
    log(f'{"="*60}')

    distributed = setup_dist()
    local_rank  = int(os.environ.get('LOCAL_RANK', 0))
    device      = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # ── datasets ──────────────────────────────────────────────────────────────
    max_s = args.max_samples

    if mode == 'cmr_only':
        train_ds = CMRDataset(CMR_CSV, max_samples=max_s)
        val_ds   = CMRDataset(CMR_VAL)
        col_fn   = collate_fn_cmr
    elif mode == 'fundus_only':
        train_ds = FundusDataset(FUNDUS_CSV, max_samples=max_s, is_train=True)
        val_ds   = FundusDataset(FUNDUS_VAL, is_train=False)
        col_fn   = collate_fn_fundus
    else:  # paired
        train_ds = PairedDataset(CMR_CSV, FUNDUS_CSV, max_samples=max_s, is_train=True)
        val_ds   = PairedDataset(CMR_VAL, FUNDUS_VAL, is_train=False)
        col_fn   = collate_fn_paired

    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_ds)
    else:
        train_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=4, pin_memory=True, drop_last=True,
        shuffle=(train_sampler is None), collate_fn=col_fn,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, num_workers=2,
        pin_memory=True, collate_fn=col_fn,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = DualTowerModel(
        retfound_ckpt=args.retfound_ckpt,
        medsam_ckpt=args.medsam_ckpt,
        mode=mode,
        grad_ckpt=args.grad_ckpt,
        freeze_fundus_blocks=args.freeze_fundus_blocks,
        freeze_cmr_blocks=args.freeze_cmr_backbone,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = build_loss()

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.05
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )
    scaler = torch.cuda.amp.GradScaler()

    # ── training ──────────────────────────────────────────────────────────────
    best_auc = 0.0
    patience = args.patience
    no_improve = 0
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        t0 = time.time()
        model_raw = model.module if isinstance(model, DDP) else model
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler,
                                     criterion, device, mode)
        scheduler.step()

        # evaluate on val (main process only for metrics, but run on all for DDP consistency)
        val_metrics = eval_auc(model_raw, val_loader, device, mode, criterion)
        mean_auc    = val_metrics['mean_auc']
        val_loss    = val_metrics['val_loss']

        dt = time.time() - t0
        log(f'Epoch {epoch:3d} | train_loss={train_loss:.4f} | '
            f'val_loss={val_loss:.4f} | mean_AUC={mean_auc:.4f} | {dt:.0f}s')
        for name in CLS_COLS:
            log(f'  {name}: AUC={val_metrics[name]:.4f}')

        if is_main() and mean_auc > best_auc:
            best_auc = mean_auc
            no_improve = 0
            ckpt_path = out_dir / f'exp{exp_name}_best.pth'
            torch.save({
                'epoch': epoch,
                'model': model_raw.state_dict(),
                'auc':   best_auc,
                'metrics': val_metrics,
            }, ckpt_path)
            log(f'  [saved] {ckpt_path}  best_auc={best_auc:.4f}')
        else:
            no_improve += 1

        if no_improve >= patience:
            log(f'Early stopping at epoch {epoch} (no improve for {patience} epochs)')
            break

    log(f'\nExp {exp_name} finished. Best AUC = {best_auc:.4f}')
    return best_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='A',
                        choices=['A', 'B', 'D', 'all'],
                        help='A=CMR only, B=Fundus only, D=Paired+CrossAttn')
    parser.add_argument('--epochs',      type=int,   default=15)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--patience',    type=int,   default=5)
    parser.add_argument('--max_samples', type=int,   default=0,
                        help='Limit training samples per dataset (0=all)')
    parser.add_argument('--grad_ckpt', action='store_true',
                        help='Use gradient checkpointing in FundusEncoder (saves memory)')
    parser.add_argument('--freeze_fundus_blocks', type=int, default=20,
                        help='Freeze first N ViT-L blocks (0=train all; 20=fine-tune last 4)')
    parser.add_argument('--freeze_cmr_backbone', action='store_true',
                        help='Freeze entire MedSAM backbone (only train FrameAttnPool+proj+heads)')
    parser.add_argument('--out_dir', type=str,
                        default=str(HERE / 'checkpoints_stage1_mini'))
    parser.add_argument('--retfound_ckpt', type=str,
                        default=str(ROOT / 'RETFound_cfp_weights.pth'))
    parser.add_argument('--medsam_ckpt',   type=str,
                        default=str(ROOT / 'pretrained_weights/hf_cache/'
                                    'models--flaviagiammarino--medsam-vit-base/blobs/'
                                    'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c'))
    args = parser.parse_args()

    exp_map = {
        'A': ('A', 'cmr_only'),
        'B': ('B', 'fundus_only'),
        'D': ('D', 'paired'),
    }

    if args.exp == 'all':
        exps = list(exp_map.values())
    else:
        exps = [exp_map[args.exp]]

    results = {}
    for name, mode in exps:
        auc = run_experiment(args, name, mode)
        results[name] = auc

    if is_main():
        print('\n=== Summary ===')
        for k, v in results.items():
            print(f'  Exp {k}: best_AUC={v:.4f}')


if __name__ == '__main__':
    main()
