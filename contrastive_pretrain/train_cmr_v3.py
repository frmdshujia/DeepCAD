"""
train_cmr_v3.py  —  CMR v3 temporal model (single GPU)

Architecture: CMREncoderV3
  - Shared MedSAM ViT-B backbone (13 frames × GAP → frame tokens)
  - 2D positional encoding (view_id + time_id)
  - 2-layer temporal Transformer + CLS token
  - TaskHead: 2 cls (ischemic_hd, I21) + 3 reg (LVEF, LVEDV, LVESV)

Input data: (13, 224, 224) float16 npy from preprocess_cmr_v3.py

Usage:
  # Preprocess first (if not done):
  python contrastive_pretrain/preprocess_cmr_v3.py \\
      --eid_file contrastive_pretrain/task_reports/task1_cmr_train.csv \\
      --cmr_dir  /data/home/shujia/UKB/CMRI/downloaded \\
      --out_dir  /data/home/shujia/UKB/CMRI/preprocessed_cmr_v3 \\
      --num_workers 8

  # Train:
  CUDA_VISIBLE_DEVICES=0 python contrastive_pretrain/train_cmr_v3.py \\
      --data_dir /data/home/shujia/UKB/CMRI/preprocessed_cmr_v3 \\
      --out_dir  contrastive_pretrain/checkpoints_cmr_v3
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.models_dualtower import TaskHead
from contrastive_pretrain.models_cmr_v3 import CMREncoderV3, NUM_FRAMES

MEDSAM_CKPT = str(ROOT / 'pretrained_weights/hf_cache/'
                   'models--flaviagiammarino--medsam-vit-base/blobs/'
                   'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c')

TRAIN_CSV = str(HERE / 'task_reports/task1_cmr_train.csv')
VAL_CSV   = str(HERE / 'task_reports/task1_cmr_val.csv')

CLS_COLS = ['composite_ischemic_hd', 'prevalent_I21']
REG_COLS = ['LV ejection fraction', 'LV end diastolic volume', 'LV end systolic volume']
REG_NORM = [
    {'mean': 58.7229, 'std': 7.3694},
    {'mean': 149.5626, 'std': 36.1142},
    {'mean': 62.615,   'std': 22.8526},
]
# task1_cmr positive rates: composite_ischemic_hd≈27.5%, prevalent_I21≈10.2%
POS_WEIGHTS = [2.64, 8.82]


# ── Augmentation ──────────────────────────────────────────────────────────────
def cmr_augment(cmr: torch.Tensor) -> torch.Tensor:
    """Spatially-consistent augmentation for all 13 CMR frames.
    cmr: (F, 1, H, W) normalised to [0,1]."""
    if random.random() < 0.5:
        cmr = cmr.flip(-1)
    if random.random() < 0.5:
        cmr = cmr.flip(-2)
    angle = random.uniform(-30, 30)
    frames = [TF.rotate(cmr[i], angle,
                        interpolation=TF.InterpolationMode.BILINEAR, fill=0.0)
              for i in range(cmr.shape[0])]
    cmr = torch.stack(frames, dim=0)
    scale = random.uniform(0.85, 1.15)
    cmr = (cmr * scale).clamp(0, 1)
    if random.random() < 0.5:
        cmr = (cmr + torch.randn_like(cmr) * 0.02).clamp(0, 1)
    return cmr


# ── Dataset ───────────────────────────────────────────────────────────────────
class CMRDatasetV3(Dataset):
    def __init__(self, csv_path: str, data_dir: str | None, is_train: bool = True):
        df = pd.read_csv(csv_path)
        if data_dir is not None:
            df['path'] = df['eid'].apply(
                lambda e: str(pathlib.Path(data_dir) / f'{e}.npy'))
        df = df[df['path'].apply(lambda p: pathlib.Path(p).exists())].reset_index(drop=True)
        self.rows     = df.to_dict('records')
        self.is_train = is_train
        print(f'[dataset] loaded {len(self.rows)} samples '
              f'({"train" if is_train else "val"})')

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        npy = np.load(r['path']).astype(np.float32)   # (13, 224, 224)
        assert npy.shape[0] == NUM_FRAMES, \
            f'expected {NUM_FRAMES} frames, got {npy.shape[0]} in {r["path"]}'

        cmr = torch.from_numpy(npy).unsqueeze(1)       # (13, 1, 224, 224)

        # per-frame normalise to [0,1] (already preprocessed but ensures clean range)
        for i in range(cmr.shape[0]):
            lo, hi = cmr[i].min(), cmr[i].max()
            if hi > lo:
                cmr[i] = (cmr[i] - lo) / (hi - lo)

        if self.is_train:
            cmr = cmr_augment(cmr)

        cls_labels = [float(r.get(c, -1)) for c in CLS_COLS]

        reg_vals, reg_mask = [], []
        for c in REG_COLS:
            v = r.get(c, float('nan'))
            if v is None or (isinstance(v, float) and np.isnan(v)):
                reg_vals.append(0.0); reg_mask.append(0)
            else:
                reg_vals.append(float(v)); reg_mask.append(1)

        return {
            'eid':        int(r['eid']),
            'cmr':        cmr,
            'cls_labels': torch.tensor(cls_labels, dtype=torch.float32),
            'reg_labels': torch.tensor(reg_vals,   dtype=torch.float32),
            'reg_mask':   torch.tensor(reg_mask,   dtype=torch.bool),
        }


# ── Loss ──────────────────────────────────────────────────────────────────────
def task_loss(cls_preds, reg_preds, cls_labels, reg_labels, reg_mask,
              pos_weights, reg_norm, device):
    cls_losses = []
    for i, pred in enumerate(cls_preds):
        valid = cls_labels[:, i] >= 0
        if not valid.any():
            continue
        pw = torch.tensor([pos_weights[i]], dtype=torch.float32, device=device)
        cls_losses.append(
            nn.BCEWithLogitsLoss(pos_weight=pw)(pred[valid], cls_labels[valid, i]))
    l_cls = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0., device=device)

    reg_losses = []
    for i, pred in enumerate(reg_preds):
        mu, sigma = reg_norm[i]['mean'], reg_norm[i]['std']
        t_norm = (reg_labels[:, i] - mu) / sigma
        valid  = reg_mask[:, i]
        if not valid.any():
            continue
        reg_losses.append(F.mse_loss(pred[valid], t_norm[valid]))
    l_reg = torch.stack(reg_losses).mean() if reg_losses else torch.tensor(0., device=device)

    return l_cls + l_reg


# ── LR scheduler with linear warmup + cosine decay ───────────────────────────
def get_scheduler(optimizer, warmup_epochs: int, total_epochs: int,
                  min_lr_ratio: float = 0.01):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(warmup_epochs, 1)
        progress = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(encoder, head, loader, device):
    encoder.eval(); head.eval()
    all_cls = [[] for _ in CLS_COLS]
    all_lbl = [[] for _ in CLS_COLS]
    all_reg = [[] for _ in REG_COLS]
    all_rgt = [[] for _ in REG_COLS]
    all_rmk = [[] for _ in REG_COLS]

    for batch in loader:
        cmr    = batch['cmr'].to(device)
        cls_gt = batch['cls_labels'].to(device)
        reg_gt = batch['reg_labels'].to(device)
        reg_mk = batch['reg_mask'].to(device)
        with torch.cuda.amp.autocast():
            _, z = encoder(cmr)
        cls_out, reg_out = head(z)
        for i in range(len(CLS_COLS)):
            all_cls[i].append(cls_out[i].cpu())
            all_lbl[i].append(cls_gt[:, i].cpu())
        for i in range(len(REG_COLS)):
            all_reg[i].append(reg_out[i].cpu())
            all_rgt[i].append(reg_gt[:, i].cpu())
            all_rmk[i].append(reg_mk[:, i].cpu())

    metrics = {}
    auc_vals = []
    for i, name in enumerate(CLS_COLS):
        logits = torch.cat(all_cls[i]).numpy()
        labels = torch.cat(all_lbl[i]).numpy()
        valid  = labels >= 0
        try:
            auc = roc_auc_score(labels[valid], logits[valid]) if valid.sum() >= 2 else float('nan')
        except Exception:
            auc = float('nan')
        metrics[f'auc/{name}'] = auc
        if not np.isnan(auc):
            auc_vals.append(auc)
    metrics['mean_auc'] = float(np.mean(auc_vals)) if auc_vals else float('nan')

    for i, name in enumerate(REG_COLS):
        pred = torch.cat(all_reg[i]).numpy()
        gt   = torch.cat(all_rgt[i]).numpy()
        mask = torch.cat(all_rmk[i]).numpy().astype(bool)
        if mask.sum() == 0:
            metrics[f'mae/{name}'] = float('nan')
            continue
        mu, std = REG_NORM[i]['mean'], REG_NORM[i]['std']
        metrics[f'mae/{name}'] = float(np.abs(pred[mask] * std + mu - gt[mask]).mean())

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir',      type=str,   required=True,
                        help='Directory with {eid}.npy from preprocess_cmr_v3.py')
    parser.add_argument('--spatial_pool',  type=int,   default=4,
                        help='Spatial pool size P: each frame → P×P region tokens (4→16 tok/frame, 209 total)')
    parser.add_argument('--epochs',        type=int,   default=60)
    parser.add_argument('--batch_size',    type=int,   default=24,
                        help='Used during frozen-backbone phase; halved to 8 when backbone unfreezes')
    parser.add_argument('--unfreeze_batch_size', type=int, default=8,
                        help='Batch size after backbone unfreeze (grad-checkpoint active)')
    parser.add_argument('--enc_lr',        type=float, default=1e-5)
    parser.add_argument('--head_lr',       type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int,   default=3)
    parser.add_argument('--freeze_epochs', type=int,   default=5,
                        help='Freeze MedSAM backbone for first N epochs')
    parser.add_argument('--patience',      type=int,   default=10)
    parser.add_argument('--out_dir',       type=str,
                        default=str(HERE / 'checkpoints_cmr_v3'))
    parser.add_argument('--medsam_ckpt',   type=str,   default=MEDSAM_CKPT)
    parser.add_argument('--resume',        action='store_true')
    parser.add_argument('--smoke_test',    action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = CMRDatasetV3(TRAIN_CSV, args.data_dir, is_train=True)
    val_ds   = CMRDatasetV3(VAL_CSV,   args.data_dir, is_train=False)
    if args.smoke_test:
        train_ds.rows = train_ds.rows[:32]
        val_ds.rows   = val_ds.rows[:16]

    def make_loaders(bs):
        tl = DataLoader(train_ds, batch_size=bs, shuffle=True,
                        num_workers=4, pin_memory=False, drop_last=True)
        vl = DataLoader(val_ds, batch_size=bs, shuffle=False,
                        num_workers=2, pin_memory=False)
        return tl, vl

    train_loader, val_loader = make_loaders(args.batch_size)

    print(f'[config] enc_lr={args.enc_lr}  wd={args.weight_decay}  '
          f'bs={args.batch_size}  freeze_epochs={args.freeze_epochs}  '
          f'warmup={args.warmup_epochs}')

    # ── model ──
    encoder = CMREncoderV3(
        proj_dim=256, embed_dim=768,
        spatial_pool=args.spatial_pool,
        transformer_heads=8, transformer_depth=2,
        medsam_ckpt=args.medsam_ckpt,
        freeze_backbone=False,
    ).to(device)
    head = TaskHead(in_dim=768, n_cls=len(CLS_COLS), n_reg=len(REG_COLS)).to(device)

    if args.freeze_epochs > 0:
        for p in encoder.backbone.parameters():
            p.requires_grad_(False)
        print(f'[freeze] backbone frozen for first {args.freeze_epochs} epochs')

    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': args.enc_lr,  'name': 'encoder'},
        {'params': head.parameters(),    'lr': args.head_lr, 'name': 'head'},
    ], weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = torch.cuda.amp.GradScaler()

    best_auc   = 0.0
    no_improve = 0
    start_ep   = 0
    history    = []
    last_ckpt  = out_dir / 'last.pth'

    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        encoder.load_state_dict(ckpt['encoder'])
        head.load_state_dict(ckpt['head'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep   = ckpt['epoch'] + 1
        best_auc   = ckpt.get('best_auc', 0.0)
        no_improve = ckpt.get('no_improve', 0)
        history    = ckpt.get('history', [])
        print(f'[resume] epoch {ckpt["epoch"]}  best_auc={best_auc:.4f}')

    n_trainable = (sum(p.numel() for p in encoder.parameters() if p.requires_grad)
                   + sum(p.numel() for p in head.parameters() if p.requires_grad))
    n_total = (sum(p.numel() for p in encoder.parameters())
               + sum(p.numel() for p in head.parameters()))
    print(f'Params: {n_total/1e6:.1f}M total  {n_trainable/1e6:.1f}M trainable')
    print(f'Peak GPU after init: {torch.cuda.max_memory_allocated(device)/1e9:.2f} GB')

    for epoch in range(start_ep, args.epochs):
        if args.freeze_epochs > 0 and epoch == args.freeze_epochs:
            for p in encoder.backbone.parameters():
                p.requires_grad_(True)
            encoder.grad_checkpoint = True   # activate gradient checkpointing
            train_loader, val_loader = make_loaders(args.unfreeze_batch_size)
            print(f'[unfreeze] backbone unfrozen at epoch {epoch}, '
                  f'grad_checkpoint=True, bs → {args.unfreeze_batch_size}')

        encoder.train(); head.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0

        for batch in train_loader:
            cmr    = batch['cmr'].to(device, non_blocking=True)
            cls_gt = batch['cls_labels'].to(device)
            reg_gt = batch['reg_labels'].to(device)
            reg_mk = batch['reg_mask'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                _, z = encoder(cmr)
                cls_out, reg_out = head(z)
                loss = task_loss(cls_out, reg_out, cls_gt, reg_gt, reg_mk,
                                 POS_WEIGHTS, REG_NORM, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), 3.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n += 1

        scheduler.step()
        train_loss /= max(n, 1)
        elapsed = time.time() - t0

        metrics = evaluate(encoder, head, val_loader, device)
        auc = metrics['mean_auc']
        mem = torch.cuda.max_memory_allocated(device) / 1e9
        lr_now = optimizer.param_groups[0]['lr']

        row = {'epoch': epoch, 'train_loss': train_loss, 'lr': lr_now, **metrics}
        history.append(row)

        print(f'Epoch {epoch:3d} | {elapsed:.0f}s | loss={train_loss:.4f} | '
              f'mean_AUC={auc:.4f}  lr={lr_now:.2e}  mem={mem:.2f}GB')
        for name in CLS_COLS:
            print(f'  {name}: AUC={metrics.get(f"auc/{name}", float("nan")):.4f}')

        torch.save({
            'epoch': epoch, 'encoder': encoder.state_dict(), 'head': head.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_auc': best_auc,
            'no_improve': no_improve, 'history': history,
        }, last_ckpt)
        with open(out_dir / 'metrics_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if auc > best_auc:
            best_auc = auc
            no_improve = 0
            torch.save({'epoch': epoch, 'encoder': encoder.state_dict(),
                        'head': head.state_dict(), 'metrics': metrics},
                       out_dir / 'best.pth')
            print(f'  [saved] best mean_AUC={best_auc:.4f}')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'[early stop] no improvement for {args.patience} epochs.')
                break

    print(f'\nTraining complete. Best mean_AUC = {best_auc:.4f}')
    print(f'Checkpoints: {out_dir}')


if __name__ == '__main__':
    main()
