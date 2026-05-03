"""
train_stage1.py  —  P1 Enhanced CMR encoder via frozen-fundus CrossAttention (single GPU)

Architecture:
  Frozen RETFound → patch tokens [B, 196, 1024]
  CMREncoder (trainable) → z_c [B, 768]
  CrossAttention (trainable): q=z_c, kv=fundus_tokens → attn_out [B, 768]
  z_c_enhanced = z_c + attn_out
  TaskHead_c → cls [B, 2] + reg [B, 3]
  Loss: L_task_c (BCE + MSE), no alignment

Data: paired_max_{train,val}.csv  (6,725 / 1,379 paired samples)
CLS : composite_ischemic_hd, prevalent_I21
REG : LV ejection fraction, LV end diastolic volume, LV end systolic volume

Usage:
  CUDA_VISIBLE_DEVICES=1 python contrastive_pretrain/train_stage1.py \
      --out_dir contrastive_pretrain/checkpoints_stage1
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.datasets_paired_max import PairedMaxDataset, UniqueEIDSampler
from contrastive_pretrain.models_crossattn import Stage1EnhancedCMR

TRAIN_CSV = str(HERE / 'task_reports/paired_max_train.csv')
VAL_CSV   = str(HERE / 'task_reports/paired_max_val.csv')

CLS_COLS    = ['composite_ischemic_hd', 'prevalent_I21']
REG_COLS    = ['LV ejection fraction', 'LV end diastolic volume', 'LV end systolic volume']
POS_WEIGHTS = [2.39, 8.1]
REG_NORM    = [
    {'mean': 58.7229, 'std': 7.3694},
    {'mean': 149.5626, 'std': 36.1142},
    {'mean': 62.615,   'std': 22.8526},
]


# ── Task loss (vascular only: 2 cls, 3 reg) ───────────────────────────────────
def task_loss_fn(cls_preds, reg_preds, cls_labels, reg_labels, reg_mask,
                 pos_weights, reg_norm, device):
    cls_losses = []
    for i, pred in enumerate(cls_preds):
        valid = cls_labels[:, i] >= 0
        if not valid.any():
            continue
        pw = torch.tensor([pos_weights[i]], dtype=torch.float32, device=device)
        cls_losses.append(nn.BCEWithLogitsLoss(pos_weight=pw)(pred[valid], cls_labels[valid, i]))
    l_cls = torch.stack(cls_losses).mean() if cls_losses \
        else torch.tensor(0., device=device)

    reg_losses = []
    for i, pred in enumerate(reg_preds):
        mu, sig = reg_norm[i]['mean'], reg_norm[i]['std']
        t_norm  = (reg_labels[:, i] - mu) / sig
        valid   = reg_mask[:, i]
        if not valid.any():
            continue
        reg_losses.append(F.mse_loss(pred[valid], t_norm[valid]))
    l_reg = torch.stack(reg_losses).mean() if reg_losses \
        else torch.tensor(0., device=device)

    return l_cls + l_reg


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device, cls_cols, reg_cols, reg_norm):
    model.eval()
    all_cls = [[] for _ in cls_cols]
    all_lbl = [[] for _ in cls_cols]
    all_reg = [[] for _ in reg_cols]
    all_rgt = [[] for _ in reg_cols]
    all_rmk = [[] for _ in reg_cols]

    for batch in loader:
        fundus = batch['fundus'].to(device)
        cmr    = batch['cmr'].to(device)
        cls_gt = batch['cls_labels'].to(device)
        reg_gt = batch['reg_labels'].to(device)
        reg_mk = batch['reg_mask'].to(device)

        out = model(fundus, cmr)
        cls_c, reg_c = out['cls_c'], out['reg_c']

        for i in range(len(cls_cols)):
            all_cls[i].append(cls_c[i].cpu())
            all_lbl[i].append(cls_gt[:, i].cpu())
        for i in range(len(reg_cols)):
            all_reg[i].append(reg_c[i].cpu())
            all_rgt[i].append(reg_gt[:, i].cpu())
            all_rmk[i].append(reg_mk[:, i].cpu())

    metrics = {}
    auc_vals = []
    for i, name in enumerate(cls_cols):
        logits = torch.cat(all_cls[i]).numpy()
        labels = torch.cat(all_lbl[i]).numpy()
        valid  = labels >= 0
        try:
            auc = roc_auc_score(labels[valid], logits[valid]) if valid.sum() >= 2 else float('nan')
        except Exception:
            auc = float('nan')
        metrics[f'auc_c/{name}'] = auc
        if not np.isnan(auc):
            auc_vals.append(auc)

    metrics['mean_auc_c'] = float(np.mean(auc_vals)) if auc_vals else float('nan')

    for i, name in enumerate(reg_cols):
        pred = torch.cat(all_reg[i]).numpy()
        gt   = torch.cat(all_rgt[i]).numpy()
        mask = torch.cat(all_rmk[i]).numpy().astype(bool)
        mu, std = reg_norm[i]['mean'], reg_norm[i]['std']
        if mask.sum() == 0:
            metrics[f'mae_c/{name}'] = float('nan')
            continue
        metrics[f'mae_c/{name}'] = float(np.abs(pred[mask] * std + mu - gt[mask]).mean())

    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',       type=int,   default=25)
    parser.add_argument('--batch_size',   type=int,   default=8)
    parser.add_argument('--enc_lr',       type=float, default=5e-5,
                        help='LR for CMR encoder')
    parser.add_argument('--attn_lr',      type=float, default=1e-4,
                        help='LR for CrossAttention (new module, higher)')
    parser.add_argument('--head_lr',      type=float, default=5e-4,
                        help='LR for task head')
    parser.add_argument('--patience',     type=int,   default=5)
    parser.add_argument('--grad_ckpt',    action='store_true')
    parser.add_argument('--out_dir',      type=str,
                        default=str(HERE / 'checkpoints_stage1'))
    parser.add_argument('--retfound_ckpt', type=str,
                        default=str(ROOT / 'RETFound_cfp_weights.pth'))
    parser.add_argument('--medsam_ckpt',  type=str,
                        default=str(ROOT / 'pretrained_weights/hf_cache/'
                                    'models--flaviagiammarino--medsam-vit-base/blobs/'
                                    'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c'))
    parser.add_argument('--resume',       action='store_true')
    parser.add_argument('--smoke_test',   action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairedMaxDataset(TRAIN_CSV, is_train=True)
    val_ds   = PairedMaxDataset(VAL_CSV,   is_train=False)

    # drop unused label columns — dataset has 4 cls and 3 reg; we only use first 2 cls
    # (the batch['cls_labels'] is [B,4]; we index [:, :2] in loss)

    if args.smoke_test:
        train_ds.rows = train_ds.rows[:16]
        val_ds.rows   = val_ds.rows[:16]

    train_sampler = UniqueEIDSampler(train_ds, seed=42)
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=train_sampler, num_workers=2,
                               pin_memory=False, drop_last=True)
    val_loader    = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=1, pin_memory=False)

    print(f'[data] train={len(train_ds)}  val={len(val_ds)}')

    model = Stage1EnhancedCMR(
        retfound_ckpt=args.retfound_ckpt,
        medsam_ckpt=args.medsam_ckpt,
        n_cls=len(CLS_COLS),
        n_reg=len(REG_COLS),
    ).to(device)

    optimizer = torch.optim.AdamW([
        {'params': model.cmr_enc.parameters(),    'lr': args.enc_lr,  'name': 'cmr_enc'},
        {'params': model.cross_attn.parameters(), 'lr': args.attn_lr, 'name': 'cross_attn'},
        {'params': model.task_head.parameters(),  'lr': args.head_lr, 'name': 'task_head'},
    ], weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.enc_lr * 0.01)
    scaler = torch.cuda.amp.GradScaler()

    best_auc   = 0.0
    no_improve = 0
    start_ep   = 0
    history    = []
    last_ckpt  = out_dir / 'last.pth'
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep   = ckpt['epoch'] + 1
        best_auc   = ckpt.get('best_auc', 0.0)
        no_improve = ckpt.get('no_improve', 0)
        history    = ckpt.get('history', [])
        print(f'[resume] epoch {ckpt["epoch"]}  best_auc_c={best_auc:.4f}')

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_trainable/1e6:.1f}M  '
          f'(CMR enc + CrossAttn + TaskHead)')
    print(f'Peak GPU mem after model init: '
          f'{torch.cuda.max_memory_allocated(device)/1e9:.2f} GB')

    for epoch in range(start_ep, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            fundus = batch['fundus'].to(device, non_blocking=True)
            cmr    = batch['cmr'].to(device, non_blocking=True)
            # only first 2 cls cols (vascular)
            cls_gt = batch['cls_labels'][:, :len(CLS_COLS)].to(device)
            reg_gt = batch['reg_labels'].to(device)
            reg_mk = batch['reg_mask'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out  = model(fundus, cmr)
                loss = task_loss_fn(out['cls_c'], out['reg_c'],
                                    cls_gt, reg_gt, reg_mk,
                                    POS_WEIGHTS, REG_NORM, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            n += 1

        scheduler.step()
        train_loss /= max(n, 1)
        elapsed     = time.time() - t0

        metrics  = evaluate(model, val_loader, device, CLS_COLS, REG_COLS, REG_NORM)
        mean_auc = metrics['mean_auc_c']
        mem      = torch.cuda.max_memory_allocated(device) / 1e9

        row = {'epoch': epoch, 'train_loss': train_loss, **metrics}
        history.append(row)

        print(f'Epoch {epoch:3d} | {elapsed:.0f}s | train_loss={train_loss:.4f} | '
              f'val mean_AUC_c={mean_auc:.4f}  peak_mem={mem:.2f}GB')
        for name in CLS_COLS:
            print(f'  {name}: AUC_c={metrics.get(f"auc_c/{name}", float("nan")):.4f}')
        for name in REG_COLS:
            print(f'  {name}: MAE_c={metrics.get(f"mae_c/{name}", float("nan")):.2f}')

        # save every epoch
        torch.save({
            'epoch': epoch, 'model': model.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_auc': best_auc,
            'no_improve': no_improve, 'history': history,
        }, last_ckpt)
        with open(out_dir / 'metrics_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if mean_auc > best_auc:
            best_auc = mean_auc
            no_improve = 0
            torch.save({'epoch': epoch, 'model': model.state_dict(),
                        'metrics': metrics}, out_dir / 'best.pth')
            print(f'  [saved] best mean_AUC_c={best_auc:.4f}')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'[early stop] no improvement for {args.patience} epochs.')
                break

    print(f'\nTraining complete. Best mean_AUC_c = {best_auc:.4f}')
    print(f'Checkpoint: {out_dir}/best.pth')


if __name__ == '__main__':
    main()
