"""
train_fundus_standalone.py  —  P0.2 Fundus baseline (single GPU)

FundusEncoder (RETFound ViT-L, full FT) + TaskHead on task2_fundus cohort.
Predicts: composite_ischemic_hd, prevalent_I21 (vascular diseases only)
No regression targets (not in fundus CSV).

Usage:
  CUDA_VISIBLE_DEVICES=0 python contrastive_pretrain/train_fundus_standalone.py \
      --out_dir contrastive_pretrain/checkpoints_fundus_standalone
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.models_dualtower import FundusEncoder, TaskHead

RETFOUND_CKPT = str(ROOT / 'RETFound_cfp_weights.pth')
TRAIN_CSV = str(HERE / 'task_reports/task2_fundus_train.csv')
VAL_CSV   = str(HERE / 'task_reports/task2_fundus_val.csv')

CLS_COLS    = ['composite_ischemic_hd', 'prevalent_I21']
POS_WEIGHTS = [2.39, 8.1]

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Dataset ───────────────────────────────────────────────────────────────────
def _fundus_transform(is_train: bool):
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3/4, 4/3)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(degrees=(-180, 180)),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 3.0))], p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])


class FundusDataset(Dataset):
    def __init__(self, csv_path: str, is_train: bool = True):
        df = pd.read_csv(csv_path)
        df = df[df['path'].apply(lambda p: pathlib.Path(p).exists())].reset_index(drop=True)
        self.rows      = df.to_dict('records')
        self.transform = _fundus_transform(is_train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = Image.open(r['path']).convert('RGB')
        fundus = self.transform(img)
        cls_labels = torch.tensor(
            [float(r.get(c, -1)) for c in CLS_COLS], dtype=torch.float32)
        return {'eid': int(r['eid']), 'fundus': fundus, 'cls_labels': cls_labels}


# ── Loss ──────────────────────────────────────────────────────────────────────
def cls_loss(cls_preds, cls_labels, pos_weights, device):
    losses = []
    for i, pred in enumerate(cls_preds):
        valid = cls_labels[:, i] >= 0
        if not valid.any():
            continue
        pw = torch.tensor([pos_weights[i]], dtype=torch.float32, device=device)
        losses.append(nn.BCEWithLogitsLoss(pos_weight=pw)(pred[valid], cls_labels[valid, i]))
    return torch.stack(losses).mean() if losses else torch.tensor(0., device=device)


# ── Evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(encoder, head, loader, device):
    encoder.eval(); head.eval()
    all_cls = [[] for _ in CLS_COLS]
    all_lbl = [[] for _ in CLS_COLS]

    for batch in loader:
        fundus = batch['fundus'].to(device)
        cls_gt = batch['cls_labels'].to(device)
        z = encoder(fundus)
        cls_out, _ = head(z)
        for i in range(len(CLS_COLS)):
            all_cls[i].append(cls_out[i].cpu())
            all_lbl[i].append(cls_gt[:, i].cpu())

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
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=8)
    parser.add_argument('--enc_lr',      type=float, default=1e-5,
                        help='ViT-L full FT needs lower LR than CMR')
    parser.add_argument('--head_lr',     type=float, default=1e-4)
    parser.add_argument('--patience',    type=int,   default=7)
    parser.add_argument('--grad_ckpt',   action='store_true', default=True,
                        help='Gradient checkpointing (saves ~3GB for ViT-L)')
    parser.add_argument('--out_dir',     type=str,
                        default=str(HERE / 'checkpoints_fundus_standalone'))
    parser.add_argument('--retfound_ckpt', type=str, default=RETFOUND_CKPT)
    parser.add_argument('--resume',      action='store_true')
    parser.add_argument('--smoke_test',  action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = FundusDataset(TRAIN_CSV, is_train=True)
    val_ds   = FundusDataset(VAL_CSV,   is_train=False)
    if args.smoke_test:
        train_ds.rows = train_ds.rows[:16]
        val_ds.rows   = val_ds.rows[:16]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=2, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=1, pin_memory=False)

    print(f'[data] train={len(train_ds)}  val={len(val_ds)}')

    # FundusEncoder (ViT-L) + 2-class head (no regression for fundus)
    encoder = FundusEncoder(ckpt=args.retfound_ckpt, freeze=False,
                            grad_ckpt=args.grad_ckpt).to(device)
    head    = TaskHead(in_dim=1024, n_cls=len(CLS_COLS), n_reg=0).to(device)

    optimizer = torch.optim.AdamW([
        {'params': encoder.parameters(), 'lr': args.enc_lr,  'name': 'encoder'},
        {'params': head.parameters(),    'lr': args.head_lr, 'name': 'head'},
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

    mem0 = torch.cuda.max_memory_allocated(device) / 1e9
    print(f'Peak GPU mem after model init: {mem0:.2f} GB')

    for epoch in range(start_ep, args.epochs):
        encoder.train(); head.train()
        t0 = time.time()
        train_loss = 0.0
        n = 0
        for batch in train_loader:
            fundus = batch['fundus'].to(device, non_blocking=True)
            cls_gt = batch['cls_labels'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                z       = encoder(fundus)
                cls_out, _ = head(z)
                loss    = cls_loss(cls_out, cls_gt, POS_WEIGHTS, device)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(head.parameters()), 5.0)
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

        row = {'epoch': epoch, 'train_loss': train_loss, **metrics}
        history.append(row)

        print(f'Epoch {epoch:3d} | {elapsed:.0f}s | train_loss={train_loss:.4f} | '
              f'val mean_AUC={auc:.4f}  peak_mem={mem:.2f}GB')
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
