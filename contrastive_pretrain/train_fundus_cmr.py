"""
train_fundus_cmr.py  —  Branch 1: Fundus ← CMR knowledge transfer

Three-loss objective (all weights configurable, no pre-assumed hierarchy):
  L1  embedding alignment  : 1 - cosine_sim(fundus_proj, cmr_proj).mean()
  L2  RKD structure        : MSE of normalised pairwise distances in proj space
  L3  task prediction      : BCE(task_head(fundus_feat), composite_ischemic_hd)

  total = lam1*L1 + lam2*L2 + lam3*L3

Grid search driver: run_fundus_cmr_hpo.sh
  Fix lam1=1.0, sweep lam2 × lam3 ∈ {0.1, 1.0, 5.0}² = 9 combos

CMR teacher:  CMREncoderV3  (completely frozen, loaded from --cmr_ckpt)
Fundus student: RETFound ViT-L, freeze first --freeze_blocks blocks (default 20)
  + projection head:  Linear(1024→256) + L2-norm
  + task head:        Linear(1024→1)  for composite_ischemic_hd

Data: paired_max_{train,val}.csv (cmr_npy_path overridden by --cmr_npy_dir)

Usage:
  CUDA_VISIBLE_DEVICES=0 python contrastive_pretrain/train_fundus_cmr.py \\
    --cmr_ckpt   contrastive_pretrain/checkpoints_cmr_v3/best.pth \\
    --cmr_npy_dir /data/home/shujia/UKB/CMRI/preprocessed_cmr_v3 \\
    --out_dir    contrastive_pretrain/checkpoints_fundus_cmr_lam1_1.0_lam2_1.0_lam3_1.0 \\
    --lam1 1.0 --lam2 1.0 --lam3 1.0
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
from torchvision import transforms
from PIL import Image
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.models_dualtower import FundusEncoder
from contrastive_pretrain.models_cmr_v3 import CMREncoderV3
from contrastive_pretrain.models_cmr_v3 import NUM_FRAMES as NUM_FRAMES_V3
from contrastive_pretrain.models_cmr import CMREncoder as CMREncoderV2

RETFOUND_CKPT = str(ROOT / 'pretrained_weights/RETFound_cfp_weights.pth')
MEDSAM_CKPT   = str(ROOT / 'pretrained_weights/hf_cache/'
                    'models--flaviagiammarino--medsam-vit-base/blobs/'
                    'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c')

LAX_SAX_DIR   = '/data/home/shujia/UKB/CMRI/preprocessed_lax_sax'   # 4-frame, 11079 paired eids
NUM_FRAMES_V2 = 4

TRAIN_CSV = str(HERE / 'task_reports/paired_max_train.csv')
VAL_CSV   = str(HERE / 'task_reports/paired_max_val.csv')

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ── Dataset ───────────────────────────────────────────────────────────────────
class PairedDataset(Dataset):
    """Paired (fundus, CMR) samples from paired_max CSV."""

    _train_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3/4, 4/3)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation((-180, 180)),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(5, sigma=(0.1, 3.0))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    _val_tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    def __init__(self, csv_path: str, cmr_npy_dir: str | None, is_train: bool):
        df = pd.read_csv(csv_path)
        if cmr_npy_dir is not None:
            df['cmr_npy_path'] = df['eid'].apply(
                lambda e: str(pathlib.Path(cmr_npy_dir) / f'{e}.npy'))
        # keep rows where both fundus and CMR exist
        df = df[
            df['cmr_npy_path'].apply(lambda p: pathlib.Path(p).exists()) &
            df['fundus_image_path'].apply(lambda p: pathlib.Path(p).exists())
        ].reset_index(drop=True)
        self.rows     = df.to_dict('records')
        self.is_train = is_train
        self.tfm      = self._train_tfm if is_train else self._val_tfm
        print(f'[dataset] {len(self.rows)} paired samples ({"train" if is_train else "val"})')

    def __len__(self):
        return len(self.rows)

    def _load_cmr(self, path: str) -> torch.Tensor:
        npy = np.load(path).astype(np.float32)    # (F, H, W)
        cmr = torch.from_numpy(npy).unsqueeze(1)  # (F, 1, H, W)
        for i in range(cmr.shape[0]):
            lo, hi = cmr[i].min(), cmr[i].max()
            if hi > lo:
                cmr[i] = (cmr[i] - lo) / (hi - lo)
        return cmr

    def __getitem__(self, idx):
        r   = self.rows[idx]
        img = Image.open(r['fundus_image_path']).convert('RGB')
        fundus = self.tfm(img)
        cmr    = self._load_cmr(r['cmr_npy_path'])
        label  = float(r.get('composite_ischemic_hd', -1))
        return {
            'fundus': fundus,
            'cmr':    cmr,
            'label':  torch.tensor(label, dtype=torch.float32),
        }


# ── Loss functions ─────────────────────────────────────────────────────────────
def l1_alignment(f_proj: torch.Tensor, c_proj: torch.Tensor) -> torch.Tensor:
    """1 - mean cosine similarity. Both inputs are already L2-normalised."""
    return 1.0 - F.cosine_similarity(f_proj, c_proj, dim=-1).mean()


def l2_rkd(f_proj: torch.Tensor, c_proj: torch.Tensor) -> torch.Tensor:
    """RKD distance loss: preserve pairwise distance structure across modalities."""
    d_f = torch.cdist(f_proj, f_proj, p=2)   # (B, B)
    d_c = torch.cdist(c_proj, c_proj, p=2)   # (B, B)
    # normalise by mean so scale is comparable
    d_f = d_f / (d_f.mean() + 1e-8)
    d_c = d_c / (d_c.mean() + 1e-8)
    return F.mse_loss(d_f, d_c)


def l3_task(logits: torch.Tensor, labels: torch.Tensor, device, pos_weight: float) -> torch.Tensor:
    valid = labels >= 0
    if not valid.any():
        return torch.tensor(0., device=device)
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    return F.binary_cross_entropy_with_logits(
        logits[valid], labels[valid], pos_weight=pw)


# ── Model components ──────────────────────────────────────────────────────────
class FundusProjector(nn.Module):
    """Projection head: 1024 → 256 (L2-normalised output for L1/L2 losses)."""
    def __init__(self, in_dim: int = 1024, proj_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, proj_dim),
        )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class FundusTaskHead(nn.Module):
    """Task head: 1024 → 1 logit for composite_ischemic_hd."""
    def __init__(self, in_dim: int = 1024, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )
        nn.init.trunc_normal_(self.net[1].weight, std=0.02)
        nn.init.trunc_normal_(self.net[3].weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)   # (B,)


# ── LR scheduler ──────────────────────────────────────────────────────────────
def get_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    def lr_lambda(ep):
        if ep < warmup_epochs:
            return (ep + 1) / max(warmup_epochs, 1)
        p = (ep - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Eval ──────────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(fundus_enc, projector, task_head, cmr_enc, loader, device, lam1, lam2, lam3, pos_weight=2.64):
    fundus_enc.eval(); projector.eval(); task_head.eval(); cmr_enc.eval()
    all_logits, all_labels = [], []
    total_l1 = total_l2 = total_l3 = 0.0
    n = 0
    for batch in loader:
        fundus = batch['fundus'].to(device)
        cmr    = batch['cmr'].to(device)
        labels = batch['label'].to(device)
        with torch.cuda.amp.autocast():
            f_feat = fundus_enc(fundus)
            f_proj = projector(f_feat)
            logits = task_head(f_feat)
            c_proj, _ = cmr_enc(cmr)
        total_l1 += l1_alignment(f_proj, c_proj).item()
        total_l2 += l2_rkd(f_proj, c_proj).item()
        total_l3 += l3_task(logits, labels, device, pos_weight).item()
        all_logits.append(logits.float().cpu())
        all_labels.append(labels.cpu())
        n += 1

    logits = torch.cat(all_logits).numpy()
    labels = torch.cat(all_labels).numpy()
    valid  = labels >= 0
    try:
        auc = roc_auc_score(labels[valid], logits[valid]) if valid.sum() >= 2 else float('nan')
    except Exception:
        auc = float('nan')
    return {
        'auc/composite_ischemic_hd': auc,
        'mean_auc': auc,
        'val_l1': total_l1 / max(n, 1),
        'val_l2': total_l2 / max(n, 1),
        'val_l3': total_l3 / max(n, 1),
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmr_ckpt',     type=str, required=True,
                        help='Path to CMR teacher best.pth (frozen)')
    parser.add_argument('--cmr_npy_dir',  type=str, default=None,
                        help='Override CMR npy dir. Default: use path from CSV (lax_sax for v2)')
    parser.add_argument('--cmr_version',  type=str, default='v2', choices=['v2', 'v3'],
                        help='v2=4-frame lax_sax (11079 pairs), v3=16-frame (2258 pairs)')
    parser.add_argument('--out_dir',      type=str, required=True)
    # loss weights — explored equally, no pre-assumed hierarchy
    parser.add_argument('--lam1', type=float, default=1.0,
                        help='Weight for L1 embedding alignment loss')
    parser.add_argument('--lam2', type=float, default=1.0,
                        help='Weight for L2 RKD structure loss')
    parser.add_argument('--lam3', type=float, default=1.0,
                        help='Weight for L3 task prediction loss')
    # model
    parser.add_argument('--freeze_blocks', type=int, default=20,
                        help='Freeze first N ViT-L blocks (0=train all, 24=freeze all)')
    parser.add_argument('--proj_dim',      type=int, default=256)
    parser.add_argument('--spatial_pool',  type=int, default=4,
                        help='Must match the CMR checkpoint (default 4)')
    # training
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--batch_size',    type=int,   default=16)
    parser.add_argument('--fundus_lr',     type=float, default=1e-5)
    parser.add_argument('--head_lr',       type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int,   default=2)
    parser.add_argument('--patience',      type=int,   default=10)
    parser.add_argument('--retfound_ckpt', type=str,   default=RETFOUND_CKPT)
    parser.add_argument('--smoke_test',    action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    # ── datasets ──
    # v2: use lax_sax dir (CSV paths), override only if explicitly set
    # v3: must override with preprocessed_cmr_v3 dir
    if args.cmr_version == 'v2' and args.cmr_npy_dir is None:
        cmr_npy_dir = None   # use cmr_npy_path column from CSV (points to lax_sax)
    else:
        cmr_npy_dir = args.cmr_npy_dir
    print(f'[cmr] version={args.cmr_version}  npy_dir={cmr_npy_dir or "(from CSV)"}')
    train_ds = PairedDataset(TRAIN_CSV, cmr_npy_dir, is_train=True)
    val_ds   = PairedDataset(VAL_CSV,   cmr_npy_dir, is_train=False)
    if args.smoke_test:
        train_ds.rows = train_ds.rows[:32]
        val_ds.rows   = val_ds.rows[:16]

    # compute pos_weight from actual paired training labels
    train_labels = [r.get('composite_ischemic_hd', -1) for r in train_ds.rows]
    n_pos = sum(1 for l in train_labels if l == 1)
    n_neg = sum(1 for l in train_labels if l == 0)
    pos_weight = n_neg / max(n_pos, 1)
    print(f'[pos_weight] n_pos={n_pos}  n_neg={n_neg}  pos_weight={pos_weight:.2f}')

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=False, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=False)

    # ── CMR teacher (frozen) ──
    if args.cmr_version == 'v3':
        cmr_enc = CMREncoderV3(
            proj_dim=args.proj_dim, embed_dim=768,
            spatial_pool=args.spatial_pool,
            transformer_heads=8, transformer_depth=2,
            medsam_ckpt=None,
            freeze_backbone=True,
        ).to(device)
    else:  # v2: 4-frame CMREncoder
        cmr_enc = CMREncoderV2(
            proj_dim=args.proj_dim, num_frames=NUM_FRAMES_V2,
            embed_dim=768,
            medsam_ckpt=None,
            freeze_backbone=True,
        ).to(device)
    ckpt = torch.load(args.cmr_ckpt, map_location=device)
    enc_sd = ckpt.get('encoder', ckpt)
    cmr_enc.load_state_dict(enc_sd, strict=True)
    for p in cmr_enc.parameters():
        p.requires_grad_(False)
    cmr_enc.eval()
    print(f'[CMR teacher v{args.cmr_version}] loaded {args.cmr_ckpt}  (frozen)')

    # ── Fundus student ──
    fundus_enc = FundusEncoder(
        ckpt=args.retfound_ckpt,
        freeze=False,
        freeze_blocks=args.freeze_blocks,
    ).to(device)
    projector  = FundusProjector(in_dim=1024, proj_dim=args.proj_dim).to(device)
    task_head  = FundusTaskHead(in_dim=1024).to(device)

    trainable_params = (
        list(p for p in fundus_enc.parameters() if p.requires_grad)
        + list(projector.parameters())
    )
    optimizer = torch.optim.AdamW([
        {'params': trainable_params,      'lr': args.fundus_lr, 'name': 'fundus'},
        {'params': task_head.parameters(),'lr': args.head_lr,   'name': 'head'},
    ], weight_decay=args.weight_decay)
    scheduler = get_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = torch.cuda.amp.GradScaler()

    n_trainable = sum(p.numel() for p in trainable_params) + sum(p.numel() for p in task_head.parameters())
    print(f'Trainable: {n_trainable/1e6:.1f}M  |  '
          f'lam1={args.lam1}  lam2={args.lam2}  lam3={args.lam3}  '
          f'freeze_blocks={args.freeze_blocks}  bs={args.batch_size}')

    best_auc   = 0.0
    no_improve = 0
    history    = []

    for epoch in range(args.epochs):
        fundus_enc.train(); projector.train(); task_head.train()
        t0 = time.time()
        train_loss = tl1 = tl2 = tl3 = 0.0
        n = 0

        for batch in train_loader:
            fundus = batch['fundus'].to(device, non_blocking=True)
            cmr    = batch['cmr'].to(device,    non_blocking=True)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                f_feat = fundus_enc(fundus)          # (B, 1024)
                f_proj = projector(f_feat)            # (B, 256) normalised
                logits = task_head(f_feat)            # (B,)
                with torch.no_grad():
                    c_proj, _ = cmr_enc(cmr)          # (B, 256) normalised

                loss1 = l1_alignment(f_proj, c_proj)
                loss2 = l2_rkd(f_proj, c_proj)
                loss3 = l3_task(logits, labels, device, pos_weight)
                loss  = args.lam1 * loss1 + args.lam2 * loss2 + args.lam3 * loss3

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(
                list(fundus_enc.parameters()) + list(projector.parameters())
                + list(task_head.parameters()), 3.0)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            tl1 += loss1.item(); tl2 += loss2.item(); tl3 += loss3.item()
            n += 1

        scheduler.step()
        train_loss /= max(n, 1)
        tl1 /= max(n, 1); tl2 /= max(n, 1); tl3 /= max(n, 1)
        elapsed = time.time() - t0
        mem     = torch.cuda.max_memory_allocated(device) / 1e9
        lr_now  = optimizer.param_groups[0]['lr']

        metrics = evaluate(fundus_enc, projector, task_head, cmr_enc,
                           val_loader, device, args.lam1, args.lam2, args.lam3, pos_weight)
        auc = metrics['mean_auc']

        row = {'epoch': epoch, 'train_loss': train_loss,
               'train_l1': tl1, 'train_l2': tl2, 'train_l3': tl3,
               'lr': lr_now, **metrics}
        history.append(row)
        print(f'Ep {epoch:3d} | {elapsed:.0f}s | loss={train_loss:.4f} '
              f'(L1={tl1:.3f} L2={tl2:.3f} L3={tl3:.3f}) | '
              f'AUC={auc:.4f}  lr={lr_now:.2e}  mem={mem:.2f}GB')

        torch.save({
            'epoch': epoch, 'fundus_enc': fundus_enc.state_dict(),
            'projector': projector.state_dict(), 'task_head': task_head.state_dict(),
            'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(), 'best_auc': best_auc,
            'no_improve': no_improve, 'history': history,
        }, out_dir / 'last.pth')
        with open(out_dir / 'metrics_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        if auc > best_auc:
            best_auc = auc; no_improve = 0
            torch.save({'epoch': epoch, 'fundus_enc': fundus_enc.state_dict(),
                        'projector': projector.state_dict(),
                        'task_head': task_head.state_dict(), 'metrics': metrics},
                       out_dir / 'best.pth')
            print(f'  [saved] best_auc={best_auc:.4f}')
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f'[early stop] patience={args.patience} exhausted')
                break

    print(f'\nDone. best_auc={best_auc:.4f}  lam=({args.lam1},{args.lam2},{args.lam3})')


if __name__ == '__main__':
    main()
