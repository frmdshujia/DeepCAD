"""
train_linear_probe.py  —  P0.3 / P2 / P4  Linear probe evaluation (single GPU)

Loads a frozen encoder checkpoint, extracts embeddings, trains a linear head.
Supports both fundus (RETFound) and CMR (MedSAM) encoders.

Usage:
  # P0.3 — frozen RETFound baseline on paired cohort
  CUDA_VISIBLE_DEVICES=1 python contrastive_pretrain/train_linear_probe.py \
      --encoder fundus --freeze \
      --data paired_max \
      --out_dir contrastive_pretrain/checkpoints_linear_probe_p03

  # P4 — evaluate Stage2 fundus encoder
  CUDA_VISIBLE_DEVICES=1 python contrastive_pretrain/train_linear_probe.py \
      --encoder fundus --encoder_ckpt checkpoints_stage2_alpha0.1/best.pth \
      --data paired_max \
      --out_dir contrastive_pretrain/checkpoints_linear_probe_stage2_a01
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.models_dualtower import FundusEncoder, TaskHead
from contrastive_pretrain.datasets_paired_max import PairedMaxDataset

RETFOUND_CKPT = str(ROOT / 'RETFound_cfp_weights.pth')

CLS_COLS = ['composite_ischemic_hd', 'prevalent_I21']
POS_WEIGHTS = [2.39, 8.1]

DATA_CSVS = {
    'paired_max': {
        'train': str(HERE / 'task_reports/paired_max_train.csv'),
        'val':   str(HERE / 'task_reports/paired_max_val.csv'),
    },
}


@torch.no_grad()
def extract_fundus_embeddings(encoder, loader, device):
    encoder.eval()
    all_z, all_y = [], []
    for batch in loader:
        fundus = batch['fundus'].to(device)
        z = encoder(fundus)
        all_z.append(z.cpu().float())
        all_y.append(batch['cls_labels'][:, :len(CLS_COLS)])
    return torch.cat(all_z).numpy(), torch.cat(all_y).numpy()


def run_linear_probe(z_train, y_train, z_val, y_val, device,
                     epochs=50, lr=1e-3, batch_size=256) -> dict:
    """PyTorch linear probe: 1-layer linear, BCE loss."""
    in_dim = z_train.shape[1]
    head   = nn.Linear(in_dim, len(CLS_COLS)).to(device)
    opt    = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)

    # normalise
    mean = z_train.mean(0, keepdims=True)
    std  = z_train.std(0, keepdims=True) + 1e-8
    z_tr = torch.tensor((z_train - mean) / std, dtype=torch.float32)
    z_vl = torch.tensor((z_val   - mean) / std, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)
    y_vl = torch.tensor(y_val,   dtype=torch.float32)

    n = len(z_tr)
    for ep in range(epochs):
        perm   = torch.randperm(n)
        head.train()
        for start in range(0, n, batch_size):
            idx    = perm[start:start + batch_size]
            xb, yb = z_tr[idx].to(device), y_tr[idx].to(device)
            pred   = head(xb)
            losses = []
            for i in range(len(CLS_COLS)):
                valid = yb[:, i] >= 0
                if valid.any():
                    pw   = torch.tensor([POS_WEIGHTS[i]], device=device)
                    losses.append(
                        nn.BCEWithLogitsLoss(pos_weight=pw)(pred[valid, i], yb[valid, i]))
            if losses:
                loss = torch.stack(losses).mean()
                opt.zero_grad(); loss.backward(); opt.step()

    # evaluate
    head.eval()
    with torch.no_grad():
        logits = head(z_vl.to(device)).cpu().numpy()

    metrics = {}
    auc_vals = []
    for i, name in enumerate(CLS_COLS):
        labels = y_vl[:, i].numpy()
        valid  = labels >= 0
        try:
            auc = roc_auc_score(labels[valid], logits[valid, i]) if valid.sum() >= 2 else float('nan')
        except Exception:
            auc = float('nan')
        metrics[f'auc/{name}'] = auc
        if not np.isnan(auc): auc_vals.append(auc)

    metrics['mean_auc'] = float(np.mean(auc_vals)) if auc_vals else float('nan')
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder',      type=str, default='fundus',
                        choices=['fundus'],
                        help='Encoder type (currently only fundus supported)')
    parser.add_argument('--encoder_ckpt', type=str, default=None,
                        help='Optional path to fine-tuned encoder checkpoint '
                             '(keys: encoder or fundus_enc). '
                             'If None, uses raw pretrained weights.')
    parser.add_argument('--data',         type=str, default='paired_max',
                        choices=list(DATA_CSVS.keys()))
    parser.add_argument('--probe_epochs', type=int, default=50)
    parser.add_argument('--probe_lr',     type=float, default=1e-3)
    parser.add_argument('--batch_size',   type=int, default=32)
    parser.add_argument('--out_dir',      type=str,
                        default=str(HERE / 'checkpoints_linear_probe'))
    parser.add_argument('--retfound_ckpt', type=str, default=RETFOUND_CKPT)
    parser.add_argument('--smoke_test',   action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csvs = DATA_CSVS[args.data]
    train_ds = PairedMaxDataset(csvs['train'], is_train=False)   # no augmentation for probe
    val_ds   = PairedMaxDataset(csvs['val'],   is_train=False)
    if args.smoke_test:
        train_ds.rows = train_ds.rows[:32]
        val_ds.rows   = val_ds.rows[:32]

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=1, pin_memory=False)

    print(f'[data] train={len(train_ds)}  val={len(val_ds)}')

    # build encoder (always frozen for linear probe)
    encoder = FundusEncoder(ckpt=args.retfound_ckpt, freeze=True,
                            grad_ckpt=False).to(device)

    if args.encoder_ckpt is not None:
        ckpt = torch.load(args.encoder_ckpt, map_location='cpu')
        state = ckpt.get('encoder',      None) or \
                ckpt.get('fundus_enc',   None) or \
                ckpt.get('fundus_enc_new', None) or \
                ckpt.get('model',        None)
        if state is None:
            # try top-level (whole model checkpoint)
            state = {k.replace('fundus_enc.', ''): v
                     for k, v in ckpt.items() if k.startswith('fundus_enc.')}
        if state:
            msg = encoder.load_state_dict(state, strict=False)
            print(f'[encoder] loaded {args.encoder_ckpt}  '
                  f'missing={len(msg.missing_keys)}  unexpected={len(msg.unexpected_keys)}')
        else:
            print(f'[encoder] WARNING: could not find encoder state in {args.encoder_ckpt}')

    for p in encoder.parameters():
        p.requires_grad_(False)
    encoder.eval()

    print('Extracting embeddings...')
    z_train, y_train = extract_fundus_embeddings(encoder, train_loader, device)
    z_val,   y_val   = extract_fundus_embeddings(encoder, val_loader,   device)
    print(f'z_train: {z_train.shape}  z_val: {z_val.shape}')
    print(f'Positive rates (train): '
          + ', '.join(f'{CLS_COLS[i]}={(y_train[:,i]==1).mean():.3f}'
                      for i in range(len(CLS_COLS))))

    print(f'Training linear probe ({args.probe_epochs} epochs)...')
    metrics = run_linear_probe(z_train, y_train, z_val, y_val, device,
                               epochs=args.probe_epochs, lr=args.probe_lr)

    print('\n=== Linear Probe Results ===')
    for k, v in metrics.items():
        print(f'  {k}: {v:.4f}')

    result = {
        'encoder_ckpt': args.encoder_ckpt,
        'data': args.data,
        'probe_epochs': args.probe_epochs,
        'metrics': metrics,
    }
    with open(out_dir / 'result.json', 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nResult saved: {out_dir}/result.json')


if __name__ == '__main__':
    main()
