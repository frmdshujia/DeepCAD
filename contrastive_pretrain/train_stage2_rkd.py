"""
train_stage2_rkd.py  —  P3 Fundus representation learning via RKD (single GPU)

Frozen anchor: Stage1 CMREncoder + CrossAttention → z_c_enhanced [B, 768]
Trainable:     FundusEncoder_NEW (RETFound init) → z_f [B, 1024]
               proj_head_f  [1024 → 256] L2-normalised
               proj_head_c  [768  → 256] L2-normalised  (maps anchor to align space)
               task_head_f  [1024 → 2 cls] (only used if --alpha > 0)

Loss: L_total = RKD(p_f, p_c) + alpha * L_task_f

Evaluation: every epoch run linear probe on frozen z_f against val labels.
            (train probe 50 epochs, report mean_auc_f on val)

Data: paired_max_{train,val}.csv  (6,725 / 1,379 paired samples)
CLS:  composite_ischemic_hd, prevalent_I21  (vascular only)

Usage:
  CUDA_VISIBLE_DEVICES=0 python contrastive_pretrain/train_stage2_rkd.py \
      --stage1_ckpt contrastive_pretrain/checkpoints_stage1/best.pth \
      --alpha 0.1 \
      --out_dir contrastive_pretrain/checkpoints_stage2_alpha0.1
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.datasets_paired_max import PairedMaxDataset, UniqueEIDSampler
from contrastive_pretrain.models_crossattn import Stage2RKDModel, rkd_loss

TRAIN_CSV = str(HERE / 'task_reports/paired_max_train.csv')
VAL_CSV   = str(HERE / 'task_reports/paired_max_val.csv')

CLS_COLS    = ['composite_ischemic_hd', 'prevalent_I21']
REG_COLS    = []   # fundus stage2 has no regression targets
POS_WEIGHTS = [2.39, 8.1]


# ── Task loss (classification only for fundus) ───────────────────────────────
def task_loss_fn(cls_preds, cls_labels, pos_weights, device):
    losses = []
    for i, pred in enumerate(cls_preds):
        valid = cls_labels[:, i] >= 0
        if not valid.any():
            continue
        pw = torch.tensor([pos_weights[i]], dtype=torch.float32, device=device)
        losses.append(nn.BCEWithLogitsLoss(pos_weight=pw)(pred[valid], cls_labels[valid, i]))
    return torch.stack(losses).mean() if losses else torch.tensor(0., device=device)


# ── Extract embeddings for linear probe ───────────────────────────────────────
@torch.no_grad()
def extract_embeddings(model, loader, device):
    """Return (embeddings [N, 1024], labels [N, 2]) from frozen fundus_enc_new."""
    model.eval()
    all_z = []
    all_y = []
    for batch in loader:
        fundus = batch['fundus'].to(device)
        cmr    = batch['cmr'].to(device)
        out    = model(fundus, cmr)
        all_z.append(out['z_f'].cpu().float())
        all_y.append(batch['cls_labels'][:, :len(CLS_COLS)])  # [B, 2]
    return torch.cat(all_z).numpy(), torch.cat(all_y).numpy()


def linear_probe_auc(z_train, y_train, z_val, y_val) -> dict:
    """Quick sklearn logistic regression linear probe — no GPU needed."""
    scaler = StandardScaler()
    z_tr   = scaler.fit_transform(z_train)
    z_vl   = scaler.transform(z_val)

    aucs = {}
    for i, name in enumerate(CLS_COLS):
        lbl_tr = y_train[:, i]
        lbl_vl = y_val[:, i]
        valid_tr = lbl_tr >= 0
        valid_vl = lbl_vl >= 0
        if valid_tr.sum() < 10 or lbl_tr[valid_tr].max() < 1:
            aucs[name] = float('nan')
            continue
        clf = LogisticRegression(max_iter=300, C=1.0, random_state=42)
        clf.fit(z_tr[valid_tr], lbl_tr[valid_tr])
        proba = clf.predict_proba(z_vl[valid_vl])[:, 1]
        try:
            aucs[name] = roc_auc_score(lbl_vl[valid_vl], proba)
        except Exception:
            aucs[name] = float('nan')

    vals = [v for v in aucs.values() if not np.isnan(v)]
    aucs['mean'] = float(np.mean(vals)) if vals else float('nan')
    return aucs


# ── Evaluation (optional: model forward task prediction) ─────────────────────
@torch.no_grad()
def evaluate_task(model, loader, device):
    """Evaluate task_head_f (only if model has it, alpha > 0)."""
    if model.task_head_f is None:
        return {}
    model.eval()
    all_cls = [[] for _ in CLS_COLS]
    all_lbl = [[] for _ in CLS_COLS]
    for batch in loader:
        fundus = batch['fundus'].to(device)
        cmr    = batch['cmr'].to(device)
        cls_gt = batch['cls_labels'][:, :len(CLS_COLS)].to(device)
        out    = model(fundus, cmr)
        for i in range(len(CLS_COLS)):
            all_cls[i].append(out['cls_f'][i].cpu())
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
        metrics[f'e2e_auc_f/{name}'] = auc
        if not np.isnan(auc): auc_vals.append(auc)
    metrics['e2e_mean_auc_f'] = float(np.mean(auc_vals)) if auc_vals else float('nan')
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=16)
    parser.add_argument('--enc_lr',      type=float, default=1e-5,
                        help='LR for FundusEncoder_NEW (ViT-L full FT)')
    parser.add_argument('--proj_lr',     type=float, default=1e-3,
                        help='LR for projection heads')
    parser.add_argument('--head_lr',     type=float, default=1e-4,
                        help='LR for task_head_f (only if alpha>0)')
    parser.add_argument('--alpha',       type=float, default=0.1,
                        help='Weight of L_task_f. 0=pure RKD, 1.0=joint')
    parser.add_argument('--rkd_temp',    type=float, default=4.0,
                        help='RKD temperature for pairwise similarity KL')
    parser.add_argument('--patience',    type=int,   default=7,
                        help='Early stop on linear probe mean_auc_f')
    parser.add_argument('--probe_every', type=int,   default=1,
                        help='Run linear probe evaluation every N epochs')
    parser.add_argument('--grad_ckpt',   action='store_true', default=True)
    parser.add_argument('--out_dir',     type=str,
                        default=str(HERE / 'checkpoints_stage2'))
    parser.add_argument('--stage1_ckpt', type=str,   default=None,
                        help='Path to Stage1 best.pth (CMR encoder anchor)')
    parser.add_argument('--retfound_ckpt', type=str,
                        default=str(ROOT / 'RETFound_cfp_weights.pth'))
    parser.add_argument('--medsam_ckpt', type=str,
                        default=str(ROOT / 'pretrained_weights/hf_cache/'
                                    'models--flaviagiammarino--medsam-vit-base/blobs/'
                                    'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c'))
    parser.add_argument('--resume',      action='store_true')
    parser.add_argument('--smoke_test',  action='store_true')
    args = parser.parse_args()

    device  = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = PairedMaxDataset(TRAIN_CSV, is_train=True)
    val_ds   = PairedMaxDataset(VAL_CSV,   is_train=False)
    if args.smoke_test:
        train_ds.rows = train_ds.rows[:32]
        val_ds.rows   = val_ds.rows[:32]

    train_sampler = UniqueEIDSampler(train_ds, seed=42)
    train_loader  = DataLoader(train_ds, batch_size=args.batch_size,
                               sampler=train_sampler, num_workers=2,
                               pin_memory=False, drop_last=True)
    val_loader    = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=1, pin_memory=False)

    print(f'[data] train={len(train_ds)}  val={len(val_ds)}')
    print(f'[config] alpha={args.alpha}  rkd_temp={args.rkd_temp}')

    # n_cls=2 for task_head_f (alpha>0), n_reg=0 (no regression in fundus standalone)
    n_cls = len(CLS_COLS) if args.alpha > 0 else 0
    model = Stage2RKDModel(
        retfound_ckpt=args.retfound_ckpt,
        stage1_ckpt=args.stage1_ckpt,
        medsam_ckpt=args.medsam_ckpt,
        n_cls=n_cls,
        n_reg=0,
        grad_ckpt=args.grad_ckpt,
    ).to(device)

    # layer-wise LR
    param_groups = [
        {'params': model.fundus_enc_new.parameters(), 'lr': args.enc_lr,  'name': 'enc'},
        {'params': model.proj_head_f.parameters(),    'lr': args.proj_lr, 'name': 'proj_f'},
        {'params': model.proj_head_c.parameters(),    'lr': args.proj_lr, 'name': 'proj_c'},
    ]
    if model.task_head_f is not None:
        param_groups.append(
            {'params': model.task_head_f.parameters(), 'lr': args.head_lr, 'name': 'head_f'})

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.enc_lr * 0.01)
    scaler = torch.cuda.amp.GradScaler()

    best_probe_auc = 0.0
    no_improve     = 0
    start_ep       = 0
    history        = []
    last_ckpt      = out_dir / 'last.pth'
    if args.resume and last_ckpt.exists():
        ckpt = torch.load(last_ckpt, map_location=device)
        model.fundus_enc_new.load_state_dict(ckpt.get('fundus_enc_new', {}), strict=False)
        model.proj_head_f.load_state_dict(ckpt.get('proj_head_f', {}), strict=False)
        model.proj_head_c.load_state_dict(ckpt.get('proj_head_c', {}), strict=False)
        if model.task_head_f is not None and 'task_head_f' in ckpt:
            model.task_head_f.load_state_dict(ckpt['task_head_f'], strict=False)
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_ep       = ckpt['epoch'] + 1
        best_probe_auc = ckpt.get('best_probe_auc', 0.0)
        no_improve     = ckpt.get('no_improve', 0)
        history        = ckpt.get('history', [])
        print(f'[resume] epoch {ckpt["epoch"]}  best_probe_auc={best_probe_auc:.4f}')

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Trainable params: {n_train/1e6:.1f}M')
    print(f'Peak GPU mem after model init: '
          f'{torch.cuda.max_memory_allocated(device)/1e9:.2f} GB')

    for epoch in range(start_ep, args.epochs):
        train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        total_rkd   = 0.0
        total_task  = 0.0
        total_total = 0.0
        n = 0

        for batch in train_loader:
            fundus = batch['fundus'].to(device, non_blocking=True)
            cmr    = batch['cmr'].to(device, non_blocking=True)
            cls_gt = batch['cls_labels'][:, :len(CLS_COLS)].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                out    = model(fundus, cmr)
                l_rkd  = rkd_loss(out['p_f'], out['p_c'], temperature=args.rkd_temp)
                if args.alpha > 0 and 'cls_f' in out:
                    l_task = task_loss_fn(out['cls_f'], cls_gt, POS_WEIGHTS, device)
                    loss   = l_rkd + args.alpha * l_task
                else:
                    l_task = torch.tensor(0., device=device)
                    loss   = l_rkd

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.trainable_parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()

            total_rkd   += l_rkd.item()
            total_task  += l_task.item()
            total_total += loss.item()
            n += 1

            # step-1 sanity check
            if epoch == start_ep and n == 1:
                pf_std = out['p_f'].float().std().item()
                pc_std = out['p_c'].float().std().item()
                pf_nrm = out['p_f'].float().norm(dim=-1).mean().item()
                print(f'\n[sanity ep{epoch} step1]  L_rkd={l_rkd.item():.4f}  '
                      f'L_task={l_task.item():.4f}')
                print(f'  p_f: std={pf_std:.4f} norm={pf_nrm:.4f} '
                      f'(should be std>0, norm≈1)')
                print(f'  p_c: std={pc_std:.4f} '
                      f'(should be std>0, non-constant anchor)\n')

        scheduler.step()
        elapsed  = time.time() - t0
        avg_rkd  = total_rkd  / max(n, 1)
        avg_task = total_task / max(n, 1)
        avg_loss = total_total / max(n, 1)

        # linear probe every probe_every epochs
        probe_metrics = {}
        if (epoch + 1) % args.probe_every == 0:
            z_tr, y_tr = extract_embeddings(model, train_loader, device)
            z_vl, y_vl = extract_embeddings(model, val_loader,   device)
            probe_metrics = linear_probe_auc(z_tr, y_tr, z_vl, y_vl)

        # optional e2e task head eval
        task_metrics = evaluate_task(model, val_loader, device)

        mem = torch.cuda.max_memory_allocated(device) / 1e9
        row = {
            'epoch': epoch, 'L_rkd': avg_rkd, 'L_task': avg_task,
            'L_total': avg_loss, **probe_metrics, **task_metrics,
        }
        history.append(row)

        print(f'Epoch {epoch:3d} | {elapsed:.0f}s | '
              f'L_rkd={avg_rkd:.4f}  L_task={avg_task:.4f}  '
              f'peak_mem={mem:.2f}GB')
        if probe_metrics:
            print(f'  linear probe: mean_auc_f={probe_metrics.get("mean", float("nan")):.4f}  '
                  + '  '.join(f'{k}={v:.4f}' for k, v in probe_metrics.items()
                               if k != 'mean' and not np.isnan(v)))

        # save every epoch
        ckpt_dict = {
            'epoch': epoch,
            'fundus_enc_new': model.fundus_enc_new.state_dict(),
            'proj_head_f': model.proj_head_f.state_dict(),
            'proj_head_c': model.proj_head_c.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'best_probe_auc': best_probe_auc,
            'no_improve': no_improve,
            'history': history,
        }
        if model.task_head_f is not None:
            ckpt_dict['task_head_f'] = model.task_head_f.state_dict()
        torch.save(ckpt_dict, last_ckpt)
        with open(out_dir / 'metrics_history.json', 'w') as f:
            json.dump(history, f, indent=2)

        probe_auc = probe_metrics.get('mean', float('nan'))
        if not np.isnan(probe_auc):
            if probe_auc > best_probe_auc:
                best_probe_auc = probe_auc
                no_improve = 0
                torch.save({
                    'epoch': epoch,
                    'fundus_enc': model.fundus_enc_new.state_dict(),
                    'proj_head_f': model.proj_head_f.state_dict(),
                    'probe_metrics': probe_metrics, **task_metrics,
                }, out_dir / 'best.pth')
                print(f'  [saved] best linear probe mean_auc_f={best_probe_auc:.4f}')
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f'[early stop] probe_auc no improvement for {args.patience} epochs.')
                    break

    print(f'\nTraining complete. Best linear probe mean_auc_f = {best_probe_auc:.4f}')
    print(f'Checkpoint: {out_dir}/best.pth')


if __name__ == '__main__':
    main()
