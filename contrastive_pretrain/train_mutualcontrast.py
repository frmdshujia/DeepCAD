"""
train_mutualcontrast.py

Mutual Contrastive + Downstream Task joint training.

Data: paired_max_{train,val,test}.csv (both fundus + CMR files verified).
Loss: L_task_f + gamma * L_task_c + lam * L_align (symmetric InfoNCE).
Primary metric: mean_AUC_f (fundus tower only; deployment target).
Early stopping on mean_AUC_f, patience=5.

Checkpoints:
  {out_dir}/mutualcontrast_best_full.pth   — full model + optimizer state
  {out_dir}/mutualcontrast_best_fundus.pt  — fundus encoder + task_head_f only

Usage (4-GPU DDP):
  torchrun --nproc_per_node=4 contrastive_pretrain/train_mutualcontrast.py \
      --epochs 30 --batch_size 8 --lam 0.3

Smoke test (CPU, 2 epochs, tiny batch):
  python contrastive_pretrain/train_mutualcontrast.py \
      --epochs 2 --batch_size 8 --smoke_test --no_dist
"""
from __future__ import annotations

import argparse
import os
import pathlib
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
import torch.distributed as dist
from sklearn.metrics import roc_auc_score

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.datasets_paired_max import PairedMaxDataset, UniqueEIDSampler
from contrastive_pretrain.models_mutualcontrast import MutualContrastModel, build_loss

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]
REG_COLS = [
    'LV ejection fraction',
    'LV end diastolic volume',
    'LV end systolic volume',
]

TRAIN_CSV = str(HERE / 'task_reports/paired_max_train.csv')
VAL_CSV   = str(HERE / 'task_reports/paired_max_val.csv')


# ── distributed helpers ───────────────────────────────────────────────────────
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


# ── collate ───────────────────────────────────────────────────────────────────
def collate_fn(batch: list, check_unique_eid: bool = False) -> dict:
    eids = [b['eid'] for b in batch]
    if check_unique_eid:
        assert len(eids) == len(set(eids)), \
            f'Duplicate EIDs in batch: {[e for e in eids if eids.count(e) > 1]}'
    keys = [k for k in batch[0] if k != 'eid']
    out = {'eid': eids}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch])
    return out


def collate_fn_train(batch: list) -> dict:
    return collate_fn(batch, check_unique_eid=True)


def collate_fn_eval(batch: list) -> dict:
    return collate_fn(batch, check_unique_eid=False)


# ── evaluation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model_raw: MutualContrastModel, loader: DataLoader,
             device: torch.device, criterion) -> dict:
    model_raw.eval()
    all_cls_f = [[] for _ in CLS_COLS]
    all_cls_c = [[] for _ in CLS_COLS]
    all_labels = [[] for _ in CLS_COLS]
    all_reg_f  = [[] for _ in REG_COLS]
    all_reg_c  = [[] for _ in REG_COLS]
    all_reg_gt = [[] for _ in REG_COLS]
    all_reg_mask = [[] for _ in REG_COLS]
    total_align = 0.0
    n_batches   = 0

    for batch in loader:
        fundus = batch['fundus'].to(device, non_blocking=True)
        cmr    = batch['cmr'].to(device, non_blocking=True)
        cls_gt = batch['cls_labels'].to(device)
        reg_gt = batch['reg_labels'].to(device)
        reg_mk = batch['reg_mask'].to(device)

        results = model_raw(fundus, cmr)
        losses  = criterion(results, {
            'cls_labels': cls_gt, 'reg_labels': reg_gt, 'reg_mask': reg_mk,
        })
        total_align += losses['align'].item()
        n_batches   += 1

        for i in range(len(CLS_COLS)):
            all_cls_f[i].append(results['cls_f'][i].cpu())
            all_cls_c[i].append(results['cls_c'][i].cpu())
            all_labels[i].append(cls_gt[:, i].cpu())
        for i in range(len(REG_COLS)):
            all_reg_f[i].append(results['reg_f'][i].cpu())
            all_reg_c[i].append(results['reg_c'][i].cpu())
            all_reg_gt[i].append(reg_gt[:, i].cpu())
            all_reg_mask[i].append(reg_mk[:, i].cpu())

    def _auc(logits_list, labels_list):
        logits = torch.cat(logits_list).numpy()
        labels = torch.cat(labels_list).numpy()
        valid  = labels >= 0
        if valid.sum() < 2 or labels[valid].max() < 1:
            return float('nan')
        try:
            return roc_auc_score(labels[valid], logits[valid])
        except Exception:
            return float('nan')

    def _mae(pred_list, gt_list, mask_list, norm_mean, norm_std):
        pred = torch.cat(pred_list).numpy()
        gt   = torch.cat(gt_list).numpy()
        mask = torch.cat(mask_list).numpy().astype(bool)
        if mask.sum() == 0:
            return float('nan')
        pred_denorm = pred[mask] * norm_std + norm_mean
        return float(np.abs(pred_denorm - gt[mask]).mean())

    reg_norms = [
        (58.72, 7.37),
        (149.56, 36.11),
        (62.62, 22.85),
    ]

    metrics = {}
    auc_f_vals, auc_c_vals = [], []
    for i, name in enumerate(CLS_COLS):
        af = _auc(all_cls_f[i], all_labels[i])
        ac = _auc(all_cls_c[i], all_labels[i])
        metrics[f'auc_f/{name}'] = af
        metrics[f'auc_c/{name}'] = ac
        if not np.isnan(af): auc_f_vals.append(af)
        if not np.isnan(ac): auc_c_vals.append(ac)

    metrics['mean_auc_f'] = float(np.mean(auc_f_vals)) if auc_f_vals else float('nan')
    metrics['mean_auc_c'] = float(np.mean(auc_c_vals)) if auc_c_vals else float('nan')

    for i, name in enumerate(REG_COLS):
        mu, std = reg_norms[i]
        metrics[f'mae_f/{name}'] = _mae(all_reg_f[i], all_reg_gt[i], all_reg_mask[i], mu, std)
        metrics[f'mae_c/{name}'] = _mae(all_reg_c[i], all_reg_gt[i], all_reg_mask[i], mu, std)

    metrics['val_align'] = total_align / max(n_batches, 1)
    return metrics


# ── training loop ─────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scaler, criterion,
                    device, sampler, epoch) -> dict:
    model.train()
    if sampler is not None:
        sampler.set_epoch(epoch)

    total = {'total': 0., 'task_f': 0., 'task_c': 0., 'align': 0.}
    n = 0
    for batch in loader:
        fundus = batch['fundus'].to(device, non_blocking=True)
        cmr    = batch['cmr'].to(device, non_blocking=True)
        cls_gt = batch['cls_labels'].to(device)
        reg_gt = batch['reg_labels'].to(device)
        reg_mk = batch['reg_mask'].to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            results = model(fundus, cmr) if not isinstance(model, DDP) else model(fundus, cmr)
            losses  = criterion(
                results,
                {'cls_labels': cls_gt, 'reg_labels': reg_gt, 'reg_mask': reg_mk},
            )
        scaler.scale(losses['total']).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        scaler.step(optimizer)
        scaler.update()

        for k in total:
            total[k] += losses[k].item()
        n += 1

    return {k: v / max(n, 1) for k, v in total.items()}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs',      type=int,   default=30)
    parser.add_argument('--batch_size',  type=int,   default=8,
                        help='Per-GPU batch size')
    parser.add_argument('--lr',          type=float, default=1e-4)
    parser.add_argument('--gamma',       type=float, default=1.0,
                        help='Weight for CMR task loss')
    parser.add_argument('--lam',         type=float, default=0.3,
                        help='Weight for contrastive alignment loss')
    parser.add_argument('--temperature', type=float, default=0.07)
    parser.add_argument('--patience',    type=int,   default=5)
    parser.add_argument('--grad_ckpt',   action='store_true',
                        help='Gradient checkpointing for FundusEncoder (saves ~3GB/GPU)')
    parser.add_argument('--out_dir',     type=str,
                        default=str(HERE / 'checkpoints_mutualcontrast'))
    parser.add_argument('--retfound_ckpt', type=str, default=str(
        ROOT / 'RETFound_cfp_weights.pth'))
    parser.add_argument('--medsam_ckpt', type=str, default=str(
        ROOT / 'pretrained_weights/hf_cache/'
               'models--flaviagiammarino--medsam-vit-base/blobs/'
               'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c'))
    parser.add_argument('--smoke_test',  action='store_true',
                        help='8 samples, 2 epochs — fast pipeline check')
    parser.add_argument('--no_dist',     action='store_true',
                        help='Force single-GPU mode (skip torchrun init)')
    args = parser.parse_args()

    distributed = False if args.no_dist else setup_dist()
    local_rank  = int(os.environ.get('LOCAL_RANK', 0))
    device      = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

    # ── datasets ──────────────────────────────────────────────────────────────
    train_ds = PairedMaxDataset(TRAIN_CSV, is_train=True)
    val_ds   = PairedMaxDataset(VAL_CSV,   is_train=False)

    if args.smoke_test:
        log('=== SMOKE TEST: 8 train / 8 val samples ===')
        train_ds.rows = train_ds.rows[:8]
        val_ds.rows   = val_ds.rows[:8]
        args.batch_size = 2   # guarantee full batch even if <8 unique EIDs

    train_sampler = UniqueEIDSampler(train_ds, seed=42)

    if distributed:
        # Wrap UniqueEIDSampler in DistributedSampler-like behaviour:
        # each GPU gets a non-overlapping slice of the epoch's index list.
        # Simple approach: let each GPU see the full shuffled list but skip
        # by world_size — this preserves the unique-EID-per-batch guarantee
        # within each GPU's local batch.
        from torch.utils.data import DistributedSampler
        dist_sampler = DistributedSampler(train_ds, shuffle=False)
        log('[WARNING] DDP mode: using DistributedSampler; '
            'UniqueEIDSampler enforced via collate assertion only.')
        effective_sampler = dist_sampler
    else:
        effective_sampler = train_sampler

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=effective_sampler,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn_train,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=2,
        pin_memory=True,
        collate_fn=collate_fn_eval,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    model = MutualContrastModel(
        retfound_ckpt=args.retfound_ckpt,
        medsam_ckpt=args.medsam_ckpt,
        grad_ckpt=args.grad_ckpt,
    ).to(device)

    if distributed:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    criterion = build_loss(gamma=args.gamma, lam=args.lam, temperature=args.temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = torch.cuda.amp.GradScaler()

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── smoke test: gradient flow check (fp32, no AMP to avoid init-scale NaN) ─
    if args.smoke_test and is_main():
        log('\n--- Gradient flow check (1 step, fp32) ---')
        model.train()
        # use a simple loader (no UniqueEIDSampler) just to get a batch for grad check
        _check_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                   shuffle=False, collate_fn=collate_fn_eval)
        batch  = next(iter(_check_loader))
        fundus = batch['fundus'].to(device)
        cmr    = batch['cmr'].to(device)
        cls_gt = batch['cls_labels'].to(device)
        reg_gt = batch['reg_labels'].to(device)
        reg_mk = batch['reg_mask'].to(device)

        model_raw = model.module if isinstance(model, DDP) else model
        # fp32 forward — avoids GradScaler initial-scale overflow
        results = model_raw(fundus, cmr)
        losses  = criterion(results,
            {'cls_labels': cls_gt, 'reg_labels': reg_gt, 'reg_mask': reg_mk})
        losses['total'].backward()

        def _first_grad(params):
            for p in params:
                if p.grad is not None and not torch.isnan(p.grad).any():
                    return p.grad
            return None

        fe_grad = _first_grad(model_raw.fundus_enc.parameters())
        ce_grad = _first_grad(model_raw.cmr_enc.parameters())
        fe_norm = fe_grad.norm().item() if fe_grad is not None else float('nan')
        ce_norm = ce_grad.norm().item() if ce_grad is not None else float('nan')
        log(f'  fundus_enc grad norm: {fe_norm:.4f}  cmr_enc grad norm: {ce_norm:.4f}')
        log(f'  L_task_f={losses["task_f"].item():.4f}  '
            f'L_task_c={losses["task_c"].item():.4f}  '
            f'L_align={losses["align"].item():.4f}  '
            f'L_total={losses["total"].item():.4f}')
        assert not torch.isnan(losses['total']), 'NaN loss in smoke test!'
        assert fe_norm > 0 and not np.isnan(fe_norm), 'fundus_enc has zero/NaN gradient!'
        assert ce_norm > 0 and not np.isnan(ce_norm), 'cmr_enc has zero/NaN gradient!'
        log('  [PASS] gradients non-zero, loss finite')
        optimizer.zero_grad()

    # ── training ──────────────────────────────────────────────────────────────
    best_auc_f  = 0.0
    no_improve  = 0
    history     = []   # list of dicts, one per epoch

    if device.type == 'cuda' and is_main():
        mem = torch.cuda.max_memory_allocated(device) / 1e9
        log(f'Peak GPU memory after model init: {mem:.2f} GB')

    for epoch in range(args.epochs):
        if not distributed and not args.smoke_test:
            train_sampler.set_epoch(epoch)
        elif distributed:
            effective_sampler.set_epoch(epoch)

        t0         = time.time()
        model_raw  = model.module if isinstance(model, DDP) else model
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, criterion,
            device, train_sampler if not distributed else None, epoch,
        )
        scheduler.step()

        val_metrics = evaluate(model_raw, val_loader, device, criterion)
        dt          = time.time() - t0
        mean_auc_f  = val_metrics['mean_auc_f']

        log(f'\nEpoch {epoch:3d} | {dt:.0f}s | '
            f'train L_total={train_loss["total"]:.4f}  '
            f'task_f={train_loss["task_f"]:.4f}  '
            f'task_c={train_loss["task_c"]:.4f}  '
            f'align={train_loss["align"]:.4f}')
        log(f'         val mean_AUC_f={mean_auc_f:.4f}  '
            f'mean_AUC_c={val_metrics["mean_auc_c"]:.4f}  '
            f'val_align={val_metrics["val_align"]:.4f}')
        for name in CLS_COLS:
            log(f'           {name}: AUC_f={val_metrics[f"auc_f/{name}"]:.4f}  '
                f'AUC_c={val_metrics[f"auc_c/{name}"]:.4f}')
        for i, name in enumerate(REG_COLS):
            log(f'           {name}: MAE_f={val_metrics[f"mae_f/{name}"]:.2f}  '
                f'MAE_c={val_metrics[f"mae_c/{name}"]:.2f}')

        if device.type == 'cuda' and is_main():
            mem = torch.cuda.max_memory_allocated(device) / 1e9
            log(f'         peak_mem={mem:.2f} GB')

        # record history
        if is_main():
            row = {'epoch': epoch, 'dt': dt}
            row.update({f'train_{k}': v for k, v in train_loss.items()})
            row.update(val_metrics)
            history.append(row)

        if is_main() and mean_auc_f > best_auc_f:
            best_auc_f = mean_auc_f
            no_improve = 0

            torch.save({
                'epoch':      epoch,
                'model':      model_raw.state_dict(),
                'optimizer':  optimizer.state_dict(),
                'scheduler':  scheduler.state_dict(),
                'scaler':     scaler.state_dict(),
                'metrics':    val_metrics,
                'args':       vars(args),
            }, out_dir / 'mutualcontrast_best_full.pth')

            torch.save({
                'epoch':      epoch,
                'metrics':    val_metrics,
                'state_dict': model_raw.fundus_only_state_dict(),
            }, out_dir / 'mutualcontrast_best_fundus.pt')

            log(f'  [saved] best mean_AUC_f={best_auc_f:.4f}')
        elif is_main():
            no_improve += 1

        if no_improve >= args.patience:
            log(f'Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)')
            break

    # ── save history & print final summary ────────────────────────────────────
    if is_main() and history:
        import json as _json
        hist_path = out_dir / 'metrics_history.json'
        with open(hist_path, 'w') as f:
            _json.dump(history, f, indent=2)
        log(f'\nHistory saved → {hist_path}')

        # find best epoch row
        best_row = max(history, key=lambda r: r.get('mean_auc_f', 0.0))
        log('\n' + '=' * 65)
        log('FINAL RESULTS  (best val epoch = {:d})'.format(best_row['epoch']))
        log('=' * 65)
        log(f'  mean_AUC_f (fundus, deploy target) : {best_row["mean_auc_f"]:.4f}')
        log(f'  mean_AUC_c (CMR,  reference)       : {best_row["mean_auc_c"]:.4f}')
        log('')
        log('  Classification AUC (val):')
        log(f'  {"Task":<42} {"AUC_f":>7}  {"AUC_c":>7}')
        log('  ' + '-' * 58)
        for name in CLS_COLS:
            af = best_row.get(f'auc_f/{name}', float('nan'))
            ac = best_row.get(f'auc_c/{name}', float('nan'))
            log(f'  {name:<42} {af:>7.4f}  {ac:>7.4f}')
        log('')
        log('  Regression MAE (val, original units):')
        log(f'  {"Task":<42} {"MAE_f":>7}  {"MAE_c":>7}')
        log('  ' + '-' * 58)
        units = ['%', 'mL', 'mL']
        for i, name in enumerate(REG_COLS):
            mf = best_row.get(f'mae_f/{name}', float('nan'))
            mc = best_row.get(f'mae_c/{name}', float('nan'))
            log(f'  {name:<42} {mf:>7.2f}  {mc:>7.2f}  {units[i]}')
        log('')
        log(f'  val L_align : {best_row.get("val_align", float("nan")):.4f}')
        log('=' * 65)

    log(f'\nTraining complete. Best mean_AUC_f = {best_auc_f:.4f}')
    log(f'Checkpoints in: {out_dir}')


if __name__ == '__main__':
    main()
