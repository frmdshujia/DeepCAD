"""trainval_tier1.py -- Stage 2 Tier 1 training (1,196 EIDs, native 3-split).

Uses fundus_table.csv's original train/val/test split:
  train = 959 EID, val = 115 EID, test = 122 EID
(all filtered by CMR npy availability)

Key differences vs trainval_tier0.py:
  - No artificial 80/20 sub-split. Use the CSV's native split.
  - Train only on split='train'; eval retrieval on split='val' every `--eval_every`
    epochs; eval on split='test' ONLY at final (held-out, reported once).
  - Saves best checkpoint (by val R@1) in an encoder-only format that matches
    what end2end_stage2_ceiling.py expects.
  - --freeze_backbones flag kept for parallel frozen/unfrozen comparison.

Deliverable file:
  {out_dir}/checkpoint_best.pth   : {'fundus_model': ..., 'cmr_model': ...,
                                     'args': {...}, 'val_metrics': {...}}
  {out_dir}/history.json
  {out_dir}/run.log                : stdout redirect
"""
from __future__ import annotations
import sys, os, time, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, Sampler

from datasets_stage2 import FundusCMRImageDataset
from models_stage2 import Stage2DualTower, build_stage2_param_groups
from loss_contrast import hard_infonce_loss


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return math.sqrt(total)


class UniqueEIDSampler(Sampler):
    """One random fundus image per unique EID per epoch."""

    def __init__(self, dataset_eids, seed: int = 0):
        self.seed = seed
        self.epoch = 0
        self.eid_to_indices: dict = {}
        for idx, e in enumerate(dataset_eids):
            self.eid_to_indices.setdefault(int(e), []).append(idx)
        self.unique_eids = list(self.eid_to_indices.keys())

    def __len__(self):
        return len(self.unique_eids)

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        shuffled = rng.permutation(self.unique_eids)
        out = []
        for e in shuffled:
            out.append(int(rng.choice(self.eid_to_indices[int(e)])))
        return iter(out)

    def set_epoch(self, epoch: int):
        self.epoch = epoch


@torch.no_grad()
def eval_retrieval(model, dataset, device, batch_size: int = 16, eval_num_workers: int = 3):
    """Compute retrieval R@1/5/10 over unique-EID representatives of `dataset`."""
    model.eval()
    seen = set()
    idx_list = []
    for i, e in enumerate(dataset.eids):
        e = int(e)
        if e not in seen:
            seen.add(e)
            idx_list.append(i)
    if not idx_list:
        model.train()
        return {'N': 0}

    subset = Subset(dataset, idx_list)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False,
                        num_workers=eval_num_workers, pin_memory=True)

    zfs, zcs = [], []
    for img, cmr, _ in loader:
        img = img.to(device, non_blocking=True)
        cmr = cmr.to(device, non_blocking=True)
        zf, zc = model(img, cmr)
        zfs.append(zf); zcs.append(zc)
    zf = torch.cat(zfs, 0)
    zc = torch.cat(zcs, 0)
    N = zf.shape[0]
    sim = zf @ zc.T

    ranks_fc = sim.argsort(dim=1, descending=True)
    correct_fc = (ranks_fc == torch.arange(N, device=device).unsqueeze(1))
    ranks_cf = sim.T.argsort(dim=1, descending=True)
    correct_cf = (ranks_cf == torch.arange(N, device=device).unsqueeze(1))

    out = {'N': int(N)}
    for k in (1, 5, 10):
        k_eff = min(k, N)
        r_fc = correct_fc[:, :k_eff].any(dim=1).float().mean().item()
        r_cf = correct_cf[:, :k_eff].any(dim=1).float().mean().item()
        out[f'R@{k}_f2c'] = r_fc
        out[f'R@{k}_c2f'] = r_cf
        out[f'R@{k}_avg'] = (r_fc + r_cf) / 2
    model.train()
    return out


def _fmt(m, tag):
    if m.get('N', 0) == 0:
        return f'{tag}: N=0'
    return (f"{tag} N={m['N']} "
            f"R@1={m['R@1_avg']:.3f} R@5={m['R@5_avg']:.3f} R@10={m['R@10_avg']:.3f} "
            f"(f2c R@1={m['R@1_f2c']:.3f}, c2f R@1={m['R@1_c2f']:.3f})")


def save_encoder_checkpoint(path, model, args, val_metrics):
    """Save encoder-only weights in the format expected by downstream scripts."""
    ckpt = {
        'fundus_model': model.fundus.state_dict(),
        'cmr_model': model.cmr.state_dict(),
        'args': {
            'proj_dim': 256,
            'num_frames': 4,
            'drop_path': 0.0,
            'freeze_backbones': args.freeze_backbones,
            'base_lr': args.base_lr,
            'temperature': args.temperature,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
        },
        'val_metrics': val_metrics,
    }
    torch.save(ckpt, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str,
                    default='contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table.csv')
    ap.add_argument('--cmr_dir', type=str,
                    default='/data/home/shujia/UKB/CMRI/preprocessed_lax_sax')
    ap.add_argument('--medsam_ckpt', type=str,
                    default='pretrained_weights/hf_cache/models--flaviagiammarino--medsam-vit-base/snapshots/a3cb4c518fc3ae8beaf1af0cd3a43867cfa28335/pytorch_model.bin')
    ap.add_argument('--retfound_ckpt', type=str, default='RETFound_cfp_weights.pth')
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--base_lr', type=float, default=5e-5)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--eval_every', type=int, default=2)
    ap.add_argument('--train_eval_every', type=int, default=10,
                    help='eval retrieval on train subset this often (0=never, only at end)')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--lr_schedule', choices=['const', 'cosine'], default='cosine')
    ap.add_argument('--warmup_epochs', type=int, default=10)
    ap.add_argument('--min_lr', type=float, default=1e-7)
    ap.add_argument('--freeze_backbones', action='store_true')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--train_augment', action='store_true',
                    help='Enable fundus data augmentation during training. '
                         'Off by default because strong rotations + asymmetric CMR '
                         'side caused representation collapse in first attempt.')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[tier1] device={device}  out_dir={args.out_dir}  freeze={args.freeze_backbones}')

    # Datasets
    ds_train = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='train', num_frames=4,
        use_eval_transform=(not args.train_augment),
    )
    ds_val = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='val', num_frames=4,
        use_eval_transform=True,
    )
    ds_test = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='test', num_frames=4,
        use_eval_transform=True,
    )
    print(f'[tier1] train: {len(ds_train)} pairs, {len(set(ds_train.eids))} unique EIDs')
    print(f'[tier1] val  : {len(ds_val)} pairs, {len(set(ds_val.eids))} unique EIDs')
    print(f'[tier1] test : {len(ds_test)} pairs, {len(set(ds_test.eids))} unique EIDs')

    sampler = UniqueEIDSampler(ds_train.eids, seed=args.seed)
    loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    n_iters_per_epoch = len(loader)
    print(f'[tier1] iters/epoch = {n_iters_per_epoch}')

    # Also build an eval dataset for train retrieval (uses eval transform, no augment)
    ds_train_eval = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='train', num_frames=4,
        use_eval_transform=True,
    )

    # Model
    model = Stage2DualTower(
        proj_dim=256, num_frames=4, drop_path_rate=0.0,
        medsam_ckpt=args.medsam_ckpt, retfound_ckpt=args.retfound_ckpt,
        freeze_fundus_backbone=args.freeze_backbones,
        freeze_cmr_backbone=args.freeze_backbones,
    ).to(device)
    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[tier1] params: total={n_tot/1e6:.1f}M  trainable={n_tr/1e6:.2f}M')

    groups = build_stage2_param_groups(
        model, weight_decay=args.weight_decay,
        fundus_layer_decay=0.75, cmr_layer_decay=0.80,
        fundus_proj_lr_scale=10.0, cmr_pool_lr_scale=20.0, cmr_proj_lr_scale=20.0,
    )
    for g in groups:
        g['lr'] = args.base_lr * g.get('lr_scale', 1.0)
        g.setdefault('lr_scale', 1.0)
    opt = torch.optim.AdamW(groups, lr=args.base_lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    def _set_lr(ep_f):
        if args.lr_schedule == 'const':
            factor = 1.0
        elif ep_f < args.warmup_epochs:
            factor = ep_f / max(args.warmup_epochs, 1e-8)
        else:
            progress = (ep_f - args.warmup_epochs) / max(args.epochs - args.warmup_epochs, 1e-8)
            progress = min(max(progress, 0.0), 1.0)
            cos = 0.5 * (1.0 + math.cos(math.pi * progress))
            min_ratio = args.min_lr / args.base_lr
            factor = min_ratio + (1.0 - min_ratio) * cos
        for g in opt.param_groups:
            g['lr'] = args.base_lr * g['lr_scale'] * factor
        return factor

    history = []
    best_val_r1 = -1.0
    best_epoch = -1
    t_start = time.time()

    print(f'[tier1] base_lr={args.base_lr} temp={args.temperature} bs={args.batch_size} '
          f'ep={args.epochs} amp={args.amp} sched={args.lr_schedule} warmup={args.warmup_epochs}')

    # Baseline eval
    m_val0 = eval_retrieval(model, ds_val, device)
    print('[tier1] epoch 0 [baseline]')
    print('        ' + _fmt(m_val0, 'val'))
    history.append({'epoch': 0, 'phase': 'eval', 'val_metrics': m_val0})

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        ep_losses, ep_grad = [], []

        for it, (img, cmr, _) in enumerate(loader):
            ep_f = (epoch - 1) + it / max(n_iters_per_epoch, 1)
            lr_factor = _set_lr(ep_f)

            img = img.to(device, non_blocking=True)
            cmr = cmr.to(device, non_blocking=True)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                zf, zc = model(img, cmr)
                loss_fc = hard_infonce_loss(zf, zc, temperature=args.temperature)
                loss_cf = hard_infonce_loss(zc, zf, temperature=args.temperature)
                loss = 0.5 * (loss_fc + loss_cf)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = _grad_norm([p for p in model.parameters() if p.requires_grad])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()

            ep_losses.append(loss.item())
            ep_grad.append(gn)

        mean_loss = float(np.mean(ep_losses))
        mean_gn = float(np.mean(ep_grad))
        elapsed = time.time() - t_start
        print(f'[tier1] ep {epoch:3d}/{args.epochs} loss={mean_loss:.4f} '
              f'grad={mean_gn:.2f} lr_f={lr_factor:.3f} elapsed={elapsed/60:.1f}min')
        history.append({'epoch': epoch, 'phase': 'train', 'loss': mean_loss,
                        'grad_norm': mean_gn, 'lr_factor': lr_factor})

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m_val = eval_retrieval(model, ds_val, device)
            print('        ' + _fmt(m_val, 'val'))
            log_entry = {'epoch': epoch, 'phase': 'eval', 'val_metrics': m_val}

            if args.train_eval_every and epoch % args.train_eval_every == 0:
                m_tr = eval_retrieval(model, ds_train_eval, device)
                print('        ' + _fmt(m_tr, 'train'))
                log_entry['train_metrics'] = m_tr

            history.append(log_entry)

            # checkpoint best
            v_r1 = m_val.get('R@1_avg', 0.0)
            if v_r1 > best_val_r1:
                best_val_r1 = v_r1
                best_epoch = epoch
                save_encoder_checkpoint(os.path.join(args.out_dir, 'checkpoint_best.pth'),
                                        model, args, m_val)
                print(f'        [best] val R@1_avg={v_r1:.3f} saved at ep{epoch}')

            with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

    # Final: eval test set
    print(f'\n[tier1] final test eval (best checkpoint from ep{best_epoch})')
    # Load best checkpoint weights
    ck = torch.load(os.path.join(args.out_dir, 'checkpoint_best.pth'), map_location=device)
    model.fundus.load_state_dict(ck['fundus_model'])
    model.cmr.load_state_dict(ck['cmr_model'])
    m_test = eval_retrieval(model, ds_test, device)
    m_val_best = eval_retrieval(model, ds_val, device)  # sanity confirm
    print('        ' + _fmt(m_val_best, 'val(best)'))
    print('        ' + _fmt(m_test, 'test'))
    history.append({'epoch': 'final', 'phase': 'final_eval',
                    'best_epoch': best_epoch,
                    'val_metrics': m_val_best, 'test_metrics': m_test})

    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[tier1] done. total {(time.time()-t_start)/60:.1f} min')
    print(f'[tier1] best val R@1={best_val_r1:.3f} at ep{best_epoch}')
    print(f'[tier1] test R@1={m_test["R@1_avg"]:.3f} '
          f'R@5={m_test["R@5_avg"]:.3f} R@10={m_test["R@10_avg"]:.3f}')


if __name__ == '__main__':
    main()
