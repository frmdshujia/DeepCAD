"""trainval_tier0.py -- Stage 2 Tier 0 generalization smoke (80/20 EID split).

Based on overfit_tier0.py but:
  - Splits the Tier 0 cohort into train_eids / val_eids (by unique EID) using
    a fixed seed. Default 80/20.
  - Trains only on train_eids.
  - Each eval computes retrieval on BOTH train_eids and val_eids separately,
    so we can watch generalization (val R@k) vs capacity (train R@k).

Goal: if val R@k clearly rises above random chance while train still saturates,
      the representation generalizes and we can proceed to Tier 1.
      If val stays at chance while train rockets to 100%, it's pure memorization
      and we need to revisit regularization / augmentation before scaling up.

Random-chance baselines (N_val ~ 16):
  R@1 ~ 6.25%, R@5 ~ 31.25%, R@10 ~ 62.5%
"""
from __future__ import annotations
import sys, os, time, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
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


class EIDRestrictedUniqueSampler(Sampler):
    """Like UniqueEIDSampler but restricted to a given set of allowed EIDs.
    Emits exactly one index per allowed EID per epoch, randomly choosing among
    available fundus images for that EID."""

    def __init__(self, dataset_eids, allowed_eids, seed: int = 0):
        self.seed = seed
        self.epoch = 0
        allowed = set(int(e) for e in allowed_eids)
        eid_to_indices: dict = {}
        for idx, e in enumerate(dataset_eids):
            if int(e) in allowed:
                eid_to_indices.setdefault(int(e), []).append(idx)
        self.eid_to_indices = eid_to_indices
        self.unique_eids = list(eid_to_indices.keys())

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
def eval_retrieval_subset(model, dataset, allowed_eids, device, batch_size: int = 8):
    """Retrieval R@k on a specific EID subset. Picks one fundus/CMR pair per
    unique EID from the dataset whose eid is in `allowed_eids`."""
    model.eval()
    allowed = set(int(e) for e in allowed_eids)
    seen = set()
    idx_list = []
    for i, e in enumerate(dataset.eids):
        e = int(e)
        if e in allowed and e not in seen:
            seen.add(e)
            idx_list.append(i)
    if len(idx_list) == 0:
        model.train()
        return {'N': 0}

    subset = Subset(dataset, idx_list)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=2)

    zfs, zcs = [], []
    for img, cmr, _ in loader:
        img = img.to(device); cmr = cmr.to(device)
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


def _fmt_metrics(m, tag):
    if m.get('N', 0) == 0:
        return f'{tag}: N=0 (empty)'
    return (f"{tag} N={m['N']} "
            f"R@1={m['R@1_avg']:.3f} R@5={m['R@5_avg']:.3f} R@10={m['R@10_avg']:.3f} "
            f"(f2c R@1={m['R@1_f2c']:.3f}, c2f R@1={m['R@1_c2f']:.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str,
                    default='contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table.csv')
    ap.add_argument('--cmr_dir', type=str,
                    default='/data/home/shujia/UKB/CMRI/preprocessed_lax_sax')
    ap.add_argument('--medsam_ckpt', type=str,
                    default='pretrained_weights/hf_cache/models--flaviagiammarino--medsam-vit-base/snapshots/a3cb4c518fc3ae8beaf1af0cd3a43867cfa28335/pytorch_model.bin')
    ap.add_argument('--retfound_ckpt', type=str, default='RETFound_cfp_weights.pth')
    ap.add_argument('--out_dir', type=str, default='output_dir/stage2_trainval_tier0')
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--base_lr', type=float, default=5e-5)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--split_seed', type=int, default=42,
                    help='seed specifically used for the train/val EID split')
    ap.add_argument('--val_eid_frac', type=float, default=0.20,
                    help='fraction of unique EIDs to reserve for val retrieval')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--lr_schedule', choices=['const', 'cosine'], default='cosine')
    ap.add_argument('--warmup_epochs', type=int, default=10)
    ap.add_argument('--min_lr', type=float, default=1e-7)
    ap.add_argument('--freeze_backbones', action='store_true',
                    help='Freeze RETFound + MedSAM backbones; only train FrameAttnPool '
                         'and projection heads (much lower capacity).')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[trainval] device={device}  out_dir={args.out_dir}')

    # Dataset with eval transform so augmentation doesn't muddy the comparison.
    ds = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='train', num_frames=4,
        use_eval_transform=True,
    )
    unique_eids = sorted(set(int(e) for e in ds.eids))
    n_unique = len(unique_eids)
    print(f'[trainval] dataset: {len(ds)} pairs, {n_unique} unique EIDs')

    # Deterministic 80/20 split by EID.
    rng = np.random.default_rng(args.split_seed)
    perm = rng.permutation(n_unique)
    n_val = max(1, int(round(n_unique * args.val_eid_frac)))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    train_eids = sorted(int(unique_eids[i]) for i in train_idx)
    val_eids = sorted(int(unique_eids[i]) for i in val_idx)
    print(f'[trainval] split (seed={args.split_seed}): '
          f'{len(train_eids)} train EIDs / {len(val_eids)} val EIDs')
    print(f'[trainval] val_eids = {val_eids}')

    # Dump split for reproducibility.
    with open(os.path.join(args.out_dir, 'split.json'), 'w') as f:
        json.dump({'split_seed': args.split_seed,
                   'val_eid_frac': args.val_eid_frac,
                   'train_eids': train_eids,
                   'val_eids': val_eids}, f, indent=2)

    sampler = EIDRestrictedUniqueSampler(ds.eids, train_eids, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                        num_workers=3, drop_last=True, pin_memory=True)
    n_iters_per_epoch = len(loader)
    print(f'[trainval] iters/epoch = {n_iters_per_epoch}')

    # Model
    model = Stage2DualTower(
        proj_dim=256, num_frames=4, drop_path_rate=0.0,
        medsam_ckpt=args.medsam_ckpt, retfound_ckpt=args.retfound_ckpt,
        freeze_fundus_backbone=args.freeze_backbones,
        freeze_cmr_backbone=args.freeze_backbones,
    ).to(device)
    n_total = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[trainval] params: total={n_total/1e6:.1f}M  trainable={n_trainable/1e6:.2f}M '
          f'(freeze_backbones={args.freeze_backbones})')

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

    def _set_lr(ep_f: float):
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
    t_start = time.time()
    print(f'[trainval] base_lr={args.base_lr} temperature={args.temperature} '
          f'batch={args.batch_size} epochs={args.epochs} amp={args.amp} '
          f'lr_schedule={args.lr_schedule} warmup={args.warmup_epochs}')

    # Baseline eval at epoch 0
    m_tr0 = eval_retrieval_subset(model, ds, train_eids, device)
    m_va0 = eval_retrieval_subset(model, ds, val_eids, device)
    print(f'[trainval] epoch 0 [baseline]')
    print(f'           ' + _fmt_metrics(m_tr0, 'train'))
    print(f'           ' + _fmt_metrics(m_va0, '  val'))
    history.append({'epoch': 0, 'phase': 'eval',
                    'train_metrics': m_tr0, 'val_metrics': m_va0})

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
        print(f'[trainval] ep {epoch:3d}/{args.epochs} loss={mean_loss:.4f} '
              f'grad_norm={mean_gn:.2f} lr_factor={lr_factor:.3f} '
              f'elapsed={elapsed/60:.1f}min')
        history.append({'epoch': epoch, 'phase': 'train', 'loss': mean_loss,
                        'grad_norm': mean_gn, 'lr_factor': lr_factor})

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m_tr = eval_retrieval_subset(model, ds, train_eids, device)
            m_va = eval_retrieval_subset(model, ds, val_eids, device)
            print(f'           ' + _fmt_metrics(m_tr, 'train'))
            print(f'           ' + _fmt_metrics(m_va, '  val'))
            history.append({'epoch': epoch, 'phase': 'eval',
                            'train_metrics': m_tr, 'val_metrics': m_va})

            hist_path = os.path.join(args.out_dir, 'history.json')
            with open(hist_path, 'w') as f:
                json.dump(history, f, indent=2)

    hist_path = os.path.join(args.out_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[trainval] saved history to {hist_path}')
    print(f'[trainval] done. total {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
