"""overfit_tier0.py -- Stage 2 pipeline sanity test on the 100-EID Tier 0 cohort.

Goal: prove that the dual-tower + InfoNCE setup has enough capacity to fit a
small cohort. Success criteria (by epoch 30-50):
  - loss drops from ~ln(B) to < 0.5
  - train-set Recall@1 on unique-EID retrieval jumps from ~1/N to > 80%

If this fails, the problem is architectural or the loss / param-group config.
If this passes, we can proceed to Tier 1 (1K EID) for generalization checks.
"""
from __future__ import annotations
import sys, os, time, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from datasets_stage2 import FundusCMRImageDataset
from datasets_contrast import UniqueEIDSampler
from models_stage2 import Stage2DualTower, build_stage2_param_groups
from loss_contrast import hard_infonce_loss


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return math.sqrt(total)


@torch.no_grad()
def eval_retrieval(model, dataset, device, batch_size: int = 8):
    """Compute fundus->CMR and CMR->fundus retrieval recall on unique-EID
    representatives from the dataset (one fundus image per EID, one CMR per EID).
    Returns dict of R@1/R@5/R@10 (symmetric average)."""
    model.eval()
    # Pick first image per unique EID
    seen = set()
    idx_list = []
    for i, e in enumerate(dataset.eids):
        if e not in seen:
            seen.add(e)
            idx_list.append(i)
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
    sim = zf @ zc.T  # (N, N)  already L2 normalized

    # fundus -> cmr
    ranks_fc = sim.argsort(dim=1, descending=True)
    correct_fc = (ranks_fc == torch.arange(N, device=device).unsqueeze(1))
    # cmr -> fundus
    ranks_cf = sim.T.argsort(dim=1, descending=True)
    correct_cf = (ranks_cf == torch.arange(N, device=device).unsqueeze(1))

    out = {'N': int(N)}
    for k in (1, 5, 10):
        r_fc = correct_fc[:, :k].any(dim=1).float().mean().item()
        r_cf = correct_cf[:, :k].any(dim=1).float().mean().item()
        out[f'R@{k}_f2c'] = r_fc
        out[f'R@{k}_c2f'] = r_cf
        out[f'R@{k}_avg'] = (r_fc + r_cf) / 2
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', type=str,
                    default='contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table.csv')
    ap.add_argument('--cmr_dir', type=str,
                    default='/data/home/shujia/UKB/CMRI/preprocessed_lax_sax')
    ap.add_argument('--medsam_ckpt', type=str,
                    default='pretrained_weights/hf_cache/models--flaviagiammarino--medsam-vit-base/snapshots/a3cb4c518fc3ae8beaf1af0cd3a43867cfa28335/pytorch_model.bin')
    ap.add_argument('--retfound_ckpt', type=str, default='RETFound_cfp_weights.pth')
    ap.add_argument('--out_dir', type=str, default='output_dir/stage2_overfit_tier0')
    ap.add_argument('--batch_size', type=int, default=8)
    ap.add_argument('--epochs', type=int, default=50)
    ap.add_argument('--base_lr', type=float, default=5e-5)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--eval_every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--amp', action='store_true', help='enable mixed precision (fp16)')
    ap.add_argument('--lr_schedule', choices=['const', 'cosine'], default='const')
    ap.add_argument('--warmup_epochs', type=int, default=0,
                    help='linear warmup from 0 to base_lr over this many epochs (cosine only)')
    ap.add_argument('--min_lr', type=float, default=1e-7,
                    help='final cosine floor')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[overfit] device={device}  out_dir={args.out_dir}')

    # Dataset: split='train', EVAL transform (no augmentation) so we truly
    # test capacity-to-fit rather than invariance under augment.
    ds_train = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='train', num_frames=4,
        use_eval_transform=True,
    )
    print(f'[overfit] dataset: {len(ds_train)} pairs, '
          f'{len(set(ds_train.eids))} unique EIDs')

    sampler = UniqueEIDSampler(ds_train, seed=args.seed)
    loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                        num_workers=3, drop_last=True, pin_memory=True)
    n_iters_per_epoch = len(loader)
    print(f'[overfit] iters/epoch = {n_iters_per_epoch}')

    # Model
    model = Stage2DualTower(
        proj_dim=256, num_frames=4, drop_path_rate=0.0,
        medsam_ckpt=args.medsam_ckpt, retfound_ckpt=args.retfound_ckpt,
    ).to(device)

    groups = build_stage2_param_groups(
        model,
        weight_decay=args.weight_decay,
        fundus_layer_decay=0.75, cmr_layer_decay=0.80,
        fundus_proj_lr_scale=10.0, cmr_pool_lr_scale=20.0, cmr_proj_lr_scale=20.0,
    )
    for g in groups:
        g['lr'] = args.base_lr * g.get('lr_scale', 1.0)
        g.setdefault('lr_scale', 1.0)
    opt = torch.optim.AdamW(groups, lr=args.base_lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    def _set_lr(ep_f: float):
        """Apply cosine (with warmup) or constant LR at a fractional epoch."""
        if args.lr_schedule == 'const':
            factor = 1.0
        elif ep_f < args.warmup_epochs:
            factor = ep_f / max(args.warmup_epochs, 1e-8)
        else:
            # cosine from 1.0 -> min_lr/base_lr
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
    print(f'[overfit] starting: base_lr={args.base_lr}, temperature={args.temperature}, '
          f'batch={args.batch_size}, epochs={args.epochs}, '
          f'amp={args.amp}, lr_schedule={args.lr_schedule}, warmup={args.warmup_epochs}')

    # Eval at epoch 0 (baseline)
    met0 = eval_retrieval(model, ds_train, device)
    print(f'[overfit] epoch 0 [baseline] N={met0["N"]} '
          f"R@1_avg={met0['R@1_avg']:.3f} R@5_avg={met0['R@5_avg']:.3f} "
          f"R@10_avg={met0['R@10_avg']:.3f}")
    history.append({'epoch': 0, 'phase': 'eval', **met0})

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        ep_losses = []
        ep_grad = []
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
        print(f'[overfit] ep {epoch:3d}/{args.epochs}  loss={mean_loss:.4f}  '
              f'grad_norm={mean_gn:.2f}  lr_factor={lr_factor:.3f}  elapsed={elapsed/60:.1f}min')
        history.append({'epoch': epoch, 'phase': 'train', 'loss': mean_loss,
                        'grad_norm': mean_gn, 'lr_factor': lr_factor})

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            met = eval_retrieval(model, ds_train, device)
            print(f'           [eval] R@1_avg={met["R@1_avg"]:.3f}  '
                  f'R@5_avg={met["R@5_avg"]:.3f}  R@10_avg={met["R@10_avg"]:.3f}  '
                  f'(f2c R@1={met["R@1_f2c"]:.3f}, c2f R@1={met["R@1_c2f"]:.3f})')
            history.append({'epoch': epoch, 'phase': 'eval', **met})

    # Save history
    hist_path = os.path.join(args.out_dir, 'history.json')
    with open(hist_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f'[overfit] saved history to {hist_path}')
    print(f'[overfit] done. total {(time.time()-t_start)/60:.1f} min')


if __name__ == '__main__':
    main()
