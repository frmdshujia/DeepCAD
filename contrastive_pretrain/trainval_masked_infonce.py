"""trainval_masked_infonce.py -- Masked InfoNCE training (Stage 2, hot-start from Tier 2).

Key differences vs trainval_tier2.py:
  - Loss: masked_infonce_loss (PC14-guided soft positives/negatives)
  - PC14 vectors loaded from cmr_table.csv into memory; EIDs without PC14 are
    filtered out of the train split (typically ~13% loss).
  - Val / test eval includes Spearman(cosine_sim, -PC14_dist) in addition to R@k.
  - --resume: hot-start from an existing checkpoint (e.g. Tier 2 best ckpt).

Typical launch:
    CUDA_VISIBLE_DEVICES=0 python contrastive_pretrain/trainval_masked_infonce.py \\
        --resume output_dir/stage2_tier2_unfrozen/checkpoint_best.pth \\
        --epochs 30 --base_lr 2e-5 \\
        --pc_mask_k_pos 2 --pc_mask_k_neg 4 \\
        --out_dir output_dir/stage2_masked_infonce_v1
"""
from __future__ import annotations
import sys, os, time, argparse, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, Sampler

from datasets_stage2 import FundusCMRImageDataset
from models_stage2 import Stage2DualTower, build_stage2_param_groups
from loss_masked_infonce import masked_infonce_loss

try:
    from scipy.stats import spearmanr as _spearmanr
    _SCIPY = True
except ImportError:
    _SCIPY = False
    print('[warn] scipy not available — Spearman eval disabled')


# ---------------------------------------------------------------------------
# PC14 helpers
# ---------------------------------------------------------------------------

def load_eid2pc(cmr_csv: str, pc_cols_txt: str) -> dict:
    """Return {eid(int): np.array (14,) float32} for EIDs with complete PC14."""
    pc_cols_raw = open(pc_cols_txt).read().strip()
    pc_cols = [c.strip() for c in pc_cols_raw.replace('\n', ',').split(',') if c.strip()]
    cmr = pd.read_csv(cmr_csv)
    cmr = cmr[cmr['instance'] == 2]
    result = {}
    for _, row in cmr.iterrows():
        vals = row[pc_cols].values
        if not np.any(np.isnan(vals.astype(float))):
            result[int(row['eid'])] = vals.astype(np.float32)
    print(f'[PC14] loaded {len(result)} EIDs with complete PC14 ({len(pc_cols)} dims)')
    return result


def eids_with_pc14(eids: list, eid2pc: dict) -> list:
    return [e for e in eids if int(e) in eid2pc]


# ---------------------------------------------------------------------------
# Sampler: one random fundus per unique EID per epoch
# ---------------------------------------------------------------------------

class UniqueEIDSampler(Sampler):
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


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_retrieval(model, dataset, device, batch_size: int = 16,
                   num_workers: int = 3, max_gallery: int = 0):
    """R@1/5/10 over unique-EID representatives. Returns dict."""
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

    if max_gallery and len(idx_list) > max_gallery:
        rng = np.random.default_rng(1234)
        idx_list = sorted(rng.choice(idx_list, size=max_gallery, replace=False).tolist())

    loader = DataLoader(Subset(dataset, idx_list), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)

    zfs, zcs, batch_eids = [], [], []
    for img, cmr, eids in loader:
        img = img.to(device, non_blocking=True)
        cmr = cmr.to(device, non_blocking=True)
        zf, zc = model(img, cmr)
        zfs.append(zf); zcs.append(zc)
        batch_eids.extend([int(e) for e in eids])

    zf = torch.cat(zfs, 0)
    zc = torch.cat(zcs, 0)
    N = zf.shape[0]
    sim = zf @ zc.T

    ranks_fc = sim.argsort(dim=1, descending=True)
    correct_fc = (ranks_fc == torch.arange(N, device=device).unsqueeze(1))
    ranks_cf = sim.T.argsort(dim=1, descending=True)
    correct_cf = (ranks_cf == torch.arange(N, device=device).unsqueeze(1))

    out = {'N': N, 'eids': batch_eids}
    for k in (1, 5, 10):
        k_eff = min(k, N)
        r_fc = correct_fc[:, :k_eff].any(dim=1).float().mean().item()
        r_cf = correct_cf[:, :k_eff].any(dim=1).float().mean().item()
        out[f'R@{k}_f2c'] = r_fc
        out[f'R@{k}_c2f'] = r_cf
        out[f'R@{k}_avg'] = (r_fc + r_cf) / 2

    out['_zf'] = zf
    out['_zc'] = zc
    model.train()
    return out


def compute_spearman(m: dict, eid2pc: dict) -> float | None:
    """Spearman(cosine_sim_ij, -PC14_dist_ij) over all off-diagonal pairs
    where both EIDs have PC14. Returns None if scipy unavailable or too few pairs."""
    if not _SCIPY:
        return None
    eids = m.get('eids', [])
    zf = m.get('_zf')
    zc = m.get('_zc')
    if zf is None or len(eids) == 0:
        return None

    # Filter to EIDs with PC14
    valid_idx = [i for i, e in enumerate(eids) if e in eid2pc]
    if len(valid_idx) < 5:
        return None

    idx_t = torch.tensor(valid_idx, device=zf.device)
    zf_v = zf[idx_t]
    zc_v = zc[idx_t]
    pc14_v = torch.tensor(
        np.stack([eid2pc[eids[i]] for i in valid_idx]), device=zf.device)

    sim_mat = (zf_v @ zc_v.T).cpu().numpy()        # (M, M)
    dist_mat = torch.cdist(pc14_v.float(), pc14_v.float(), p=2).cpu().numpy()

    M = len(valid_idx)
    # Upper triangle, off-diagonal
    rows, cols = np.triu_indices(M, k=1)
    sim_flat = sim_mat[rows, cols]
    neg_dist_flat = -dist_mat[rows, cols]

    rho, pval = _spearmanr(sim_flat, neg_dist_flat)
    return float(rho)


def _fmt(m: dict, tag: str, spearman: float | None = None) -> str:
    if m.get('N', 0) == 0:
        return f'{tag}: N=0'
    s = (f"{tag} N={m['N']} "
         f"R@1={m['R@1_avg']:.4f} R@5={m['R@5_avg']:.4f} R@10={m['R@10_avg']:.4f} "
         f"(f2c R@1={m['R@1_f2c']:.4f}, c2f R@1={m['R@1_c2f']:.4f})")
    if spearman is not None:
        s += f'  Spearman={spearman:+.4f}'
    return s


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(path, model, args, val_metrics, spearman, epoch):
    torch.save({
        'epoch': epoch,
        'fundus_model': model.fundus.state_dict(),
        'cmr_model': model.cmr.state_dict(),
        'val_metrics': val_metrics,
        'val_spearman': spearman,
        'args': {
            'proj_dim': 256,
            'num_frames': 4,
            'drop_path': 0.0,
            'freeze_backbones': args.freeze_backbones,
            'base_lr': args.base_lr,
            'temperature': args.temperature,
            'batch_size': args.batch_size,
            'epochs': args.epochs,
            'k_pos': args.pc_mask_k_pos,
            'k_neg': args.pc_mask_k_neg,
        },
    }, path)


def load_resume(path, model, device):
    ck = torch.load(path, map_location=device)
    model.fundus.load_state_dict(ck['fundus_model'])
    model.cmr.load_state_dict(ck['cmr_model'])
    print(f'[resume] loaded fundus+cmr weights from {path}')
    if 'val_metrics' in ck:
        vm = ck['val_metrics']
        ep = ck.get('epoch', '?')
        print(f'[resume] ckpt epoch={ep}  '
              f"val R@1={vm.get('R@1_avg', vm.get('R@1', '?')):.4f}")


def read_eids(p: str):
    ids = []
    with open(p) as f:
        for line in f:
            s = line.strip()
            if s and not s.startswith('#'):
                ids.append(int(s))
    return ids


def _grad_norm(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            total += p.grad.detach().float().norm().item() ** 2
    return math.sqrt(total)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    # Data
    ap.add_argument('--csv', type=str,
                    default='contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv')
    ap.add_argument('--cmr_dir', type=str,
                    default='/data/home/shujia/UKB/CMRI/preprocessed_lax_sax')
    ap.add_argument('--cmr_table', type=str,
                    default='contrastive_pretrain/preprocessed_data/modeling_delivery/cmr_table.csv')
    ap.add_argument('--pc_cols_txt', type=str,
                    default='contrastive_pretrain/preprocessed_data/modeling_delivery/pc_columns.txt')
    ap.add_argument('--train_eids_file', type=str,
                    default='contrastive_pretrain/stage2_tier2_5k_eids.txt')
    # Weights
    ap.add_argument('--medsam_ckpt', type=str,
                    default='pretrained_weights/hf_cache/models--flaviagiammarino--medsam-vit-base/'
                            'snapshots/a3cb4c518fc3ae8beaf1af0cd3a43867cfa28335/pytorch_model.bin')
    ap.add_argument('--retfound_ckpt', type=str, default='RETFound_cfp_weights.pth')
    ap.add_argument('--resume', type=str, default='',
                    help='Hot-start from checkpoint (e.g. Tier 2 best ckpt).')
    # Output
    ap.add_argument('--out_dir', type=str, required=True)
    # Training
    ap.add_argument('--batch_size', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--base_lr', type=float, default=2e-5)
    ap.add_argument('--temperature', type=float, default=0.07)
    ap.add_argument('--weight_decay', type=float, default=0.05)
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--lr_schedule', choices=['const', 'cosine'], default='cosine')
    ap.add_argument('--warmup_epochs', type=int, default=3)
    ap.add_argument('--min_lr', type=float, default=1e-7)
    ap.add_argument('--freeze_backbones', action='store_true')
    ap.add_argument('--num_workers', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    # Loss
    ap.add_argument('--pc_mask_k_pos', type=int, default=2,
                    help='Extra PC14-nearest positives per anchor (beyond diagonal).')
    ap.add_argument('--pc_mask_k_neg', type=int, default=4,
                    help='PC14-farthest negatives per anchor.')
    # Eval
    ap.add_argument('--eval_every', type=int, default=2)
    ap.add_argument('--train_eval_every', type=int, default=6,
                    help='0 to disable train-set retrieval eval.')
    ap.add_argument('--val_max_gallery', type=int, default=0,
                    help='0 = full gallery. >0 subsamples for speed.')
    ap.add_argument('--test_max_gallery', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[masked_infonce] device={device}  out_dir={args.out_dir}')
    print(f'[masked_infonce] k_pos={args.pc_mask_k_pos}  k_neg={args.pc_mask_k_neg}  '
          f'temp={args.temperature}  bs={args.batch_size}  base_lr={args.base_lr}')

    # ---- Load PC14 lookup ----
    eid2pc = load_eid2pc(args.cmr_table, args.pc_cols_txt)

    # ---- Train EID allowlist ----
    train_eids = None
    if args.train_eids_file and os.path.isfile(args.train_eids_file):
        train_eids = read_eids(args.train_eids_file)
        # Keep only EIDs that have PC14 (required for masked loss)
        train_eids_pc14 = [e for e in train_eids if e in eid2pc]
        print(f'[masked_infonce] train EIDs: {len(train_eids)} total, '
              f'{len(train_eids_pc14)} with PC14 ({100*len(train_eids_pc14)/max(len(train_eids),1):.1f}%)')
        train_eids = train_eids_pc14
    else:
        print('[masked_infonce] no train_eids_file; will filter by PC14 availability')

    # ---- Datasets ----
    ds_train = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='train', num_frames=4,
        use_eval_transform=True,        # no augment (see Tier 1 collapse note)
        eid_allowlist=train_eids,
    )
    ds_val = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='val', num_frames=4, use_eval_transform=True,
    )
    ds_test = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='test', num_frames=4, use_eval_transform=True,
    )
    print(f'[masked_infonce] train: {len(ds_train)} pairs, '
          f'{len(set(ds_train.eids))} unique EIDs')
    print(f'[masked_infonce] val  : {len(ds_val)} pairs, '
          f'{len(set(ds_val.eids))} unique EIDs')
    print(f'[masked_infonce] test : {len(ds_test)} pairs, '
          f'{len(set(ds_test.eids))} unique EIDs')

    sampler = UniqueEIDSampler(ds_train.eids, seed=args.seed)
    loader = DataLoader(ds_train, batch_size=args.batch_size, sampler=sampler,
                        num_workers=args.num_workers, drop_last=True, pin_memory=True)
    n_iters = len(loader)
    print(f'[masked_infonce] iters/epoch = {n_iters}')

    # Eval-transform copy for train-subset retrieval eval
    ds_train_eval = FundusCMRImageDataset(
        args.csv, args.cmr_dir, split='train', num_frames=4,
        use_eval_transform=True, eid_allowlist=train_eids,
    )

    # ---- Model ----
    model = Stage2DualTower(
        proj_dim=256, num_frames=4, drop_path_rate=0.0,
        medsam_ckpt=args.medsam_ckpt, retfound_ckpt=args.retfound_ckpt,
        freeze_fundus_backbone=args.freeze_backbones,
        freeze_cmr_backbone=args.freeze_backbones,
    ).to(device)

    if args.resume:
        load_resume(args.resume, model, device)

    n_tot = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[masked_infonce] params: total={n_tot/1e6:.1f}M  trainable={n_tr/1e6:.2f}M')

    # ---- Optimizer ----
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

    # ---- Training loop ----
    history = []
    best_metric = -float('inf')   # primary: Spearman; fallback: R@1
    best_epoch = -1
    t_start = time.time()

    # Baseline eval before training
    m_val0 = eval_retrieval(model, ds_val, device,
                            num_workers=args.num_workers,
                            max_gallery=args.val_max_gallery)
    sp0 = compute_spearman(m_val0, eid2pc)
    print(f'[masked_infonce] epoch 0 [baseline]')
    print('        ' + _fmt(m_val0, 'val', sp0))
    history.append({'epoch': 0, 'phase': 'eval',
                    'val_metrics': {k: v for k, v in m_val0.items() if not k.startswith('_')},
                    'val_spearman': sp0})

    for epoch in range(1, args.epochs + 1):
        sampler.set_epoch(epoch)
        model.train()
        ep_losses, ep_grads = [], []

        for it, (img, cmr, eids) in enumerate(loader):
            ep_f = (epoch - 1) + it / max(n_iters, 1)
            lr_factor = _set_lr(ep_f)

            img = img.to(device, non_blocking=True)
            cmr = cmr.to(device, non_blocking=True)

            # PC14 lookup for this batch (all EIDs guaranteed to have PC14)
            pc14 = torch.tensor(
                np.stack([eid2pc[int(e)] for e in eids]),
                dtype=torch.float32, device=device,
            )  # (B, 14)

            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                zf, zc = model(img, cmr)
                loss = masked_infonce_loss(
                    zf, zc, pc14,
                    temperature=args.temperature,
                    k_pos=args.pc_mask_k_pos,
                    k_neg=args.pc_mask_k_neg,
                )

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = _grad_norm([p for p in model.parameters() if p.requires_grad])
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(opt)
            scaler.update()

            ep_losses.append(loss.item())
            ep_grads.append(gn)

        mean_loss = float(np.mean(ep_losses))
        mean_gn = float(np.mean(ep_grads))
        elapsed = time.time() - t_start
        print(f'[masked_infonce] ep {epoch:3d}/{args.epochs}  loss={mean_loss:.4f}  '
              f'grad={mean_gn:.2f}  lr_f={lr_factor:.3f}  elapsed={elapsed/60:.1f}min',
              flush=True)
        history.append({'epoch': epoch, 'phase': 'train', 'loss': mean_loss,
                        'grad_norm': mean_gn, 'lr_factor': lr_factor})

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            m_val = eval_retrieval(model, ds_val, device,
                                   num_workers=args.num_workers,
                                   max_gallery=args.val_max_gallery)
            sp = compute_spearman(m_val, eid2pc)
            print('        ' + _fmt(m_val, 'val', sp), flush=True)

            log_entry = {'epoch': epoch, 'phase': 'eval',
                         'val_metrics': {k: v for k, v in m_val.items()
                                         if not k.startswith('_')},
                         'val_spearman': sp}

            if args.train_eval_every and epoch % args.train_eval_every == 0:
                m_tr = eval_retrieval(model, ds_train_eval, device,
                                      num_workers=args.num_workers, max_gallery=500)
                sp_tr = compute_spearman(m_tr, eid2pc)
                print('        ' + _fmt(m_tr, 'train(500)', sp_tr), flush=True)
                log_entry['train_metrics'] = {k: v for k, v in m_tr.items()
                                              if not k.startswith('_')}
                log_entry['train_spearman'] = sp_tr

            history.append(log_entry)

            # Primary metric: Spearman (phenotype signal); fallback to R@1
            primary = sp if sp is not None else m_val.get('R@1_avg', 0.0)
            if primary > best_metric:
                best_metric = primary
                best_epoch = epoch
                save_checkpoint(
                    os.path.join(args.out_dir, 'checkpoint_best.pth'),
                    model, args,
                    {k: v for k, v in m_val.items() if not k.startswith('_')},
                    sp, epoch,
                )
                label = 'Spearman' if sp is not None else 'R@1'
                print(f'        [best] val {label}={primary:.4f}  saved at ep{epoch}',
                      flush=True)

            with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
                json.dump(history, f, indent=2)

    # ---- Final test eval from best checkpoint ----
    print(f'\n[masked_infonce] final test eval (best checkpoint ep{best_epoch})',
          flush=True)
    ck = torch.load(os.path.join(args.out_dir, 'checkpoint_best.pth'), map_location=device)
    model.fundus.load_state_dict(ck['fundus_model'])
    model.cmr.load_state_dict(ck['cmr_model'])

    m_test = eval_retrieval(model, ds_test, device,
                            num_workers=args.num_workers,
                            max_gallery=args.test_max_gallery)
    m_val_best = eval_retrieval(model, ds_val, device,
                                num_workers=args.num_workers,
                                max_gallery=args.val_max_gallery)
    sp_test = compute_spearman(m_test, eid2pc)
    sp_val_best = compute_spearman(m_val_best, eid2pc)

    print('        ' + _fmt(m_val_best, 'val(best)', sp_val_best), flush=True)
    print('        ' + _fmt(m_test, 'test', sp_test), flush=True)

    history.append({
        'epoch': 'final', 'phase': 'final_eval',
        'best_epoch': best_epoch,
        'val_metrics': {k: v for k, v in m_val_best.items() if not k.startswith('_')},
        'val_spearman': sp_val_best,
        'test_metrics': {k: v for k, v in m_test.items() if not k.startswith('_')},
        'test_spearman': sp_test,
    })
    with open(os.path.join(args.out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    total_min = (time.time() - t_start) / 60
    print(f'[masked_infonce] done. total {total_min:.1f} min')
    print(f'[masked_infonce] best ep={best_epoch}  primary_metric={best_metric:.4f}')
    print(f'[masked_infonce] test  R@1={m_test["R@1_avg"]:.4f}  '
          f'R@10={m_test["R@10_avg"]:.4f}  Spearman={sp_test}')


if __name__ == '__main__':
    main()
