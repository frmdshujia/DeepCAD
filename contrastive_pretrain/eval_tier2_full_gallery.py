"""eval_tier2_full_gallery.py
Run retrieval eval + Spearman correlation on the Tier 2 best checkpoint
using the full val/test gallery from fundus_table_extended.csv.

Usage:
    python contrastive_pretrain/eval_tier2_full_gallery.py \
        --ckpt output_dir/stage2_tier2_unfrozen/checkpoint_best.pth \
        --device cuda:0
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from scipy.stats import spearmanr

from contrastive_pretrain.datasets_stage2 import FundusCMRImageDataset
from contrastive_pretrain.models_stage2 import Stage2DualTower

FUNDUS_TABLE = 'contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv'
CMR_NPY_DIR  = '/data/home/shujia/UKB/CMRI/preprocessed_lax_sax'
CMR_TABLE    = 'contrastive_pretrain/preprocessed_data/modeling_delivery/cmr_table.csv'
PC_COLS_FILE = 'contrastive_pretrain/preprocessed_data/modeling_delivery/pc_columns.txt'
MEDSAM_CKPT  = ('pretrained_weights/hf_cache/models--flaviagiammarino--medsam-vit-base/'
                 'snapshots/a3cb4c518fc3ae8beaf1af0cd3a43867cfa28335/pytorch_model.bin')
RETFOUND_CKPT = 'RETFound_cfp_weights.pth'


# ── retrieval ──────────────────────────────────────────────────────────────────

def eval_retrieval(model, dataset, device, batch_size=32, num_workers=4):
    model.eval()
    seen, idx_list = set(), []
    for i, e in enumerate(dataset.eids):
        if int(e) not in seen:
            seen.add(int(e))
            idx_list.append(i)

    if not idx_list:
        return {'N': 0}

    loader = DataLoader(Subset(dataset, idx_list), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    zfs, zcs, eids_out = [], [], []
    with torch.no_grad():
        for img, cmr, eid in loader:
            img = img.to(device, non_blocking=True)
            cmr = cmr.to(device, non_blocking=True)
            zf, zc = model(img, cmr)
            zfs.append(zf.cpu())
            zcs.append(zc.cpu())
            eids_out.extend(eid.tolist())

    zf = torch.cat(zfs, 0)
    zc = torch.cat(zcs, 0)
    N  = zf.shape[0]
    sim = zf @ zc.T

    ranks_fc = sim.argsort(dim=1, descending=True)
    correct_fc = (ranks_fc == torch.arange(N).unsqueeze(1))
    ranks_cf = sim.T.argsort(dim=1, descending=True)
    correct_cf = (ranks_cf == torch.arange(N).unsqueeze(1))

    out = {'N': N, 'eids': eids_out,
           'zf': zf.numpy(), 'zc': zc.numpy(), 'sim': sim.numpy()}
    for k in (1, 5, 10):
        k_ = min(k, N)
        r_fc = correct_fc[:, :k_].any(1).float().mean().item()
        r_cf = correct_cf[:, :k_].any(1).float().mean().item()
        out[f'R@{k}_f2c'] = r_fc
        out[f'R@{k}_c2f'] = r_cf
        out[f'R@{k}_avg'] = (r_fc + r_cf) / 2
    return out


# ── Spearman ───────────────────────────────────────────────────────────────────

def spearman_analysis(metrics, cmr_table_path, pc_cols_file):
    eids = metrics['eids']
    zf   = metrics['zf']    # (N, D)
    zc   = metrics['zc']    # (N, D)
    sim  = metrics['sim']   # (N, N)  fundus→CMR cosine

    with open(pc_cols_file) as fh:
        raw = fh.read().strip()
    pc_cols = [c.strip() for c in raw.replace('\n', ',').split(',') if c.strip()]

    cmr = pd.read_csv(cmr_table_path)
    cmr = cmr[cmr['eid'].isin(eids)].copy()
    eid2pc = {}
    for _, row in cmr.iterrows():
        e = int(row['eid'])
        vals = [row.get(c, np.nan) for c in pc_cols]
        if not any(np.isnan(v) for v in vals):
            eid2pc[e] = np.array(vals, dtype=np.float32)

    # align to eids order
    valid_idx = [i for i, e in enumerate(eids) if e in eid2pc]
    if len(valid_idx) < 10:
        return {'spearman': float('nan'), 'n_pairs': 0, 'note': 'too few PC14 matches'}

    idx = np.array(valid_idx)
    sub_eids = [eids[i] for i in idx]
    pc_mat = np.stack([eid2pc[e] for e in sub_eids], axis=0)   # (M, 14)

    # fundus→CMR cosine similarities (sub-matrix)
    sub_sim = sim[np.ix_(idx, idx)]   # (M, M)  diagonal = self-pair

    # PC14 euclidean pairwise distances
    diff = pc_mat[:, None, :] - pc_mat[None, :, :]   # (M, M, 14)
    pc_dist = np.sqrt((diff ** 2).sum(-1))              # (M, M)

    # flatten upper triangle (excl diagonal)
    M = len(idx)
    triu = np.triu_indices(M, k=1)
    sims_flat = sub_sim[triu]
    dist_flat = pc_dist[triu]

    rho, pval = spearmanr(sims_flat, -dist_flat)   # higher sim → lower dist expected
    return {
        'spearman_rho': float(rho),
        'pval': float(pval),
        'n_pairs': int(len(sims_flat)),
        'n_eids_with_pc14': M,
    }


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='output_dir/stage2_tier2_unfrozen/checkpoint_best.pth')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--fundus_table', default=FUNDUS_TABLE)
    ap.add_argument('--cmr_npy_dir', default=CMR_NPY_DIR)
    ap.add_argument('--cmr_table', default=CMR_TABLE)
    ap.add_argument('--pc_cols', default=PC_COLS_FILE)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--num_workers', type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'[eval] device={device}  ckpt={args.ckpt}')

    # ── build model & load weights ─────────────────────────────────────────────
    model = Stage2DualTower(
        proj_dim=256, num_frames=4, drop_path_rate=0.0,
        medsam_ckpt=MEDSAM_CKPT, retfound_ckpt=RETFOUND_CKPT,
    )
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.fundus.load_state_dict(ckpt['fundus_model'])
    model.cmr.load_state_dict(ckpt['cmr_model'])
    model.to(device).eval()
    print(f'[eval] checkpoint loaded (val_metrics at save: {ckpt.get("val_metrics", {})})')

    results = {}

    for split in ('val', 'test'):
        print(f'\n[eval] ── {split} ──────────────────────────────────────')
        ds = FundusCMRImageDataset(
            csv_path=args.fundus_table,
            cmr_npy_dir=args.cmr_npy_dir,
            split=split,
            num_frames=4,
            use_eval_transform=True,
        )
        if len(ds) == 0:
            print(f'[eval] {split}: no samples, skipping')
            continue

        m = eval_retrieval(model, ds, device,
                           batch_size=args.batch_size, num_workers=args.num_workers)
        random_r1  = 1.0 / m['N']
        random_r10 = min(10, m['N']) / m['N']

        print(f'[eval] {split} N={m["N"]}')
        print(f'         R@1={m["R@1_avg"]:.4f}  (random={random_r1:.4f})  '
              f'f2c={m["R@1_f2c"]:.4f}  c2f={m["R@1_c2f"]:.4f}')
        print(f'         R@5={m["R@5_avg"]:.4f}')
        print(f'         R@10={m["R@10_avg"]:.4f}  (random={random_r10:.4f})')

        # Spearman
        sp = spearman_analysis(m, args.cmr_table, args.pc_cols)
        rho  = sp.get('spearman_rho', sp.get('spearman', float('nan')))
        pval = sp.get('pval', float('nan'))
        print(f'[eval] {split} Spearman(sim, -PC14_dist) = '
              f'{rho:.4f}  p={pval:.3e}  '
              f'n_pairs={sp["n_pairs"]}  n_eids_pc14={sp.get("n_eids_with_pc14", 0)}')
        if 'note' in sp:
            print(f'         note: {sp["note"]}')
        sp['spearman_rho'] = rho

        results[split] = {
            'N': m['N'],
            'R@1_avg': m['R@1_avg'], 'R@1_f2c': m['R@1_f2c'], 'R@1_c2f': m['R@1_c2f'],
            'R@5_avg': m['R@5_avg'],
            'R@10_avg': m['R@10_avg'],
            'random_R@1': random_r1,
            'random_R@10': random_r10,
            'spearman': sp,
        }

    out_path = os.path.splitext(args.ckpt)[0] + '_full_gallery_eval.json'
    with open(out_path, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f'\n[eval] results saved → {out_path}')


if __name__ == '__main__':
    main()
