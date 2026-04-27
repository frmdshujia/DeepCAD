"""
在指定数据划分（val / test）上评估已保存的对比学习 checkpoint（完整 metrics + gt_pred）。
用法：
  python contrastive_pretrain/eval_checkpoint_split.py \\
    --checkpoint output_dir/.../checkpoint_best.pth --split test
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
from torch.utils.data import DataLoader

import collections.abc
if 'torch._six' not in sys.modules:
    class _TorchSix:
        container_abcs = collections.abc
        inf = float('inf')
    sys.modules['torch._six'] = _TorchSix()

from contrastive_pretrain.datasets_contrast import FundusContrastDataset, CMRBank
from contrastive_pretrain.models_contrast import FundusContrastModel, CMREncoder
from contrastive_pretrain.eval_contrast import run_full_eval


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True, type=str)
    p.add_argument('--split', default='test', choices=['val', 'test'])
    p.add_argument('--fundus_csv', default='contrastive_pretrain/preprocessed_data/fundus_table.csv')
    p.add_argument('--cmr_csv', default='contrastive_pretrain/preprocessed_data/cmr_table.csv')
    p.add_argument('--pc_cols', default='M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3')
    p.add_argument('--sigma', type=float, default=None, help='默认从 checkpoint args 读取')
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    ca = ckpt.get('args') or {}

    sigma = float(args.sigma if args.sigma is not None else ca.get('sigma', 6.5893))
    proj_dim = int(ca.get('proj_dim', 256))
    drop_path = float(ca.get('drop_path', 0.1))
    n_pc = int(ca.get('n_pc', 14))

    pc_cols = [c.strip() for c in args.pc_cols.split(',')]
    if len(pc_cols) != n_pc and 'n_pc' in ca:
        n_pc = len(pc_cols)

    fundus_csv = args.fundus_csv if os.path.isabs(args.fundus_csv) else os.path.join(ROOT, args.fundus_csv)
    cmr_csv = args.cmr_csv if os.path.isabs(args.cmr_csv) else os.path.join(ROOT, args.cmr_csv)

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    dataset = FundusContrastDataset(fundus_csv, pc_cols, split=args.split)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    cmr_bank = CMRBank(cmr_csv, pc_cols, split=args.split, device=str(device))

    fundus_model = FundusContrastModel(proj_dim=proj_dim, drop_path_rate=drop_path)
    cmr_enc = CMREncoder(in_dim=n_pc, hidden_dim=128, out_dim=proj_dim)

    fundus_model.load_state_dict(ckpt['fundus_model'])
    cmr_enc.load_state_dict(ckpt['cmr_encoder'])
    fundus_model.to(device)
    cmr_enc.to(device)

    print(f'[eval_split] checkpoint={args.checkpoint}')
    print(f'[eval_split] split={args.split}  fundus_N={len(dataset)}  cmr_bank_N={cmr_bank.n_cmr}  sigma={sigma}')

    metrics = run_full_eval(
        fundus_model, cmr_enc, loader, cmr_bank,
        device=str(device), sigma=sigma,
    )
    print(f'[eval_split] metrics={metrics}')

    out_json = args.checkpoint.replace('.pth', f'_metrics_{args.split}.json')
    serial = {}
    for k, v in metrics.items():
        serial[k] = int(v) if k == 'n_pairs' else float(v)
    with open(out_json, 'w') as f:
        json.dump(serial, f, indent=2)
    print(f'[eval_split] wrote {out_json}')


if __name__ == '__main__':
    main()
