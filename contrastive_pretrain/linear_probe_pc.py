"""
linear_probe_pc.py
冻结 ViT backbone（1024-d CLS），训练线性头回归「目标列」。

目标列：
  - **若无**原始 MRI 表型（例如 27 个 M2 原始变量）的 CSV，则使用 fundus 表中的 **14 维 PC score**
   （可不写 --target_cols，脚本会默认用 DEFAULT_PC14）。
  - 若有原始列：用 --target_cols 列出列名，必要时 --mri_targets_csv 按 eid 合并。

数据：
  - fundus_csv 须含 eid, fundus_image_path, split；目标列须在表内或通过 mri_targets_csv 合并得到。

用法示例见 --help。
"""

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from contrastive_pretrain.models_contrast import FundusContrastModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 与对比学习、fundus_table 默认一致；无原始 MRI 变量时用此 14 维 PC
DEFAULT_PC14 = (
    'M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,'
    'M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3'
)


def parse_args():
    p = argparse.ArgumentParser(
        description='Linear probe: frozen RETFound CLS -> regression (PC score or custom targets)',
    )
    p.add_argument('--fundus_csv', required=True, type=str)
    p.add_argument(
        '--target_cols',
        default='',
        type=str,
        help='逗号分隔预测目标列名。留空则使用默认 14 维 PC score（见 DEFAULT_PC14）',
    )
    p.add_argument(
        '--pc_cols',
        default='',
        type=str,
        help='与 --target_cols 等价（兼容旧脚本）',
    )
    p.add_argument(
        '--targets_default_pc14',
        action='store_true',
        help='显式指定使用 14 维 PC score（与省略 target_cols 效果相同）',
    )
    p.add_argument(
        '--mri_targets_csv',
        default='',
        type=str,
        help='可选：含 eid + 目标数值列的 CSV（与 fundus 按 eid inner merge）。列名应与 --target_cols 一致',
    )
    p.add_argument('--retfound_ckpt', required=True, type=str, help='RETFound 初始权重（RETFound_cfp_weights.pth）')
    p.add_argument(
        '--representation',
        default='cls',
        choices=['cls', 'proj'],
        help='cls=冻结 ViT CLS(1024)上线性回归；proj=冻结映射头输出(L2前, proj_dim 维)上线性回归（更贴近对比学习目标）',
    )
    p.add_argument(
        '--contrast_backbone_ckpt',
        default='',
        type=str,
        help='representation=cls 时：对比学习后的 backbone（contrast_pretrain_encoder_best.pth）',
    )
    p.add_argument(
        '--contrast_fundus_full_ckpt',
        default='',
        type=str,
        help='representation=proj 时：完整 fundus 权重（checkpoint_best.pth，含 backbone+proj_head）；'
             '对比学习后这一路必填才有意义',
    )
    p.add_argument('--proj_dim', default=256, type=int, help='FundusContrastModel 结构需与训练一致')
    p.add_argument('--drop_path', default=0.1, type=float)
    p.add_argument('--epochs', default=80, type=int)
    p.add_argument('--batch_size', default=64, type=int)
    p.add_argument('--lr', default=1e-3, type=float)
    p.add_argument('--weight_decay', default=1e-4, type=float)
    p.add_argument('--num_workers', default=4, type=int)
    p.add_argument('--seed', default=0, type=int)
    p.add_argument(
        '--probe_train_strong_aug',
        action='store_true',
        help='训练集用强增强；默认 train/val 均 CenterCrop',
    )
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--output_json', default='', type=str, help='保存结果 JSON 路径')
    return p.parse_args()


def _resolve_target_cols(args):
    if getattr(args, 'targets_default_pc14', False):
        cols = [c.strip() for c in DEFAULT_PC14.split(',') if c.strip()]
        print(f'[linear_probe] 使用 --targets_default_pc14：{len(cols)} 维 PC score')
        return cols
    raw = (args.target_cols or args.pc_cols or '').strip()
    if not raw:
        cols = [c.strip() for c in DEFAULT_PC14.split(',') if c.strip()]
        print(
            f'[linear_probe] 未指定 --target_cols / --pc_cols，默认使用 {len(cols)} 维 PC score（无原始 MRI 变量表时）'
        )
        return cols
    return [c.strip() for c in raw.split(',') if c.strip()]


def load_merged_table(fundus_csv: str, target_cols: list, mri_targets_csv: str) -> pd.DataFrame:
    """返回含 fundus 行 + target 列的表；过滤缺失图像与目标 NaN。"""
    fundus = pd.read_csv(fundus_csv)
    req = {'eid', 'fundus_image_path', 'split'}
    miss = req - set(fundus.columns)
    if miss:
        raise ValueError(f'fundus_csv 缺少列: {miss}')

    if mri_targets_csv:
        extra = pd.read_csv(mri_targets_csv)
        if 'eid' not in extra.columns:
            raise ValueError('mri_targets_csv 必须包含 eid 列')
        miss_t = [c for c in target_cols if c not in extra.columns]
        if miss_t:
            raise ValueError(f'mri_targets_csv 缺少目标列: {miss_t}')
        extra = extra.drop_duplicates(subset=['eid'], keep='first')
        merge_cols = ['eid'] + target_cols
        fundus = fundus.merge(extra[merge_cols], on='eid', how='inner')
    else:
        miss_t = [c for c in target_cols if c not in fundus.columns]
        if miss_t:
            raise ValueError(
                f'目标列不在 fundus_csv 中: {miss_t}。请提供 --mri_targets_csv 或把衍生物列并入 fundus 表。'
            )

    exist_mask = fundus['fundus_image_path'].apply(os.path.exists)
    n_bad = int((~exist_mask).sum())
    if n_bad:
        fundus = fundus[exist_mask].reset_index(drop=True)
        print(f'[load_merged_table] 跳过缺失图像行: {n_bad}')
    fundus = fundus.dropna(subset=target_cols).reset_index(drop=True)
    if len(fundus) == 0:
        raise ValueError('合并后无有效样本（检查 eid 对齐、目标 NaN、图像路径）。')
    print(f'[load_merged_table] 有效行数={len(fundus)}, 目标={target_cols}')
    return fundus


class FundusTargetDataset(Dataset):
    """与 FundusContrastDataset 相同预处理；标签为 target_cols 任意数值列。"""

    def __init__(self, df: pd.DataFrame, target_cols: list, use_train_augmentation: bool):
        self.image_paths = df['fundus_image_path'].tolist()
        self.eids = df['eid'].tolist()
        self.targets = df[target_cols].values.astype(np.float32)
        self.is_train = use_train_augmentation
        self.transform = self._build_transform()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        eid = self.eids[idx]
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        image = Image.open(path).convert('RGB')
        image = self.transform(image)
        return image, eid, y

    def _build_transform(self):
        mean = IMAGENET_DEFAULT_MEAN
        std = IMAGENET_DEFAULT_STD
        if self.is_train:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3 / 4, 4 / 3)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=(-180, 180)),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 3.0))
                ], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


@torch.no_grad()
def extract_features(loader, backbone: nn.Module, device):
    backbone.eval()
    xs, ys = [], []
    for images, _eid, y in loader:
        images = images.to(device, non_blocking=True)
        feat = backbone.forward_features(images)
        xs.append(feat.cpu())
        ys.append(y)
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return X, Y


@torch.no_grad()
def extract_features_proj(loader, model: FundusContrastModel, device):
    """映射头输出，L2 归一化之前（与对比损失一致的可学习空间）。"""
    model.eval()
    xs, ys = [], []
    for images, _eid, y in loader:
        images = images.to(device, non_blocking=True)
        feat = model.backbone.forward_features(images)
        h = model.proj_head(feat)
        xs.append(h.cpu())
        ys.append(y)
    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return X, Y


def load_fundus_from_full_checkpoint(model: FundusContrastModel, path: str) -> None:
    ckpt = torch.load(path, map_location='cpu')
    if 'fundus_model' not in ckpt:
        raise ValueError(f'{path} 中无 fundus_model 键，请使用 main_contrast 保存的 checkpoint_best.pth')
    msg = model.load_state_dict(ckpt['fundus_model'], strict=True)
    print(f'[linear_probe] 已加载完整 fundus（backbone+proj_head）: {path}\n  {msg}')


def train_linear_probe(X_tr, Y_tr, X_val, Y_val, device, epochs, lr, weight_decay, seed, target_names):
    torch.manual_seed(seed)
    n_in, n_out = X_tr.shape[1], Y_tr.shape[1]
    head = nn.Linear(n_in, n_out).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 1e-3)

    Y_mean = Y_tr.mean(0, keepdim=True)
    Y_std = Y_tr.std(0, keepdim=True).clamp_min(1e-8)
    Y_tr_n = (Y_tr.to(device) - Y_mean.to(device)) / Y_std.to(device)
    Y_val_n = (Y_val.to(device) - Y_mean.to(device)) / Y_std.to(device)
    X_tr = X_tr.to(device)
    X_val = X_val.to(device)

    n = X_tr.shape[0]
    best_val = float('inf')
    best_state = None

    for ep in range(epochs):
        head.train()
        perm = torch.randperm(n, device=device)
        loss_acc = 0.0
        bs = min(512, n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            pred = head(X_tr[idx])
            loss = torch.nn.functional.mse_loss(pred, Y_tr_n[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_acc += loss.item() * idx.numel()
        sched.step()
        loss_acc /= n

        head.eval()
        with torch.no_grad():
            pv = head(X_val)
            val_mse = torch.nn.functional.mse_loss(pv, Y_val_n).item()
        if val_mse < best_val:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred_tr = head(X_tr).cpu().numpy()
        pred_val = head(X_val).cpu().numpy()
    y_tr_np = Y_tr_n.cpu().numpy()
    y_val_np = Y_val_n.cpu().numpy()

    pearsons_tr, pearsons_val = [], []
    spearmans_tr, spearmans_val = [], []
    for i in range(n_out):
        pearsons_tr.append(pearsonr(pred_tr[:, i], y_tr_np[:, i])[0])
        pearsons_val.append(pearsonr(pred_val[:, i], y_val_np[:, i])[0])
        spearmans_tr.append(spearmanr(pred_tr[:, i], y_tr_np[:, i]).correlation)
        spearmans_val.append(spearmanr(pred_val[:, i], y_val_np[:, i]).correlation)

    def _mean_nan_safe(arr):
        x = np.array(arr, dtype=np.float64)
        return float(np.nanmean(x))

    per_target = {}
    for i, name in enumerate(target_names):
        per_target[name] = {
            'pearson_val': float(pearsons_val[i]),
            'spearman_val': float(spearmans_val[i]),
        }

    return {
        'train_mse_normalized': float(loss_acc),
        'best_val_mse_normalized': float(best_val),
        'mean_pearson_train': _mean_nan_safe(pearsons_tr),
        'mean_pearson_val': _mean_nan_safe(pearsons_val),
        'mean_spearman_train': _mean_nan_safe(spearmans_tr),
        'mean_spearman_val': _mean_nan_safe(spearmans_val),
        'per_dim_pearson_val': [float(x) for x in pearsons_val],
        'per_dim_spearman_val': [float(x) for x in spearmans_val],
        'per_target_val': per_target,
        'target_names': target_names,
    }


def run_one_probe(name, model_or_backbone, train_loader, val_loader, device, args, target_names, representation: str):
    print(f'\n========== {name}  ({representation}) ==========')
    if representation == 'cls':
        X_tr, Y_tr = extract_features(train_loader, model_or_backbone, device)
        X_val, Y_val = extract_features(val_loader, model_or_backbone, device)
    else:
        X_tr, Y_tr = extract_features_proj(train_loader, model_or_backbone, device)
        X_val, Y_val = extract_features_proj(val_loader, model_or_backbone, device)
    print(f'Features: train {X_tr.shape}, val {X_val.shape}')
    stats = train_linear_probe(
        X_tr, Y_tr, X_val, Y_val, device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, seed=args.seed,
        target_names=target_names,
    )
    stats['backbone'] = name
    stats['representation'] = representation
    print(
        f"[{name}] val  mean Pearson={stats['mean_pearson_val']:.4f}  "
        f"mean Spearman={stats['mean_spearman_val']:.4f}  "
        f"(train mean Pearson={stats['mean_pearson_train']:.4f})"
    )
    print(f"  best val MSE (normalized targets): {stats['best_val_mse_normalized']:.6f}")
    if len(target_names) <= 8:
        for t in target_names:
            pt = stats['per_target_val'][t]
            print(f"    {t}:  rho={pt['spearman_val']:.4f}  r={pt['pearson_val']:.4f}")
    return stats


def main():
    args = parse_args()
    target_cols = _resolve_target_cols(args)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    use_train_aug = args.probe_train_strong_aug

    merged = load_merged_table(args.fundus_csv, target_cols, args.mri_targets_csv or '')
    df_tr = merged[merged['split'] == 'train'].reset_index(drop=True)
    df_val = merged[merged['split'] == 'val'].reset_index(drop=True)
    if len(df_tr) == 0 or len(df_val) == 0:
        raise ValueError(f'train/val 样本为空: train={len(df_tr)}, val={len(df_val)}')

    ds_tr = FundusTargetDataset(df_tr, target_cols, use_train_augmentation=use_train_aug)
    ds_val = FundusTargetDataset(df_val, target_cols, use_train_augmentation=False)

    train_loader = DataLoader(
        ds_tr, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        ds_val, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    rep = args.representation
    results = {
        'args': vars(args),
        'target_cols': target_cols,
        'representation': rep,
        'n_train': len(df_tr),
        'n_val': len(df_val),
        'runs': [],
    }

    model_init = FundusContrastModel(proj_dim=args.proj_dim, drop_path_rate=args.drop_path)
    model_init.load_pretrained(args.retfound_ckpt)
    model_init.to(device)
    for p in model_init.parameters():
        p.requires_grad_(False)
    if rep == 'cls':
        results['runs'].append(
            run_one_probe(
                'retfound_init', model_init.backbone, train_loader, val_loader, device, args, target_cols, 'cls',
            )
        )
    else:
        print('[linear_probe] proj 基线：RETFound backbone + **随机初始化且冻结**的映射头（未经过对比学习）')
        results['runs'].append(
            run_one_probe(
                'retfound_init_random_proj_head', model_init, train_loader, val_loader, device, args, target_cols, 'proj',
            )
        )

    del model_init
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if rep == 'cls' and args.contrast_backbone_ckpt:
        model_ft = FundusContrastModel(proj_dim=args.proj_dim, drop_path_rate=args.drop_path)
        model_ft.load_pretrained(args.retfound_ckpt)
        model_ft.load_pretrained(args.contrast_backbone_ckpt)
        model_ft.to(device)
        for p in model_ft.parameters():
            p.requires_grad_(False)
        results['runs'].append(
            run_one_probe(
                'contrastive_finetuned', model_ft.backbone, train_loader, val_loader, device, args, target_cols, 'cls',
            )
        )
        del model_ft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if rep == 'proj' and args.contrast_fundus_full_ckpt:
        model_ft = FundusContrastModel(proj_dim=args.proj_dim, drop_path_rate=args.drop_path)
        model_ft.load_pretrained(args.retfound_ckpt)
        load_fundus_from_full_checkpoint(model_ft, args.contrast_fundus_full_ckpt)
        model_ft.to(device)
        for p in model_ft.parameters():
            p.requires_grad_(False)
        results['runs'].append(
            run_one_probe(
                'contrastive_finetuned', model_ft, train_loader, val_loader, device, args, target_cols, 'proj',
            )
        )
        del model_ft
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    elif rep == 'proj' and not args.contrast_fundus_full_ckpt:
        print('[linear_probe] 未提供 --contrast_fundus_full_ckpt，跳过「对比学习后」映射头探针')

    if args.output_json:
        out = args.output_json
        os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
        with open(out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f'\nWrote {out}')

    print('\n--- Summary (validation, mean over targets) ---')
    for r in results['runs']:
        rep_tag = r.get('representation', 'cls')
        print(
            f"  [{rep_tag}] {r['backbone']:<32}  Pearson={r['mean_pearson_val']:.4f}  "
            f"Spearman={r['mean_spearman_val']:.4f}"
        )


if __name__ == '__main__':
    main()
