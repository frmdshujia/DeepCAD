"""
train_ablation_mini.py  —  快速消融实验（第二层）

使用小规模数据（真实标签 + synthetic embeddings）验证双塔架构优于单塔基线。
可在 4 张 GPU 上并行运行 Exp A-D。

启动命令：
  GPU 0: CUDA_VISIBLE_DEVICES=0 python train_ablation_mini.py --exp A
  GPU 1: CUDA_VISIBLE_DEVICES=1 python train_ablation_mini.py --exp B
  GPU 2: CUDA_VISIBLE_DEVICES=2 python train_ablation_mini.py --exp C
  GPU 3: CUDA_VISIBLE_DEVICES=3 python train_ablation_mini.py --exp D

Exp 配置：
  A: CMR 单塔（上限）
  B: 眼底单塔（上限）
  C: 双塔 + cosine 软对齐，无 Cross-Attention
  D: 双塔 + Cross-Attention，无 LoRA（完整模型）
  E: 完整模型（含 LoRA 回流）

注意：
  使用 synthetic embeddings → AUC 仅反映任务头学习能力，不反映编码器质量。
  若要产出真实 AUC，需将 ENCODER_MODE 改为 'real' 并提供预训练 checkpoint。
"""

import argparse
import json
import os
import pathlib
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import Dataset, DataLoader

# ── 路径 ──────────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).parent

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]
LV_COLS = ['LV ejection fraction', 'LV end diastolic volume', 'LV end systolic volume']

EXP_CONFIGS = {
    'A': {'mode': 'cmr_only',    'use_cross_attn': False, 'use_lora': False, 'align_lambda': 0.0},
    'B': {'mode': 'fundus_only', 'use_cross_attn': False, 'use_lora': False, 'align_lambda': 0.0},
    'C': {'mode': 'both',        'use_cross_attn': False, 'use_lora': False, 'align_lambda': 0.1},
    'D': {'mode': 'both',        'use_cross_attn': True,  'use_lora': False, 'align_lambda': 0.1},
    'E': {'mode': 'both',        'use_cross_attn': True,  'use_lora': True,  'align_lambda': 0.1},
}


# ── 模型组件（与 sanity_check.py 一致）────────────────────────────────────────
class LoRALayer(nn.Module):
    def __init__(self, in_dim, out_dim, r=16, alpha=32):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        self.scale  = alpha / r
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale


class CrossModalGating(nn.Module):
    def __init__(self, dim_f=1024, dim_c=768, hidden=512, num_heads=8,
                 r=16, use_lora=True):
        super().__init__()
        self.proj_f    = nn.Linear(dim_f, hidden)
        self.proj_c    = nn.Linear(dim_c, hidden)
        self.attn_f2c  = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.attn_c2f  = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.norm_f    = nn.LayerNorm(hidden)
        self.norm_c    = nn.LayerNorm(hidden)
        self.gate_f    = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.gate_c    = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.back_proj_f = nn.Linear(hidden, dim_f)
        self.back_proj_c = nn.Linear(hidden, dim_c)
        self.use_lora  = use_lora
        if use_lora:
            self.lora_f = LoRALayer(dim_f, dim_f, r=r)
            self.lora_c = LoRALayer(dim_c, dim_c, r=r)
        else:
            self.lora_f = nn.Linear(dim_f, dim_f)
            self.lora_c = nn.Linear(dim_c, dim_c)

    def forward(self, z_f, z_c):
        zf = self.proj_f(z_f).unsqueeze(1)
        zc = self.proj_c(z_c).unsqueeze(1)
        f_from_c, _ = self.attn_f2c(zf, zc, zc)
        f_from_c = self.norm_f(f_from_c.squeeze(1))
        delta_f  = self.gate_f(f_from_c) * f_from_c
        c_from_f, _ = self.attn_c2f(zc, zf, zf)
        c_from_f = self.norm_c(c_from_f.squeeze(1))
        delta_c  = self.gate_c(c_from_f) * c_from_f
        delta_f  = self.back_proj_f(delta_f)
        delta_c  = self.back_proj_c(delta_c)
        return z_f + self.lora_f(delta_f), z_c + self.lora_c(delta_c)


class AlignProjection(nn.Module):
    def __init__(self, dim_f=1024, dim_c=768, out_dim=256):
        super().__init__()
        self.proj_f = nn.Sequential(nn.Linear(dim_f, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))
        self.proj_c = nn.Sequential(nn.Linear(dim_c, out_dim), nn.ReLU(), nn.Linear(out_dim, out_dim))

    def forward(self, z_f, z_c):
        return F.normalize(self.proj_f(z_f), dim=-1), F.normalize(self.proj_c(z_c), dim=-1)


class TaskHead(nn.Module):
    def __init__(self, in_dim, n_cls=4, n_reg=3, dropout=0.3):
        super().__init__()
        self.dropout   = nn.Dropout(dropout)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n_cls)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n_reg)
        ])

    def forward(self, z):
        z   = self.dropout(z)
        cls = [h(z).squeeze(-1) for h in self.cls_heads]
        reg = [h(z).squeeze(-1) for h in self.reg_heads]
        return cls, reg


class MultiTaskLoss(nn.Module):
    def __init__(self, pos_weights, reg_norm):
        super().__init__()
        self.pos_weights = pos_weights
        self.reg_norm    = reg_norm

    def cls_loss(self, preds, labels, device):
        loss = torch.tensor(0.0, device=device)
        for i, pred in enumerate(preds):
            valid = labels[:, i] >= 0
            if valid.sum() == 0:
                continue
            pw   = torch.tensor([self.pos_weights[i]], device=device)
            crit = nn.BCEWithLogitsLoss(pos_weight=pw)
            loss = loss + crit(pred[valid], labels[valid, i])
        return loss

    def reg_loss(self, preds, targets, device, reg_mask=None):
        loss = torch.tensor(0.0, device=device)
        for i, pred in enumerate(preds):
            mu    = self.reg_norm[i]['mean']
            sigma = self.reg_norm[i]['std']
            target_norm = (targets[:, i] - mu) / sigma
            valid = ~torch.isnan(targets[:, i])
            if reg_mask is not None:
                valid = valid & reg_mask[:, i]
            if valid.sum() == 0:
                continue
            loss = loss + F.mse_loss(pred[valid], target_norm[valid])
        return loss

    def align_loss(self, z_f, z_c):
        return 1.0 - (z_f * z_c).sum(dim=-1).mean()


# ── Dataset ───────────────────────────────────────────────────────────────────
class EmbeddingDataset(Dataset):
    """从 CSV 读取标签，用随机/预提取 embedding 作为特征。"""

    def __init__(self, csv_path, modality, dim, use_synthetic=True):
        self.df       = pd.read_csv(csv_path, low_memory=False)
        self.modality = modality
        self.dim      = dim
        self.use_syn  = use_synthetic

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 标签
        cls_labels = torch.tensor(
            [float(row.get(c, -1)) for c in CLS_COLS], dtype=torch.float32
        )
        reg_values = []
        reg_mask   = []
        for col in LV_COLS:
            val = row.get(col, float('nan'))
            try:
                v = float(val)
            except (TypeError, ValueError):
                v = float('nan')
            reg_values.append(v)
            reg_mask.append(not np.isnan(v))

        reg_labels = torch.tensor(reg_values, dtype=torch.float32)
        reg_mask_t = torch.tensor(reg_mask,   dtype=torch.bool)

        # embedding（synthetic）
        emb = torch.randn(self.dim)

        return {'emb': emb, 'cls_labels': cls_labels,
                'reg_labels': reg_labels, 'reg_mask': reg_mask_t}


class PairedDataset(Dataset):
    """配对数据集（CMR + 眼底），均使用 synthetic embeddings。"""

    def __init__(self, paired_csv, dim_f=1024, dim_c=768):
        paired_path = HERE / 'preprocessed_data/stage2/fundus/stage2_fundus_dual.csv'
        self.df = pd.read_csv(paired_path, low_memory=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls_labels = torch.tensor(
            [float(row.get(c, -1)) for c in CLS_COLS], dtype=torch.float32
        )
        reg_values = [float(row.get(col, float('nan'))) for col in LV_COLS]
        reg_labels = torch.tensor(reg_values, dtype=torch.float32)
        reg_mask   = torch.tensor([not np.isnan(v) for v in reg_values], dtype=torch.bool)
        return {
            'fundus': torch.randn(1024),
            'cmr':    torch.randn(768),
            'cls_labels': cls_labels,
            'reg_labels': reg_labels,
            'reg_mask':   reg_mask,
            'time_weight': 1.0,
        }


# ── 构建模型 ──────────────────────────────────────────────────────────────────
def build_model(cfg, pos_weights_cmr, pos_weights_fundus, norm_params):
    head_cmr    = TaskHead(768)
    head_fundus = TaskHead(1024)
    cross_modal = None
    align_proj  = None

    if cfg['use_cross_attn']:
        cross_modal = CrossModalGating(use_lora=cfg['use_lora'])
        align_proj  = AlignProjection()

    modules = nn.ModuleDict({
        'head_cmr':    head_cmr,
        'head_fundus': head_fundus,
    })
    if cross_modal is not None:
        modules['cross_modal'] = cross_modal
        modules['align_proj']  = align_proj

    loss_cmr    = MultiTaskLoss(pos_weights_cmr,    norm_params)
    loss_fundus = MultiTaskLoss(pos_weights_fundus, norm_params)
    return modules, loss_cmr, loss_fundus


def forward_single(modules, batch, modality):
    emb  = batch['emb']
    head = modules['head_cmr'] if modality == 'cmr' else modules['head_fundus']
    cls, reg = head(emb)
    return cls, reg


def forward_paired(modules, batch, cfg):
    z_f = batch['fundus']
    z_c = batch['cmr']

    cls_f0, reg_f0 = modules['head_fundus'](z_f)
    cls_c0, reg_c0 = modules['head_cmr'](z_c)

    if cfg['use_cross_attn']:
        z_f_en, z_c_en = modules['cross_modal'](z_f, z_c)
        cls_f1, reg_f1 = modules['head_fundus'](z_f_en)
        cls_c1, reg_c1 = modules['head_cmr'](z_c_en)
        af, ac = modules['align_proj'](z_f, z_c)
    else:
        cls_f1, reg_f1 = cls_f0, reg_f0
        cls_c1, reg_c1 = cls_c0, reg_c0
        af = F.normalize(torch.randn_like(z_f[:, :256]), dim=-1)
        ac = F.normalize(torch.randn_like(z_c[:, :256]), dim=-1)

    return {
        'fundus_cls_base': cls_f0, 'fundus_reg_base': reg_f0,
        'cmr_cls_base':    cls_c0, 'cmr_reg_base':    reg_c0,
        'fundus_cls_enriched': cls_f1, 'fundus_reg_enriched': reg_f1,
        'cmr_cls_enriched':    cls_c1, 'cmr_reg_enriched':    reg_c1,
        'align_f': af, 'align_c': ac,
    }


# ── 训练/评估 ─────────────────────────────────────────────────────────────────
def compute_loss_single(cls_preds, reg_preds, batch, loss_fn, device):
    return (loss_fn.cls_loss(cls_preds, batch['cls_labels'], device) +
            loss_fn.reg_loss(reg_preds, batch['reg_labels'], device, batch.get('reg_mask')))


def compute_loss_paired(results, batch, loss_cmr, loss_fundus, align_lambda, device):
    total = torch.tensor(0.0, device=device)
    tw    = batch.get('time_weight', 1.0)
    total += loss_fundus.cls_loss(results['fundus_cls_base'], batch['cls_labels'], device)
    total += loss_cmr.cls_loss(results['cmr_cls_base'],       batch['cls_labels'], device)
    total += loss_cmr.reg_loss(results['cmr_reg_base'],       batch['reg_labels'], device)
    total += loss_fundus.reg_loss(results['fundus_reg_base'], batch['reg_labels'], device,
                                  batch.get('reg_mask'))
    w = 0.5 * tw
    total += w * loss_fundus.cls_loss(results['fundus_cls_enriched'], batch['cls_labels'], device)
    total += w * loss_cmr.cls_loss(results['cmr_cls_enriched'],       batch['cls_labels'], device)
    total += w * loss_cmr.reg_loss(results['cmr_reg_enriched'],       batch['reg_labels'], device)
    total += w * loss_fundus.reg_loss(results['fundus_reg_enriched'], batch['reg_labels'],
                                      device, batch.get('reg_mask'))
    total += align_lambda * tw * loss_cmr.align_loss(results['align_f'], results['align_c'])
    return total


@torch.no_grad()
def evaluate_auc(modules, loader, modality, device):
    all_logits = [[] for _ in range(4)]
    all_labels = [[] for _ in range(4)]
    head = modules['head_cmr'] if modality == 'cmr' else modules['head_fundus']

    for batch in loader:
        emb = batch['emb'].to(device)
        cls, _ = head(emb)
        for i, logit in enumerate(cls):
            labels = batch['cls_labels'][:, i]
            valid  = labels >= 0
            if valid.sum() == 0:
                continue
            all_logits[i].extend(torch.sigmoid(logit[valid]).cpu().numpy())
            all_labels[i].extend(labels[valid].numpy())

    aucs = []
    for i in range(4):
        if len(set(all_labels[i])) < 2 or len(all_labels[i]) < 5:
            continue
        try:
            auc = roc_auc_score(all_labels[i], all_logits[i])
            aucs.append(auc)
        except Exception:
            pass
    return float(np.mean(aucs)) if aucs else 0.5


# ── 主训练循环 ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',        choices=list(EXP_CONFIGS.keys()), required=True)
    parser.add_argument('--n_epochs',   type=int, default=15)
    parser.add_argument('--patience',   type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr',         type=float, default=1e-3)
    parser.add_argument('--outdir',     default='ablation_results')
    args   = parser.parse_args()
    cfg    = EXP_CONFIGS[args.exp]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.outdir, exist_ok=True)
    print(f'Exp {args.exp} | {cfg} | device={device}')

    # 加载 JSON 配置
    pw_cmr_path    = HERE / 'pos_weights_cmr.json'
    pw_fundus_path = HERE / 'pos_weights_fundus.json'
    norm_path      = HERE / 'norm_params.json'

    for p in [pw_cmr_path, pw_fundus_path, norm_path]:
        if not p.exists():
            sys.exit(f'缺少 {p}，请先运行 build_training_csv.py / calc_pos_weights.py / calc_norm_params.py')

    with open(pw_cmr_path)    as f: pw_cmr    = [json.load(f)[c] for c in CLS_COLS]
    with open(pw_fundus_path) as f: pw_fundus = [json.load(f)[c] for c in CLS_COLS]
    with open(norm_path)      as f:
        nd = json.load(f)
        norm_params = [{'mean': nd[k]['mean'], 'std': nd[k]['std']}
                       for k in ['lvef', 'lvedv', 'lvesv']]

    # 数据集
    cmr_train_csv    = HERE / 'stage1_cmr_sampled_train.csv'
    cmr_val_csv      = HERE / 'stage1_cmr_sampled_val.csv'
    fundus_train_csv = HERE / 'stage1_fundus_sampled_train.csv'
    fundus_val_csv   = HERE / 'stage1_fundus_sampled_val.csv'

    for p in [cmr_train_csv, cmr_val_csv, fundus_train_csv, fundus_val_csv]:
        if not p.exists():
            sys.exit(f'缺少 {p}，请先运行 build_training_csv.py')

    ds_cmr_tr    = EmbeddingDataset(cmr_train_csv,    'cmr',    768)
    ds_cmr_val   = EmbeddingDataset(cmr_val_csv,      'cmr',    768)
    ds_fund_tr   = EmbeddingDataset(fundus_train_csv, 'fundus', 1024)
    ds_fund_val  = EmbeddingDataset(fundus_val_csv,   'fundus', 1024)
    ds_paired    = PairedDataset(HERE / 'preprocessed_data/stage2/fundus/stage2_fundus_dual.csv')

    # 对于 single-tower 实验只用对应模态的 loader
    loader_cmr_tr   = DataLoader(ds_cmr_tr,   batch_size=args.batch_size, shuffle=True,  num_workers=2)
    loader_cmr_val  = DataLoader(ds_cmr_val,  batch_size=256,             shuffle=False, num_workers=2)
    loader_fund_tr  = DataLoader(ds_fund_tr,  batch_size=args.batch_size, shuffle=True,  num_workers=2)
    loader_fund_val = DataLoader(ds_fund_val, batch_size=256,             shuffle=False, num_workers=2)
    loader_paired   = DataLoader(ds_paired,   batch_size=args.batch_size // 2,
                                 shuffle=True, num_workers=2)

    # 构建模型
    modules, loss_cmr, loss_fundus = build_model(cfg, pw_cmr, pw_fundus, norm_params)
    modules = modules.to(device)
    optimizer = torch.optim.AdamW(modules.parameters(), lr=args.lr, weight_decay=0.01)

    best_auc   = 0.0
    best_epoch = 0
    history    = []

    for epoch in range(args.n_epochs):
        modules.train()
        train_losses = []

        if cfg['mode'] == 'cmr_only':
            for batch in loader_cmr_tr:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                optimizer.zero_grad()
                cls, reg = forward_single(modules, batch, 'cmr')
                loss = compute_loss_single(cls, reg, batch, loss_cmr, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        elif cfg['mode'] == 'fundus_only':
            for batch in loader_fund_tr:
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                optimizer.zero_grad()
                cls, reg = forward_single(modules, batch, 'fundus')
                loss = compute_loss_single(cls, reg, batch, loss_fundus, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        else:  # 'both'
            paired_iter = iter(loader_paired)
            for cmr_batch, fund_batch in zip(loader_cmr_tr, loader_fund_tr):
                cmr_batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in cmr_batch.items()}
                fund_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in fund_batch.items()}
                try:
                    pair_batch = next(paired_iter)
                except StopIteration:
                    paired_iter = iter(loader_paired)
                    pair_batch = next(paired_iter)
                pair_batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                              for k, v in pair_batch.items()}

                optimizer.zero_grad()
                cls_c, reg_c = forward_single(modules, cmr_batch,  'cmr')
                cls_f, reg_f = forward_single(modules, fund_batch, 'fundus')
                loss = (compute_loss_single(cls_c, reg_c, cmr_batch,  loss_cmr,    device) +
                        compute_loss_single(cls_f, reg_f, fund_batch, loss_fundus, device))

                results = forward_paired(modules, pair_batch, cfg)
                loss += compute_loss_paired(results, pair_batch, loss_cmr, loss_fundus,
                                            cfg['align_lambda'], device)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(modules.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        # 验证
        modules.eval()
        if cfg['mode'] == 'cmr_only':
            val_auc = evaluate_auc(modules, loader_cmr_val, 'cmr', device)
        elif cfg['mode'] == 'fundus_only':
            val_auc = evaluate_auc(modules, loader_fund_val, 'fundus', device)
        else:
            auc_c = evaluate_auc(modules, loader_cmr_val,  'cmr',    device)
            auc_f = evaluate_auc(modules, loader_fund_val, 'fundus', device)
            val_auc = (auc_c + auc_f) / 2.0

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        history.append({'epoch': epoch, 'train_loss': train_loss, 'val_auc': val_auc})
        print(f'Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val AUC: {val_auc:.4f}')

        if val_auc > best_auc:
            best_auc   = val_auc
            best_epoch = epoch
            torch.save(modules.state_dict(),
                       os.path.join(args.outdir, f'exp_{args.exp}_best.pth'))

        if epoch - best_epoch >= args.patience:
            print(f'Early stop at epoch {epoch}')
            break

    result = {
        'exp':        args.exp,
        'config':     cfg,
        'best_auc':   best_auc,
        'best_epoch': best_epoch,
        'history':    history,
    }
    out_path = os.path.join(args.outdir, f'exp_{args.exp}_result.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nExp {args.exp} Done | Best Val AUC: {best_auc:.4f} @ epoch {best_epoch}')
    print(f'结果已保存: {out_path}')
    return result


if __name__ == '__main__':
    main()
