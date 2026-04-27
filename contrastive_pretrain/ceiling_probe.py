"""ceiling_probe.py
CMR / Fundus ceiling experiments: linear probe + full fine-tune.

Experiments:
  1. MedSAM linear  CMR (all 8224 with npy)  tasks: i21 isch_hd lvef lvedv
  2. MedSAM finetune CMR                      tasks: i21 isch_hd lvef lvedv
  3. RETFound linear Fundus (10k)             tasks: i21 isch_hd
  4. RETFound finetune Fundus                 tasks: i21 isch_hd lvef lvedv
  5. MedSAM finetune CMR different sequences  tasks: i21 isch_hd
     modalities: cmr_all cmr_lax cmr_sax

Usage:
  CUDA_VISIBLE_DEVICES=0 python contrastive_pretrain/ceiling_probe.py \\
      --encoder medsam --modality cmr_all --mode linear --task i21 \\
      --out_dir output_dir/ceiling/medsam_linear_cmr_all_i21

  CUDA_VISIBLE_DEVICES=1 python contrastive_pretrain/ceiling_probe.py \\
      --encoder medsam --modality cmr_all --mode finetune --task lvef \\
      --out_dir output_dir/ceiling/medsam_finetune_cmr_all_lvef \\
      --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
"""
from __future__ import annotations
import sys, os, argparse, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
STAGE1_CMR_CSV   = 'contrastive_pretrain/preprocessed_data/stage1/stage1_cmr.csv'
STAGE1_FUND_CSV  = 'contrastive_pretrain/preprocessed_data/stage1/stage1_fundus.csv'
CMR_NPY_DIR      = '/data/home/shujia/UKB/CMRI/preprocessed_lax_sax'
FUND_IMG_DIR     = '/data/home/home6/fundus_data/UKB/fundus_images'
RETFOUND_CKPT    = 'RETFound_cfp_weights.pth'
MEDSAM_CKPT      = ('pretrained_weights/hf_cache/models--flaviagiammarino--medsam-vit-base'
                    '/snapshots/a3cb4c518fc3ae8beaf1af0cd3a43867cfa28335/pytorch_model.bin')

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# Task metadata
TASK_META = {
    'i21':     {'type': 'binary',     'col': 'prevalent_I21',           'label': 'Myocardial Infarction (I21)'},
    'isch_hd': {'type': 'binary',     'col': 'composite_ischemic_hd',   'label': 'Ischemic Heart Disease (composite)'},
    'i25':     {'type': 'binary',     'col': 'prevalent_I25',           'label': 'Chronic Ischemic HD (I25)'},
    'i20':     {'type': 'binary',     'col': 'prevalent_I20',           'label': 'Angina Pectoris (I20)'},
    'lvef':    {'type': 'regression', 'col': 'LV ejection fraction',    'label': 'LVEF (%)'},
    'lvedv':   {'type': 'regression', 'col': 'LV end diastolic volume', 'label': 'LVEDV (mL)'},
}

# CMR label source for fundus subjects (join from stage1_cmr)
CMR_REGRESSION_COLS = ['LV ejection fraction', 'LV end diastolic volume']

# ---------------------------------------------------------------------------
# Cohort builders
# ---------------------------------------------------------------------------

def build_cmr_cohort() -> pd.DataFrame:
    """All stage1_cmr instance=2 subjects that have a preprocessed npy file."""
    cmr = pd.read_csv(STAGE1_CMR_CSV, low_memory=False)
    cmr = cmr[cmr['instance'] == 2].drop_duplicates('eid').copy()
    npy_eids = {int(f.replace('.npy', ''))
                for f in os.listdir(CMR_NPY_DIR) if f.endswith('.npy')}
    cmr = cmr[cmr['eid'].isin(npy_eids)].reset_index(drop=True)
    print(f'[cohort] CMR: {len(cmr)} subjects with npy')
    for col in ['prevalent_I21', 'composite_ischemic_hd',
                'LV ejection fraction', 'LV end diastolic volume']:
        nn = cmr[col].notna().sum()
        if cmr[col].dtype != object and cmr[col].nunique() <= 2:
            print(f'  {col}: {int(cmr[col].sum())} pos / {nn} non-null')
        else:
            print(f'  {col}: {nn} non-null  mean={cmr[col].mean():.1f}')
    return cmr


def _fundus_img_path(eid: int, instance: int) -> str | None:
    for field in ['21015', '21016']:
        p = os.path.join(FUND_IMG_DIR, f'{eid}_{field}_{instance}_0.png')
        if os.path.exists(p):
            return p
    return None


def build_fundus_cohort(n: int = 10000, seed: int = 42) -> pd.DataFrame:
    """10k fundus subjects (or all if fewer available) with image paths."""
    fund = pd.read_csv(STAGE1_FUND_CSV, low_memory=False)
    # one row per EID (lowest instance first)
    fund = fund.sort_values('instance').groupby('eid').first().reset_index()
    fund['image_path'] = fund.apply(
        lambda r: _fundus_img_path(int(r['eid']), int(r['instance'])), axis=1)
    fund = fund[fund['image_path'].notna()].reset_index(drop=True)

    # Attach CMR regression labels (LVEF/LVEDV) for subjects who have CMR too
    cmr_reg = pd.read_csv(STAGE1_CMR_CSV, low_memory=False)
    cmr_reg = cmr_reg[cmr_reg['instance'] == 2].drop_duplicates('eid')
    cmr_reg = cmr_reg[['eid'] + CMR_REGRESSION_COLS]
    fund = fund.merge(cmr_reg, on='eid', how='left')

    # Sample
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(fund), size=min(n, len(fund)), replace=False)
    fund = fund.iloc[sorted(idx)].reset_index(drop=True)

    print(f'[cohort] Fundus: {len(fund)} subjects')
    for col in ['prevalent_I21', 'composite_ischemic_hd',
                'LV ejection fraction', 'LV end diastolic volume']:
        if col not in fund.columns:
            continue
        nn = fund[col].notna().sum()
        if fund[col].dtype != object and fund[col].nunique() <= 2:
            print(f'  {col}: {int(fund[col].sum())} pos / {nn} non-null')
        else:
            print(f'  {col}: {nn} non-null')
    return fund


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

FUNDUS_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


class FundusDataset(Dataset):
    def __init__(self, df: pd.DataFrame, label_col: str):
        valid = df[label_col].notna()
        self.paths  = df.loc[valid, 'image_path'].tolist()
        self.labels = df.loc[valid, label_col].astype(float).tolist()

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return FUNDUS_TRANSFORM(img), torch.tensor(self.labels[i], dtype=torch.float32)


class CMRDataset(Dataset):
    """frame_slice: None=all4, (0,2)=LAX, (2,4)=SAX."""
    def __init__(self, df: pd.DataFrame, label_col: str, frame_slice=None):
        valid = df[label_col].notna()
        self.eids   = df.loc[valid, 'eid'].astype(int).tolist()
        self.labels = df.loc[valid, label_col].astype(float).tolist()
        self.fslice = frame_slice

    def __len__(self): return len(self.eids)

    def __getitem__(self, i):
        npy = np.load(os.path.join(CMR_NPY_DIR, f'{self.eids[i]}.npy'))  # (4,224,224)
        if self.fslice:
            npy = npy[self.fslice[0]:self.fslice[1]]
        x = torch.from_numpy(npy.astype(np.float32))  # (F,224,224)
        x = x.unsqueeze(1)                             # (F,1,224,224)
        x = (x - 0.5) / 0.5
        return x, torch.tensor(self.labels[i], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Backbone builders
# ---------------------------------------------------------------------------

def build_medsam_backbone(device, freeze: bool = True):
    from models_cmr import CMREncoder
    enc = CMREncoder(proj_dim=256, num_frames=4, embed_dim=768,
                     img_size=224, medsam_ckpt=MEDSAM_CKPT, freeze_backbone=freeze)
    enc.backbone.eval() if freeze else enc.backbone.train()
    return enc.backbone.to(device)


def build_retfound_backbone(device, freeze: bool = True):
    from models_contrast import FundusContrastModel
    m = FundusContrastModel(proj_dim=256, drop_path_rate=0.0)
    m.load_pretrained(RETFOUND_CKPT)
    m.backbone.eval() if freeze else m.backbone.train()
    for p in m.backbone.parameters():
        p.requires_grad_(not freeze)
    return m.backbone.to(device)


# ---------------------------------------------------------------------------
# Feature extraction (linear probe)
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_cmr_features(backbone, loader, device):
    feats, labels = [], []
    for x, lbs in loader:
        x = x.to(device)
        B, F, C, H, W = x.shape
        x3 = x.expand(-1, -1, 3, -1, -1).reshape(B * F, 3, H, W)
        feat = backbone(x3)            # (B*F, 14, 14, 768)
        feat = feat.mean(dim=(1, 2))   # (B*F, 768)
        feat = feat.reshape(B, F, 768).mean(dim=1)  # (B, 768)
        feats.append(feat.cpu().numpy())
        labels.append(lbs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


@torch.no_grad()
def extract_fundus_features(backbone, loader, device):
    feats, labels = [], []
    for imgs, lbs in loader:
        imgs = imgs.to(device)
        feat = backbone.forward_features(imgs)  # (B, 1024)
        feats.append(feat.cpu().numpy())
        labels.append(lbs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Scatter plot for regression
# ---------------------------------------------------------------------------

def scatter_plot(y_true: np.ndarray, y_pred: np.ndarray,
                 title: str, out_path: str):
    pear, _ = pearsonr(y_true, y_pred)
    spear, _ = spearmanr(y_true, y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-9)

    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.scatter(y_true, y_pred, alpha=0.35, s=12, c='#2196F3', edgecolors='none')
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5)
    ax.set_xlabel('Ground Truth', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title(title, fontsize=12, fontweight='bold')
    txt = f'Pearson r = {pear:.3f}\nSpearman ρ = {spear:.3f}\nR² = {r2:.3f}\nn = {len(y_true)}'
    ax.text(0.05, 0.95, txt, transform=ax.transAxes,
            va='top', ha='left', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    return {'pearson': float(pear), 'spearman': float(spear), 'r2': float(r2)}


# ---------------------------------------------------------------------------
# Linear probe
# ---------------------------------------------------------------------------

def run_linear_probe(X_tr, y_tr, X_val, y_val, X_te, y_te,
                     task_type, task_name, out_dir):
    results = {}
    if task_type == 'binary':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced',
                                       max_iter=2000, C=1.0, solver='lbfgs'))
        ])
        pipe.fit(X_tr, y_tr)
        for sname, X, y in [('val', X_val, y_val), ('test', X_te, y_te)]:
            prob = pipe.predict_proba(X)[:, 1]
            pred = pipe.predict(X)
            auc  = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float('nan')
            bacc = balanced_accuracy_score(y, pred)
            f1   = f1_score(y, pred, zero_division=0)
            results[sname] = {'auc': float(auc), 'bacc': float(bacc), 'f1': float(f1),
                               'n': int(len(y)), 'pos': int(y.sum())}
            print(f'  [{sname}] AUC={auc:.3f}  bACC={bacc:.3f}  F1={f1:.3f}  '
                  f'n={len(y)}  pos={int(y.sum())}', flush=True)
    else:
        pipe = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))])
        pipe.fit(X_tr, y_tr)
        for sname, X, y in [('val', X_val, y_val), ('test', X_te, y_te)]:
            pred = pipe.predict(X)
            ss_res = ((y - pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2  = float(1 - ss_res / (ss_tot + 1e-9))
            mae = float(np.abs(y - pred).mean())
            pear, _  = pearsonr(y, pred)
            spear, _ = spearmanr(y, pred)
            results[sname] = {'r2': r2, 'mae': mae, 'pearson': float(pear),
                               'spearman': float(spear), 'n': int(len(y))}
            print(f'  [{sname}] R²={r2:.3f}  MAE={mae:.2f}  '
                  f'Pearson={pear:.3f}  Spearman={spear:.3f}  n={len(y)}', flush=True)
            # scatter plot
            label = TASK_META[task_name]['label']
            scatter_plot(y, pred,
                         title=f'Linear Probe — {label} [{sname}]',
                         out_path=os.path.join(out_dir, f'scatter_{sname}.png'))
    return results


# ---------------------------------------------------------------------------
# Fine-tune models
# ---------------------------------------------------------------------------

class CMRFineTuneModel(nn.Module):
    def __init__(self, backbone, out_dim: int = 1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(768, out_dim)

    def forward(self, x):
        B, F, C, H, W = x.shape
        x3 = x.expand(-1, -1, 3, -1, -1).reshape(B * F, 3, H, W)
        feat = self.backbone(x3)           # (B*F, 14, 14, 768)
        feat = feat.mean(dim=(1, 2))       # (B*F, 768)
        feat = feat.reshape(B, F, 768).mean(dim=1)  # (B, 768)
        return self.head(feat).squeeze(-1)


class FundusFineTuneModel(nn.Module):
    def __init__(self, backbone, out_dim: int = 1):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(1024, out_dim)

    def forward(self, x):
        feat = self.backbone.forward_features(x)
        return self.head(feat).squeeze(-1)


@torch.no_grad()
def _eval_model(model, loader, task_type, device):
    model.eval()
    preds, labels = [], []
    for x, lbs in loader:
        out = model(x.to(device))
        preds.extend(out.cpu().tolist())
        labels.extend(lbs.tolist())
    preds = np.array(preds); labels = np.array(labels)
    if task_type == 'binary':
        probs = 1 / (1 + np.exp(-preds))
        pred_cls = (probs > 0.5).astype(int)
        auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')
        bacc = balanced_accuracy_score(labels, pred_cls)
        f1   = f1_score(labels, pred_cls, zero_division=0)
        return {'auc': float(auc), 'bacc': float(bacc), 'f1': float(f1),
                'n': int(len(labels)), 'pos': int(labels.sum())}, preds, labels
    else:
        ss_res = ((labels - preds) ** 2).sum()
        ss_tot = ((labels - labels.mean()) ** 2).sum()
        r2  = float(1 - ss_res / (ss_tot + 1e-9))
        mae = float(np.abs(labels - preds).mean())
        pear, _  = pearsonr(labels, preds)
        spear, _ = spearmanr(labels, preds)
        return {'r2': r2, 'mae': mae, 'pearson': float(pear),
                'spearman': float(spear), 'n': int(len(labels))}, preds, labels


def _fmt(m):
    if 'auc' in m:
        return f"AUC={m['auc']:.3f} bACC={m['bacc']:.3f} F1={m['f1']:.3f}"
    return f"R²={m['r2']:.3f} MAE={m['mae']:.2f} Pearson={m['pearson']:.3f} Spearman={m['spearman']:.3f}"


def run_finetune(model, ds_tr, ds_val, ds_te, task_type, task_name, args, device, out_dir):
    pos_count = int(sum(ds_tr.labels))
    neg_count = len(ds_tr.labels) - pos_count

    def make_loader(ds, shuffle):
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=4, drop_last=shuffle, pin_memory=True)

    loader_tr  = make_loader(ds_tr,  True)
    loader_val = make_loader(ds_val, False)
    loader_te  = make_loader(ds_te,  False)

    if task_type == 'binary':
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
    head_params     = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr, 'weight_decay': 0.05},
        {'params': head_params,     'lr': args.head_lr,     'weight_decay': 0.01},
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val     = -1e9 if task_type == 'binary' else float('inf')
    best_epoch   = 0
    patience_cnt = 0
    t0 = time.time()
    best_path = os.path.join(out_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = []
        for x, lbs in loader_tr:
            x = x.to(device); lbs = lbs.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                out   = model(x)
                loss  = criterion(out, lbs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            ep_loss.append(loss.item())

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_m, _, _ = _eval_model(model, loader_val, task_type, device)
            print(f'  ep{epoch:3d} loss={np.mean(ep_loss):.4f}  val={_fmt(val_m)}'
                  f'  t={( time.time()-t0)/60:.1f}min', flush=True)

            if task_type == 'binary':
                primary  = val_m['auc']
                improved = primary > best_val
            else:
                primary  = val_m['mae']
                improved = primary < best_val

            if improved:
                best_val = primary; best_epoch = epoch; patience_cnt = 0
                torch.save(model.state_dict(), best_path)
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  Early stop at ep{epoch} (best ep{best_epoch})', flush=True)
                    break

    if not os.path.exists(best_path):
        torch.save(model.state_dict(), best_path)

    model.load_state_dict(torch.load(best_path, map_location=device))
    results = {}
    label = TASK_META[task_name]['label']
    for sname, loader in [('val', loader_val), ('test', loader_te)]:
        m, preds, labels = _eval_model(model, loader, task_type, device)
        results[sname] = m
        print(f'  [{sname}(best)] {_fmt(m)}', flush=True)
        if task_type == 'regression':
            scatter_plot(labels, preds,
                         title=f'Finetune — {label} [{sname}]',
                         out_path=os.path.join(out_dir, f'scatter_{sname}.png'))
    results['best_epoch'] = best_epoch
    return results


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def split_df(df):
    tr  = df[df['split'] == 'train'].reset_index(drop=True)
    val = df[df['split'] == 'val'].reset_index(drop=True)
    te  = df[df['split'] == 'test'].reset_index(drop=True)
    return tr, val, te


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder',   choices=['medsam', 'retfound'], required=True)
    ap.add_argument('--modality',  choices=['cmr_all', 'cmr_lax', 'cmr_sax', 'fundus'], required=True)
    ap.add_argument('--mode',      choices=['linear', 'finetune'], required=True)
    ap.add_argument('--task',      choices=['i21', 'isch_hd', 'lvef', 'lvedv', 'i25', 'i20'], required=True)
    ap.add_argument('--out_dir',   required=True)
    # finetune hyperparams
    ap.add_argument('--epochs',      type=int,   default=50)
    ap.add_argument('--batch_size',  type=int,   default=16)
    ap.add_argument('--backbone_lr', type=float, default=5e-6)
    ap.add_argument('--head_lr',     type=float, default=1e-3)
    ap.add_argument('--eval_every',  type=int,   default=5)
    ap.add_argument('--patience',    type=int,   default=15)
    ap.add_argument('--amp',         action='store_true', default=True)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tmeta = TASK_META[args.task]
    label_col = tmeta['col']
    task_type = tmeta['type']

    print(f'[ceiling] encoder={args.encoder} modality={args.modality} '
          f'mode={args.mode} task={args.task} device={device}', flush=True)

    # ------------------------------------------------------------------
    # Build cohort & datasets
    # ------------------------------------------------------------------
    frame_slice = {'cmr_all': None, 'cmr_lax': (0, 2), 'cmr_sax': (2, 4)}.get(args.modality)

    if args.modality in ('cmr_all', 'cmr_lax', 'cmr_sax'):
        assert args.encoder == 'medsam', 'CMR modality requires medsam encoder'
        df = build_cmr_cohort()
        df_tr, df_val, df_te = split_df(df)

        print(f'  splits: train={len(df_tr)} val={len(df_val)} test={len(df_te)}')
        for sname, sdf in [('train', df_tr), ('val', df_val), ('test', df_te)]:
            nn = sdf[label_col].notna().sum()
            if task_type == 'binary':
                print(f'    {sname}: {int(sdf[label_col].sum())} pos / {nn} total')
            else:
                print(f'    {sname}: {nn} with label')

        if args.mode == 'linear':
            backbone = build_medsam_backbone(device, freeze=True)
            ds_tr  = CMRDataset(df_tr,  label_col, frame_slice)
            ds_val = CMRDataset(df_val, label_col, frame_slice)
            ds_te  = CMRDataset(df_te,  label_col, frame_slice)

            def mk_loader(ds):
                return DataLoader(ds, batch_size=32, shuffle=False,
                                  num_workers=4, pin_memory=True)

            print('  extracting train features...', flush=True)
            X_tr, y_tr   = extract_cmr_features(backbone, mk_loader(ds_tr),  device)
            print('  extracting val features...',   flush=True)
            X_val, y_val = extract_cmr_features(backbone, mk_loader(ds_val), device)
            print('  extracting test features...',  flush=True)
            X_te, y_te   = extract_cmr_features(backbone, mk_loader(ds_te),  device)
            print(f'  features: train={X_tr.shape} val={X_val.shape} test={X_te.shape}', flush=True)

            results = run_linear_probe(X_tr, y_tr, X_val, y_val, X_te, y_te,
                                       task_type, args.task, args.out_dir)

        else:  # finetune
            backbone = build_medsam_backbone(device, freeze=False)
            model    = CMRFineTuneModel(backbone).to(device)
            ds_tr  = CMRDataset(df_tr,  label_col, frame_slice)
            ds_val = CMRDataset(df_val, label_col, frame_slice)
            ds_te  = CMRDataset(df_te,  label_col, frame_slice)
            results = run_finetune(model, ds_tr, ds_val, ds_te,
                                   task_type, args.task, args, device, args.out_dir)

    else:  # fundus
        assert args.encoder == 'retfound', 'Fundus modality requires retfound encoder'
        df = build_fundus_cohort(n=10000, seed=42)
        df_tr, df_val, df_te = split_df(df)

        print(f'  splits: train={len(df_tr)} val={len(df_val)} test={len(df_te)}')
        for sname, sdf in [('train', df_tr), ('val', df_val), ('test', df_te)]:
            nn = sdf[label_col].notna().sum()
            if task_type == 'binary':
                print(f'    {sname}: {int(sdf[label_col].sum())} pos / {nn} total')
            else:
                print(f'    {sname}: {nn} with label')

        if args.mode == 'linear':
            backbone = build_retfound_backbone(device, freeze=True)
            ds_tr  = FundusDataset(df_tr,  label_col)
            ds_val = FundusDataset(df_val, label_col)
            ds_te  = FundusDataset(df_te,  label_col)

            def mk_loader(ds):
                return DataLoader(ds, batch_size=32, shuffle=False,
                                  num_workers=4, pin_memory=True)

            print('  extracting train features...', flush=True)
            X_tr, y_tr   = extract_fundus_features(backbone, mk_loader(ds_tr),  device)
            print('  extracting val features...',   flush=True)
            X_val, y_val = extract_fundus_features(backbone, mk_loader(ds_val), device)
            print('  extracting test features...',  flush=True)
            X_te, y_te   = extract_fundus_features(backbone, mk_loader(ds_te),  device)
            print(f'  features: train={X_tr.shape} val={X_val.shape} test={X_te.shape}', flush=True)

            results = run_linear_probe(X_tr, y_tr, X_val, y_val, X_te, y_te,
                                       task_type, args.task, args.out_dir)

        else:  # finetune
            backbone = build_retfound_backbone(device, freeze=False)
            model    = FundusFineTuneModel(backbone).to(device)
            ds_tr  = FundusDataset(df_tr,  label_col)
            ds_val = FundusDataset(df_val, label_col)
            ds_te  = FundusDataset(df_te,  label_col)
            results = run_finetune(model, ds_tr, ds_val, ds_te,
                                   task_type, args.task, args, device, args.out_dir)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    out_json = os.path.join(args.out_dir, 'results.json')
    with open(out_json, 'w') as f:
        json.dump({'encoder': args.encoder, 'modality': args.modality,
                   'mode': args.mode, 'task': args.task, 'results': results}, f, indent=2)
    print(f'[saved] {out_json}', flush=True)


if __name__ == '__main__':
    main()
