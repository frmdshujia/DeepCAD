"""linear_probe_downstream_eval.py
实验一 & 实验二诊断：下游任务性能上限评估

支持：
  --encoder  retfound | medsam
  --modality fundus | cmr_all | cmr_lax | cmr_sax
  --mode     linear | finetune
  --task     composite_ischemic_hd | prevalent_I21 | composite_af_arrhythmia |
             lvef | lvedv

流程：
  linear: 冻结 encoder → 提取特征 → sklearn 分类/回归
  finetune: encoder 全参解冻 → PyTorch 端到端训练分类/回归头

Usage:
  CUDA_VISIBLE_DEVICES=0 python contrastive_pretrain/linear_probe_downstream_eval.py \
      --encoder retfound --modality fundus --mode linear \
      --task composite_ischemic_hd --out_dir output_dir/probe_retfound_fundus
"""
from __future__ import annotations
import sys, os, argparse, json, time, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

ImageFile.LOAD_TRUNCATED_IMAGES = True

# sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, f1_score
from scipy.stats import pearsonr

MEDSAM_CKPT = ('pretrained_weights/hf_cache/models--flaviagiammarino--medsam-vit-base'
               '/snapshots/a3cb4c518fc3ae8beaf1af0cd3a43867cfa28335/pytorch_model.bin')
RETFOUND_CKPT = 'RETFound_cfp_weights.pth'
CMR_NPY_DIR   = '/data/home/shujia/UKB/CMRI/preprocessed_lax_sax'
FUNDUS_CSV    = 'contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv'
CMR_CSV       = 'contrastive_pretrain/preprocessed_data/modeling_delivery/cmr_table.csv'
MASTER_CSV    = 'contrastive_pretrain/raw/master_long.csv'

TASK_META = {
    'composite_ischemic_hd': {'type': 'binary', 'col': 'composite_ischemic_hd'},
    'prevalent_I21':         {'type': 'binary', 'col': 'prevalent_I21'},
    'composite_af_arrhythmia': {'type': 'binary', 'col': 'composite_af_arrhythmia'},
    'lvef': {'type': 'regression',
             'col': 'LV ejection fraction | Instance 2(participant - p24103_i2)'},
    'lvedv': {'type': 'regression',
              'col': 'LV end diastolic volume | Instance 2(participant - p24100_i2)'},
}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_master_labels(task_col: str) -> pd.DataFrame:
    """Return eid-level DataFrame with [eid, label]."""
    master = pd.read_csv(MASTER_CSV, low_memory=False)
    # take first non-null label per EID across instances
    dedup = master.sort_values('instance').groupby('eid').first().reset_index()
    return dedup[['eid', task_col]].dropna(subset=[task_col]).copy()


def build_fundus_df(task_col: str) -> pd.DataFrame:
    """One fundus image per EID per split (first occurrence), joined with label."""
    fund = pd.read_csv(FUNDUS_CSV)
    labels = load_master_labels(task_col)
    # keep rows with existing image
    fund = fund[fund['fundus_image_path'].apply(os.path.exists)]
    # one row per EID (first image per EID, consistent ordering)
    fund = fund.sort_values(['eid', 'fundus_image_path']).groupby('eid').first().reset_index()
    merged = fund.merge(labels.rename(columns={task_col: 'label'}), on='eid', how='inner')
    return merged[['eid', 'fundus_image_path', 'split', 'label']].dropna()


def build_cmr_df(task_col: str) -> pd.DataFrame:
    """One CMR npy per EID with label."""
    cmr = pd.read_csv(CMR_CSV)
    cmr = cmr[cmr['instance'] == 2][['eid', 'split']].drop_duplicates('eid')
    labels = load_master_labels(task_col)
    merged = cmr.merge(labels.rename(columns={task_col: 'label'}), on='eid', how='inner')
    # filter to npy exists
    merged = merged[merged['eid'].apply(
        lambda e: os.path.exists(os.path.join(CMR_NPY_DIR, f'{int(e)}.npy'))
    )].dropna()
    return merged[['eid', 'split', 'label']].copy()


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

FUNDUS_TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


class FundusLabelDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.paths  = df['fundus_image_path'].tolist()
        self.labels = df['label'].astype(float).tolist()

    def __len__(self): return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert('RGB')
        return FUNDUS_TRANSFORM(img), torch.tensor(self.labels[i], dtype=torch.float32)


class CMRLabelDataset(Dataset):
    """Returns (cmr_frames, label). frame_slice: None=all4, (0,2)=LAX, (2,4)=SAX."""
    def __init__(self, df: pd.DataFrame, frame_slice=None):
        self.eids   = df['eid'].astype(int).tolist()
        self.labels = df['label'].astype(float).tolist()
        self.fslice = frame_slice  # tuple (start, end) or None

    def __len__(self): return len(self.eids)

    def __getitem__(self, i):
        npy = np.load(os.path.join(CMR_NPY_DIR, f'{self.eids[i]}.npy'))  # (4,224,224)
        if self.fslice is not None:
            npy = npy[self.fslice[0]:self.fslice[1]]
        cmr = torch.from_numpy(npy.astype(np.float32))   # (F,224,224)
        cmr = cmr.unsqueeze(1)                            # (F,1,224,224)
        cmr = (cmr - 0.5) / 0.5
        return cmr, torch.tensor(self.labels[i], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Feature extractors (no grad)
# ---------------------------------------------------------------------------

def build_retfound_extractor(device):
    from models_contrast import FundusContrastModel
    model = FundusContrastModel(proj_dim=256, drop_path_rate=0.0)
    model.load_pretrained(RETFOUND_CKPT)
    model.backbone.eval()
    for p in model.backbone.parameters(): p.requires_grad_(False)
    return model.backbone.to(device)


@torch.no_grad()
def extract_fundus_features(backbone, loader, device):
    feats, labels = [], []
    for imgs, lbs in loader:
        imgs = imgs.to(device)
        f = backbone.forward_features(imgs)  # (B, 1024)
        feats.append(f.cpu().numpy())
        labels.append(lbs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


def build_medsam_extractor(device):
    from models_cmr import CMREncoder
    enc = CMREncoder(proj_dim=256, num_frames=4, embed_dim=768,
                     img_size=224, medsam_ckpt=MEDSAM_CKPT, freeze_backbone=True)
    enc.backbone.eval()
    return enc.backbone.to(device)


@torch.no_grad()
def extract_cmr_features(backbone, loader, device, frame_slice=None):
    """
    backbone: SAM ViT-B; returns (N, 768) by spatial GAP then temporal mean.
    frame_slice: (0,2) for LAX, (2,4) for SAX, None for all 4.
    """
    feats, labels = [], []
    for cmr, lbs in loader:
        cmr = cmr.to(device)   # (B, F, 1, 224, 224)
        B, F, C, H, W = cmr.shape
        # Expand grayscale → 3ch, reshape to (B*F, 3, H, W)
        x = cmr.expand(-1, -1, 3, -1, -1).reshape(B * F, 3, H, W)
        feat = backbone(x)           # (B*F, 14, 14, 768)
        feat = feat.mean(dim=(1, 2)) # (B*F, 768) spatial GAP
        feat = feat.reshape(B, F, 768).mean(dim=1)  # (B, 768) temporal mean
        feats.append(feat.cpu().numpy())
        labels.append(lbs.numpy())
    return np.concatenate(feats), np.concatenate(labels)


# ---------------------------------------------------------------------------
# Sklearn linear probe
# ---------------------------------------------------------------------------

def run_linear_probe(X_tr, y_tr, X_val, y_val, X_te, y_te, task_type, task_name):
    results = {}
    if task_type == 'binary':
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                class_weight='balanced', max_iter=2000, C=1.0, solver='lbfgs'))
        ])
        pipe.fit(X_tr, y_tr)
        for split_name, X, y in [('val', X_val, y_val), ('test', X_te, y_te)]:
            prob = pipe.predict_proba(X)[:, 1]
            pred = pipe.predict(X)
            auc  = roc_auc_score(y, prob) if len(np.unique(y)) > 1 else float('nan')
            bacc = balanced_accuracy_score(y, pred)
            f1   = f1_score(y, pred, zero_division=0)
            results[split_name] = {'auc': auc, 'bacc': bacc, 'f1': f1,
                                   'n': len(y), 'pos': int(y.sum())}
            print(f'  [{split_name}] AUC={auc:.3f}  bACC={bacc:.3f}  F1={f1:.3f}  '
                  f'n={len(y)}  pos={int(y.sum())}')
    else:
        pipe = Pipeline([
            ('scaler', StandardScaler()),
            ('reg', Ridge(alpha=1.0))
        ])
        pipe.fit(X_tr, y_tr)
        for split_name, X, y in [('val', X_val, y_val), ('test', X_te, y_te)]:
            pred = pipe.predict(X)
            ss_res = ((y - pred) ** 2).sum()
            ss_tot = ((y - y.mean()) ** 2).sum()
            r2  = 1 - ss_res / (ss_tot + 1e-9)
            mae = np.abs(y - pred).mean()
            rho = pearsonr(y, pred)[0] if len(y) > 2 else float('nan')
            results[split_name] = {'r2': r2, 'mae': float(mae), 'pearson': float(rho),
                                   'n': len(y)}
            print(f'  [{split_name}] R²={r2:.3f}  MAE={mae:.2f}  Pearson={rho:.3f}  n={len(y)}')
    return results


# ---------------------------------------------------------------------------
# Full fine-tune (PyTorch)
# ---------------------------------------------------------------------------

class FundusFineTuneModel(nn.Module):
    def __init__(self, backbone, out_dim: int = 1):
        super().__init__()
        self.backbone = backbone  # RETFound ViT-L
        self.head = nn.Linear(1024, out_dim)
    def forward(self, x):
        feat = self.backbone.forward_features(x)
        return self.head(feat).squeeze(-1)


class CMRFineTuneModel(nn.Module):
    def __init__(self, backbone, out_dim: int = 1):
        super().__init__()
        self.backbone = backbone  # SAM ViT-B
        self.head = nn.Linear(768, out_dim)
    def forward(self, x):
        B, F, C, H, W = x.shape
        x3 = x.expand(-1, -1, 3, -1, -1).reshape(B * F, 3, H, W)
        feat = self.backbone(x3)           # (B*F, 14, 14, 768)
        feat = feat.mean(dim=(1, 2))       # (B*F, 768)
        feat = feat.reshape(B, F, 768).mean(dim=1)  # (B, 768)
        return self.head(feat).squeeze(-1)


def run_finetune(model, ds_tr, ds_val, ds_te, task_type, args, device):
    pos_count = int(sum(ds_tr.labels))
    neg_count = len(ds_tr.labels) - pos_count

    loader_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,
                           num_workers=4, drop_last=True, pin_memory=True)
    loader_val = DataLoader(ds_val, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)
    loader_te  = DataLoader(ds_te,  batch_size=args.batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Loss
    if task_type == 'binary':
        pos_weight = torch.tensor([neg_count / max(pos_count, 1)], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.MSELoss()

    # Optimizer: small LR for backbone, larger for head
    backbone_params = [p for n, p in model.named_parameters() if 'head' not in n]
    head_params     = list(model.head.parameters())
    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': args.backbone_lr, 'weight_decay': 0.05},
        {'params': head_params,     'lr': args.head_lr,     'weight_decay': 0.01},
    ])
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_val = -1e9 if task_type == 'binary' else float('inf')
    best_epoch = 0
    patience_cnt = 0
    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss = []
        for imgs, lbs in loader_tr:
            imgs = imgs.to(device); lbs = lbs.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=args.amp):
                out = model(imgs)
                loss = criterion(out, lbs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt); scaler.update()
            ep_loss.append(loss.item())

        if epoch % args.eval_every == 0 or epoch == args.epochs:
            val_metrics = _eval_finetune(model, loader_val, task_type, device)
            print(f'  ep{epoch:3d} loss={np.mean(ep_loss):.4f}  '
                  f'val={_fmt_metrics(val_metrics)}  '
                  f't={( time.time()-t0)/60:.1f}min', flush=True)
            history.append({'epoch': epoch, 'loss': float(np.mean(ep_loss)),
                            'val': val_metrics})

            # Early stopping
            if task_type == 'binary':
                primary = val_metrics.get('auc', 0.0)
                improved = primary > best_val
            else:
                primary = val_metrics.get('mae', float('inf'))
                improved = primary < best_val
            if improved:
                best_val = primary; best_epoch = epoch; patience_cnt = 0
                torch.save(model.state_dict(),
                           os.path.join(args.out_dir, 'best_model.pth'))
            else:
                patience_cnt += 1
                if patience_cnt >= args.patience:
                    print(f'  Early stop at ep{epoch} (best ep{best_epoch})')
                    break

    # Load best and eval test
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'best_model.pth'),
                                     map_location=device))
    val_final = _eval_finetune(model, loader_val, task_type, device)
    test_final = _eval_finetune(model, loader_te,  task_type, device)
    print(f'  [val(best)]  {_fmt_metrics(val_final)}')
    print(f'  [test]       {_fmt_metrics(test_final)}')
    return {'val': val_final, 'test': test_final, 'best_epoch': best_epoch, 'history': history}


@torch.no_grad()
def _eval_finetune(model, loader, task_type, device):
    model.eval()
    preds, labels = [], []
    for imgs, lbs in loader:
        out = model(imgs.to(device))
        preds.extend(out.cpu().tolist())
        labels.extend(lbs.tolist())
    preds = np.array(preds); labels = np.array(labels)
    if task_type == 'binary':
        probs = 1 / (1 + np.exp(-preds))  # sigmoid
        pred_cls = (probs > 0.5).astype(int)
        auc  = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else float('nan')
        bacc = balanced_accuracy_score(labels, pred_cls)
        return {'auc': float(auc), 'bacc': float(bacc), 'n': len(labels),
                'pos': int(labels.sum())}
    else:
        ss_res = ((labels - preds)**2).sum()
        ss_tot = ((labels - labels.mean())**2).sum()
        r2  = float(1 - ss_res / (ss_tot + 1e-9))
        mae = float(np.abs(labels - preds).mean())
        rho = float(pearsonr(labels, preds)[0]) if len(labels) > 2 else float('nan')
        return {'r2': r2, 'mae': mae, 'pearson': rho, 'n': len(labels)}
    model.train()


def _fmt_metrics(m):
    if 'auc' in m: return f"AUC={m['auc']:.3f} bACC={m['bacc']:.3f}"
    return f"R²={m['r2']:.3f} MAE={m['mae']:.2f} r={m['pearson']:.3f}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--encoder',  choices=['retfound', 'medsam'], required=True)
    ap.add_argument('--modality', choices=['fundus', 'cmr_all', 'cmr_lax', 'cmr_sax'],
                    required=True)
    ap.add_argument('--mode',     choices=['linear', 'finetune'], default='linear')
    ap.add_argument('--task',     choices=list(TASK_META.keys()), required=True)
    ap.add_argument('--out_dir',  required=True)
    # finetune args
    ap.add_argument('--epochs',      type=int,   default=30)
    ap.add_argument('--batch_size',  type=int,   default=16)
    ap.add_argument('--backbone_lr', type=float, default=1e-5)
    ap.add_argument('--head_lr',     type=float, default=1e-3)
    ap.add_argument('--eval_every',  type=int,   default=3)
    ap.add_argument('--patience',    type=int,   default=10)
    ap.add_argument('--amp',         action='store_true')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tmeta  = TASK_META[args.task]
    task_type = tmeta['type']
    task_col  = tmeta['col']

    print(f'[probe] encoder={args.encoder} modality={args.modality} '
          f'mode={args.mode} task={args.task} device={device}')

    # ---- Build dataframes ----
    if args.modality == 'fundus':
        df = build_fundus_df(task_col)
    else:
        df = build_cmr_df(task_col)

    df_tr  = df[df['split'] == 'train'].reset_index(drop=True)
    df_val = df[df['split'] == 'val'].reset_index(drop=True)
    df_te  = df[df['split'] == 'test'].reset_index(drop=True)

    print(f'[probe] train={len(df_tr)} val={len(df_val)} test={len(df_te)}')
    if task_type == 'binary':
        for name, d in [('train', df_tr), ('val', df_val), ('test', df_te)]:
            pos = int(d['label'].sum()); n = len(d)
            print(f'  {name}: pos={pos} neg={n-pos} pos%={pos/max(n,1)*100:.1f}%')

    # ---- Frame slice for CMR modality ----
    frame_slice = None
    if args.modality == 'cmr_lax': frame_slice = (0, 2)
    if args.modality == 'cmr_sax': frame_slice = (2, 4)

    # ---- Linear probe ----
    if args.mode == 'linear':
        if args.modality == 'fundus':
            loader = lambda d: DataLoader(FundusLabelDataset(d), batch_size=32,
                                          shuffle=False, num_workers=4, pin_memory=True)
            backbone = build_retfound_extractor(device)
            print('[probe] extracting fundus features...')
            X_tr,  y_tr  = extract_fundus_features(backbone, loader(df_tr),  device)
            X_val, y_val = extract_fundus_features(backbone, loader(df_val), device)
            X_te,  y_te  = extract_fundus_features(backbone, loader(df_te),  device)
        else:
            loader = lambda d: DataLoader(CMRLabelDataset(d, frame_slice), batch_size=16,
                                          shuffle=False, num_workers=4, pin_memory=True)
            backbone = build_medsam_extractor(device)
            print('[probe] extracting CMR features...')
            X_tr,  y_tr  = extract_cmr_features(backbone, loader(df_tr),  device, frame_slice)
            X_val, y_val = extract_cmr_features(backbone, loader(df_val), device, frame_slice)
            X_te,  y_te  = extract_cmr_features(backbone, loader(df_te),  device, frame_slice)

        print(f'[probe] features: train={X_tr.shape} val={X_val.shape} test={X_te.shape}')
        results = run_linear_probe(X_tr, y_tr, X_val, y_val, X_te, y_te, task_type, args.task)

    # ---- Full fine-tune ----
    else:
        if args.modality == 'fundus':
            backbone = build_retfound_extractor(device)
            for p in backbone.parameters(): p.requires_grad_(True)
            model = FundusFineTuneModel(backbone, out_dim=1).to(device)
            ds = lambda d: FundusLabelDataset(d)
        else:
            backbone = build_medsam_extractor(device)
            for p in backbone.parameters(): p.requires_grad_(True)
            model = CMRFineTuneModel(backbone, out_dim=1).to(device)
            ds = lambda d: CMRLabelDataset(d, frame_slice)

        print(f'[probe] finetune: trainable params = '
              f'{sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.1f}M')
        results = run_finetune(model, ds(df_tr), ds(df_val), ds(df_te),
                               task_type, args, device)

    # ---- Save ----
    out = {
        'encoder': args.encoder, 'modality': args.modality,
        'mode': args.mode, 'task': args.task, 'task_type': task_type,
        'n_train': len(df_tr), 'n_val': len(df_val), 'n_test': len(df_te),
        'results': results,
    }
    out_path = os.path.join(args.out_dir, 'results.json')
    with open(out_path, 'w') as f: json.dump(out, f, indent=2)
    print(f'[probe] saved → {out_path}')


if __name__ == '__main__':
    main()
