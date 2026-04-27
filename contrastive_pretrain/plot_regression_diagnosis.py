"""plot_regression_diagnosis.py
诊断 LVEF / LVEDV 回归：GT 分布、预测 vs GT 散点、异常值分析。

Usage:
  CUDA_VISIBLE_DEVICES=0 python contrastive_pretrain/plot_regression_diagnosis.py \
      --out_dir output_dir/regression_diagnosis
"""
from __future__ import annotations
import sys, os, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from linear_probe_downstream_eval import (
    build_cmr_df, CMRLabelDataset, build_medsam_extractor,
    extract_cmr_features, TASK_META
)

plt.rcParams.update({'font.size': 11, 'figure.dpi': 120})


def iqr_bounds(arr, k=3.0):
    """Return (lower, upper) fence at k * IQR."""
    q1, q3 = np.percentile(arr, 25), np.percentile(arr, 75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr


def make_scatter_panel(ax, y_true, y_pred, title, color, mask_out=None):
    """Scatter plot with identity line and stats."""
    if mask_out is not None:
        ax.scatter(y_true[mask_out], y_pred[mask_out], c='red', alpha=0.6,
                   s=20, label=f'outliers (n={mask_out.sum()})', zorder=3)
        m = ~mask_out
    else:
        m = np.ones(len(y_true), dtype=bool)

    ax.scatter(y_true[m], y_pred[m], c=color, alpha=0.4, s=15, label=f'normal (n={m.sum()})')

    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    ax.plot([lo, hi], [lo, hi], 'k--', lw=1, alpha=0.5, label='identity')

    r_all,  _ = pearsonr(y_true, y_pred)
    r_clean, _ = pearsonr(y_true[m], y_pred[m]) if m.sum() > 2 else (float('nan'), None)
    ss_res = ((y_true[m] - y_pred[m]) ** 2).sum()
    ss_tot = ((y_true[m] - y_true[m].mean()) ** 2).sum()
    r2_clean = 1 - ss_res / (ss_tot + 1e-9)

    ax.set_xlabel('GT'); ax.set_ylabel('Predicted')
    ax.set_title(f'{title}\nPearson(all)={r_all:.3f}  Pearson(clean)={r_clean:.3f}  R²(clean)={r2_clean:.3f}')
    ax.legend(fontsize=8)
    return r_all, r_clean, r2_clean


def plot_task(task_name, X_tr, y_tr, X_val, y_val, X_te, y_te,
              out_dir, device):
    label = 'LVEF (%)' if task_name == 'lvef' else 'LVEDV (mL)'
    color = '#2196F3' if task_name == 'lvef' else '#4CAF50'

    # Fit Ridge on train
    pipe = Pipeline([('scaler', StandardScaler()), ('reg', Ridge(alpha=1.0))])
    pipe.fit(X_tr, y_tr)
    pred_val = pipe.predict(X_val)
    pred_te  = pipe.predict(X_te)

    # IQR outlier detection on GT
    lo, hi = iqr_bounds(np.concatenate([y_tr, y_val, y_te]), k=3.0)
    out_val = (y_val < lo) | (y_val > hi)
    out_te  = (y_te  < lo) | (y_te  > hi)
    print(f'\n[{task_name}] IQR fence: ({lo:.1f}, {hi:.1f})')
    print(f'  val outliers: {out_val.sum()} / {len(y_val)}  '
          f'values: {y_val[out_val].tolist()}')
    print(f'  test outliers: {out_te.sum()} / {len(y_te)}  '
          f'values: {y_te[out_te].tolist()}')

    # ---- Figure layout ----
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'MedSAM Linear Probe: {label} Diagnosis', fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.38)

    # 1. GT distribution (train)
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(y_tr, bins=40, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax1.axvline(lo, color='red', ls='--', lw=1.2, label=f'IQR fence ({lo:.0f})')
    ax1.axvline(hi, color='red', ls='--', lw=1.2, label=f'IQR fence ({hi:.0f})')
    ax1.set_title(f'GT Distribution (train, n={len(y_tr)})')
    ax1.set_xlabel(label); ax1.set_ylabel('Count')
    ax1.legend(fontsize=7)

    # 2. GT distribution (val)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(y_val, bins=20, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax2.axvline(lo, color='red', ls='--', lw=1.2)
    ax2.axvline(hi, color='red', ls='--', lw=1.2)
    ax2.set_title(f'GT Distribution (val, n={len(y_val)})')
    ax2.set_xlabel(label)

    # 3. GT distribution (test)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(y_te, bins=20, color=color, alpha=0.7, edgecolor='white', linewidth=0.5)
    ax3.axvline(lo, color='red', ls='--', lw=1.2)
    ax3.axvline(hi, color='red', ls='--', lw=1.2)
    ax3.set_title(f'GT Distribution (test, n={len(y_te)})')
    ax3.set_xlabel(label)

    # 4. Prediction distribution (val)
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.hist(pred_val, bins=20, color='#FF9800', alpha=0.7, edgecolor='white', linewidth=0.5,
             label='Predicted')
    ax4.hist(y_val,    bins=20, color=color,    alpha=0.4, edgecolor='white', linewidth=0.5,
             label='GT')
    ax4.set_title('Val: GT vs Predicted Distribution')
    ax4.set_xlabel(label); ax4.legend(fontsize=8)

    # 5. Val scatter (all points)
    ax5 = fig.add_subplot(gs[1, 0])
    make_scatter_panel(ax5, y_val, pred_val, 'Val (all points)', color, mask_out=out_val)

    # 6. Val scatter (clean)
    ax6 = fig.add_subplot(gs[1, 1])
    m_val = ~out_val
    if m_val.sum() > 2:
        r_c, _ = pearsonr(y_val[m_val], pred_val[m_val])
        ss_r = ((y_val[m_val] - pred_val[m_val])**2).sum()
        ss_t = ((y_val[m_val] - y_val[m_val].mean())**2).sum()
        r2_c = 1 - ss_r / (ss_t + 1e-9)
        ax6.scatter(y_val[m_val], pred_val[m_val], c=color, alpha=0.5, s=15)
        lo6 = min(y_val[m_val].min(), pred_val[m_val].min())
        hi6 = max(y_val[m_val].max(), pred_val[m_val].max())
        ax6.plot([lo6, hi6], [lo6, hi6], 'k--', lw=1, alpha=0.5)
        ax6.set_title(f'Val (outliers removed)\nPearson={r_c:.3f}  R²={r2_c:.3f}')
        ax6.set_xlabel('GT'); ax6.set_ylabel('Predicted')

    # 7. Test scatter (all)
    ax7 = fig.add_subplot(gs[1, 2])
    make_scatter_panel(ax7, y_te, pred_te, 'Test (all points)', color, mask_out=out_te)

    # 8. Residuals vs GT
    ax8 = fig.add_subplot(gs[1, 3])
    residuals = pred_val - y_val
    ax8.scatter(y_val, residuals, c=color, alpha=0.4, s=15)
    ax8.axhline(0, color='k', ls='--', lw=1)
    ax8.axhline(residuals.mean(), color='red', ls='-', lw=1.2,
                label=f'mean bias={residuals.mean():.1f}')
    ax8.set_xlabel('GT'); ax8.set_ylabel('Residual (pred - GT)')
    ax8.set_title(f'Val Residuals\nbias={residuals.mean():.2f}  std={residuals.std():.2f}')
    ax8.legend(fontsize=8)

    # Summary stats box
    r_all, _ = pearsonr(y_val, pred_val)
    ss_r = ((y_val - pred_val)**2).sum()
    ss_t = ((y_val - y_val.mean())**2).sum()
    r2_all = 1 - ss_r / (ss_t + 1e-9)
    fig.text(0.5, 0.01,
             f'GT stats (all splits) — mean={np.concatenate([y_tr,y_val,y_te]).mean():.1f}  '
             f'std={np.concatenate([y_tr,y_val,y_te]).std():.1f}  '
             f'min={np.concatenate([y_tr,y_val,y_te]).min():.1f}  '
             f'max={np.concatenate([y_tr,y_val,y_te]).max():.1f}  '
             f'| val: Pearson={r_all:.3f}  R²={r2_all:.3f}',
             ha='center', fontsize=9, style='italic')

    out_path = os.path.join(out_dir, f'diagnosis_{task_name}.png')
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'[saved] {out_path}')
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out_dir', default='output_dir/regression_diagnosis')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'[diag] device={device}')

    # Build MedSAM backbone once
    backbone = build_medsam_extractor(device)

    for task_name in ['lvef', 'lvedv']:
        print(f'\n{"="*50}\n[diag] Task: {task_name}')
        task_col = TASK_META[task_name]['col']
        df = build_cmr_df(task_col)

        df_tr  = df[df['split'] == 'train'].reset_index(drop=True)
        df_val = df[df['split'] == 'val'].reset_index(drop=True)
        df_te  = df[df['split'] == 'test'].reset_index(drop=True)
        print(f'  train={len(df_tr)} val={len(df_val)} test={len(df_te)}')

        from torch.utils.data import DataLoader
        loader = lambda d: DataLoader(CMRLabelDataset(d), batch_size=16,
                                      shuffle=False, num_workers=4, pin_memory=True)

        print('  extracting features...')
        X_tr,  y_tr  = extract_cmr_features(backbone, loader(df_tr),  device)
        X_val, y_val = extract_cmr_features(backbone, loader(df_val), device)
        X_te,  y_te  = extract_cmr_features(backbone, loader(df_te),  device)

        print(f'  GT stats — mean={y_tr.mean():.2f} std={y_tr.std():.2f} '
              f'min={y_tr.min():.2f} max={y_tr.max():.2f}')

        plot_task(task_name, X_tr, y_tr, X_val, y_val, X_te, y_te, args.out_dir, device)


if __name__ == '__main__':
    main()
