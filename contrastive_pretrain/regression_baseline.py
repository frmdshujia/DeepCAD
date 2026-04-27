"""
regression_baseline.py
用冻结 fundus features 直接回归 CMR PC scores（非线性 MLP）。
如果 val Pearson R 近似于 0 ⟹ frozen features 不含 CMR 表型信号。
如果 train R 高但 val R 低 ⟹ 过拟合。

用法：
  python contrastive_pretrain/regression_baseline.py
"""
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import pearsonr

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# ── Load cached features ──────────────────────────────────────────────
train_cache = torch.load('output_dir/feature_cache/train_feats_full.pt')
val_cache   = torch.load('output_dir/feature_cache/val_feats_full.pt')

X_tr = train_cache['feats'].float().to(DEVICE)
Y_tr = train_cache['pc'].float().to(DEVICE)   # (N_tr, 14)
X_val = val_cache['feats'].float().to(DEVICE)
Y_val = val_cache['pc'].float().to(DEVICE)    # (N_val, 14)

n_pc = Y_tr.shape[1]
print(f'Train: {X_tr.shape}, Val: {X_val.shape}, PC dim: {n_pc}')

# Normalize targets
Y_mean = Y_tr.mean(0, keepdim=True)
Y_std  = Y_tr.std(0, keepdim=True) + 1e-8
Y_tr_n  = (Y_tr  - Y_mean) / Y_std
Y_val_n = (Y_val - Y_mean) / Y_std


# ── MLP Regressor ────────────────────────────────────────────────────
class MLPReg(nn.Module):
    def __init__(self, in_dim=1024, hidden=256, out_dim=14, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)


for hidden, wd in [(256, 0.1), (64, 0.5)]:
    model = MLPReg(hidden=hidden).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=200, eta_min=1e-5)

    batch_size = 128
    n_tr = len(X_tr)
    best_val_r = -1.0

    print(f'\n--- hidden={hidden}, wd={wd} ---')
    for epoch in range(200):
        model.train()
        perm = torch.randperm(n_tr)
        total_loss = 0.0
        for s in range(0, n_tr - batch_size + 1, batch_size):
            idx = perm[s:s+batch_size]
            pred = model(X_tr[idx])
            loss = F.mse_loss(pred, Y_tr_n[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step()

        if (epoch + 1) % 20 == 0 or epoch == 199:
            model.eval()
            with torch.no_grad():
                pred_tr  = model(X_tr).cpu().numpy()
                pred_val = model(X_val).cpu().numpy()
                y_tr_np  = Y_tr_n.cpu().numpy()
                y_val_np = Y_val_n.cpu().numpy()

            # Per-PC Pearson R
            tr_rs, val_rs = [], []
            for i in range(n_pc):
                r_tr,  _ = pearsonr(pred_tr[:, i],  y_tr_np[:, i])
                r_val, _ = pearsonr(pred_val[:, i], y_val_np[:, i])
                tr_rs.append(r_tr); val_rs.append(r_val)

            avg_tr_r  = float(np.mean(tr_rs))
            avg_val_r = float(np.mean(val_rs))
            max_val_r = float(np.max(val_rs))
            print(f'  [ep {epoch:3d}] loss={total_loss/max(1,n_tr//batch_size):.4f}  '
                  f'train_R={avg_tr_r:.4f}  val_R={avg_val_r:.4f}  val_R_max={max_val_r:.4f}')

print('\nDone.')
