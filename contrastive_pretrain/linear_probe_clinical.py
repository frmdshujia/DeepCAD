"""
linear_probe_clinical.py
实验二：冻结 fundus encoder，仅训练线性头 —— LVEF 回归 / 缺血性心脏病二分类。

组 A：RETFound 初始化 backbone（+ 随机冻结 proj，用于 representation=proj）
组 B：对比学习 checkpoint 中的 fundus_model（backbone+proj 与训练一致）

表示：cls = ViT CLS 1024-d；proj = 投影头输出 256-d（L2 前）

划分：train 上拟合 probe；val 上早停选最优；**test 上报告最终指标**（与实验说明一致）。

Probe 结构（encoder 始终冻结）：
  - ``--mlp_hidden_layers 0``：单层 Linear（默认，即原 linear probe）
  - ``--mlp_hidden_layers 1``：Linear → ReLU → Linear（1 个隐层）
  - ``--mlp_hidden_layers 2``：Linear → ReLU → Linear → ReLU → Linear（2 个隐层）

用法：
  conda activate retfound
  python contrastive_pretrain/linear_probe_clinical.py \\
    --retfound_ckpt RETFound_cfp_weights.pth \\
    --contrastive_ckpt output_dir/contrast_finetune_e2e_20260412_174935/checkpoint_best.pth \\
    --stage2_csv contrastive_pretrain/preprocessed_data/stage2_cmr.csv \\
    --fundus_csv contrastive_pretrain/preprocessed_data/fundus_table.csv \\
    --mlp_hidden_layers 1 \\
    --output_json output_dir/linear_probe_clinical_mlp1h.json
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
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from contrastive_pretrain.models_contrast import FundusContrastModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

LVEF_COL = "LV ejection fraction"
CHD_COL = "composite_ischemic_hd"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fundus_csv", type=str, default="contrastive_pretrain/preprocessed_data/fundus_table.csv")
    p.add_argument("--stage2_csv", type=str, default="contrastive_pretrain/preprocessed_data/stage2_cmr.csv")
    p.add_argument("--retfound_ckpt", type=str, default="RETFound_cfp_weights.pth")
    p.add_argument("--contrastive_ckpt", type=str, required=True, help="checkpoint_best.pth（含 fundus_model）")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--epochs_lvef", type=int, default=120)
    p.add_argument("--epochs_chd", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--patience", type=int, default=25, help="val 无提升则早停（epoch）")
    p.add_argument(
        "--mlp_hidden_layers",
        type=int,
        default=0,
        choices=[0, 1, 2],
        help="probe 隐层数：0=纯线性；1=一层隐层+ReLU；2=两层隐层+ReLU",
    )
    p.add_argument(
        "--mlp_hidden_dim",
        type=int,
        default=512,
        help="第一隐层宽度；第二隐层为 max(64, mlp_hidden_dim//2)（仅 mlp_hidden_layers=2）",
    )
    p.add_argument("--output_json", type=str, default="")
    return p.parse_args()


def make_probe_head(d_in: int, num_hidden_layers: int, hidden_dim: int) -> nn.Module:
    """单输出头：回归或二分类 logits 共用结构。"""
    if num_hidden_layers == 0:
        return nn.Linear(d_in, 1)
    h1 = int(hidden_dim)
    if num_hidden_layers == 1:
        return nn.Sequential(
            nn.Linear(d_in, h1),
            nn.ReLU(inplace=True),
            nn.Linear(h1, 1),
        )
    h2 = max(64, h1 // 2)
    return nn.Sequential(
        nn.Linear(d_in, h1),
        nn.ReLU(inplace=True),
        nn.Linear(h1, h2),
        nn.ReLU(inplace=True),
        nn.Linear(h2, 1),
    )


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_merged_clinical(fundus_csv: str, stage2_csv: str) -> pd.DataFrame:
    fu = pd.read_csv(_abs(fundus_csv))
    st = pd.read_csv(_abs(stage2_csv), low_memory=False)
    need = {"eid", "instance", LVEF_COL, CHD_COL, "split"}
    miss = need - set(st.columns)
    if miss:
        raise ValueError(f"stage2_csv 缺少列: {miss}")
    sub = st[list(need)].drop_duplicates(subset=["eid", "instance"])
    m = fu.merge(sub, on=["eid", "instance"], how="inner", suffixes=("", "_st2"))
    if "split_st2" in m.columns:
        if (m["split"] != m["split_st2"]).any():
            n = int((m["split"] != m["split_st2"]).sum())
            raise ValueError(f"fundus 与 stage2 的 split 不一致: {n} 行")
        m = m.drop(columns=["split_st2"])
    exist_mask = m["fundus_image_path"].apply(os.path.exists)
    n_bad = int((~exist_mask).sum())
    if n_bad:
        m = m[exist_mask].reset_index(drop=True)
        print(f"[load] 跳过缺失图像: {n_bad}")
    m = m.dropna(subset=[LVEF_COL, CHD_COL]).reset_index(drop=True)
    m[CHD_COL] = m[CHD_COL].astype(np.float32)
    # 二分类标签应为 0/1
    assert set(np.unique(m[CHD_COL].values)) <= {0.0, 1.0}, "composite_ischemic_hd 应为 0/1"
    print(f"[load] 合并后 N={len(m)}, train/val/test={len(m[m.split=='train'])}/{len(m[m.split=='val'])}/{len(m[m.split=='test'])}")
    return m


class FundusClinicalDataset(Dataset):
    """同一张图同时带 LVEF 与 CHD 标签，避免重复前向。"""

    def __init__(self, df: pd.DataFrame, train_aug: bool):
        self.paths = df["fundus_image_path"].tolist()
        self.y_lvef = df[LVEF_COL].values.astype(np.float32)
        self.y_chd = df[CHD_COL].values.astype(np.float32)
        self.train_aug = train_aug
        self.t = self._build_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.t(img)
        return (
            img,
            torch.tensor(self.y_lvef[i], dtype=torch.float32),
            torch.tensor(self.y_chd[i], dtype=torch.float32),
        )

    def _build_transform(self):
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        if self.train_aug:
            return transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3 / 4, 4 / 3)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        return transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )


@torch.no_grad()
def extract_features_cls_dual(loader, backbone: nn.Module, device):
    backbone.eval()
    xs, yl, yc = [], [], []
    for images, y_lv, y_ch in loader:
        images = images.to(device, non_blocking=True)
        feat = backbone.forward_features(images)
        xs.append(feat.cpu())
        yl.append(y_lv)
        yc.append(y_ch)
    return torch.cat(xs, dim=0), torch.cat(yl, dim=0), torch.cat(yc, dim=0)


@torch.no_grad()
def extract_features_proj_dual(loader, model: FundusContrastModel, device):
    model.eval()
    xs, yl, yc = [], [], []
    for images, y_lv, y_ch in loader:
        images = images.to(device, non_blocking=True)
        feat = model.backbone.forward_features(images)
        h = model.proj_head(feat)
        xs.append(h.cpu())
        yl.append(y_lv)
        yc.append(y_ch)
    return torch.cat(xs, dim=0), torch.cat(yl, dim=0), torch.cat(yc, dim=0)


def extract_all_splits_both_tasks(df_tr, df_va, df_te, model, representation, device, batch_size, num_workers):
    def _loader(df, aug):
        ds = FundusClinicalDataset(df, train_aug=aug)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    lt, lv, lte = _loader(df_tr, True), _loader(df_va, False), _loader(df_te, False)
    if representation == "cls":
        Xt, ylt, yct = extract_features_cls_dual(lt, model.backbone, device)
        Xv, ylv, ycv = extract_features_cls_dual(lv, model.backbone, device)
        Xe, yle, yce = extract_features_cls_dual(lte, model.backbone, device)
    else:
        Xt, ylt, yct = extract_features_proj_dual(lt, model, device)
        Xv, ylv, ycv = extract_features_proj_dual(lv, model, device)
        Xe, yle, yce = extract_features_proj_dual(lte, model, device)
    return (Xt, ylt, yct, Xv, ylv, ycv, Xe, yle, yce)


def train_lvef_probe(
    X_tr, y_tr, X_val, y_val, X_te, y_te, device, epochs, lr, weight_decay, seed, patience,
    mlp_hidden_layers: int, mlp_hidden_dim: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = X_tr.shape[1]
    y_tr = y_tr.view(-1, 1).float()
    y_val = y_val.view(-1, 1).float()
    y_te = y_te.view(-1, 1).float()
    mean = y_tr.mean(dim=0, keepdim=True)
    std = y_tr.std(dim=0, keepdim=True).clamp_min(1e-6)
    y_tr_n = (y_tr - mean) / std
    y_val_n = (y_val - mean) / std

    head = make_probe_head(d_in, mlp_hidden_layers, mlp_hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 1e-3)

    X_tr_d = X_tr.to(device)
    X_val_d = X_val.to(device)
    X_te_d = X_te.to(device)
    y_tr_n_d = y_tr_n.to(device)
    y_val_n_d = y_val_n.to(device)

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(epochs):
        head.train()
        n = X_tr_d.shape[0]
        perm = torch.randperm(n, device=device)
        bs = min(512, n)
        loss_ep = 0.0
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            pred = head(X_tr_d[idx])
            loss = nn.functional.mse_loss(pred, y_tr_n_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_ep += loss.item() * idx.numel()
        loss_ep /= n
        sched.step()

        head.eval()
        with torch.no_grad():
            pv = head(X_val_d)
            val_mse = nn.functional.mse_loss(pv, y_val_n_d).item()
        if val_mse < best_val - 1e-7:
            best_val = val_mse
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred_te_n = head(X_te_d).cpu()
    pred_te = pred_te_n * std + mean
    mae = float((pred_te - y_te).abs().mean())
    r = float(pearsonr(y_te.numpy().ravel(), pred_te.numpy().ravel())[0])
    return {
        "test_mae": mae,
        "test_pearson_r": r,
        "best_val_mse_normalized": float(best_val),
        "epochs_ran": ep + 1,
        "mlp_hidden_layers": mlp_hidden_layers,
    }


def binary_acc_stratified(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict:
    """二分类在 0.5 阈值下的整体 ACC，以及在真实阳性 / 真实阴性子集上的 ACC（即敏感度 / 特异度）。"""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    prob = np.asarray(prob, dtype=np.float64).ravel()
    pred = (prob >= threshold).astype(np.int32)
    y = y_true.astype(np.int32)
    acc = float((pred == y).mean()) if len(y) else float("nan")
    m_pos = y == 1
    m_neg = y == 0
    acc_pos = float((pred[m_pos] == y[m_pos]).mean()) if m_pos.any() else float("nan")
    acc_neg = float((pred[m_neg] == y[m_neg]).mean()) if m_neg.any() else float("nan")
    bal = (
        float((acc_pos + acc_neg) / 2.0)
        if (m_pos.any() and m_neg.any() and np.isfinite(acc_pos) and np.isfinite(acc_neg))
        else float("nan")
    )
    return {
        "accuracy": acc,
        "acc_on_true_positive": acc_pos,
        "acc_on_true_negative": acc_neg,
        "balanced_accuracy": bal,
        "n_pos": int(m_pos.sum()),
        "n_neg": int(m_neg.sum()),
    }


def train_chd_probe(
    X_tr, y_tr, X_val, y_val, X_te, y_te, device, epochs, lr, weight_decay, seed, patience,
    mlp_hidden_layers: int, mlp_hidden_dim: int,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d_in = X_tr.shape[1]
    y_tr = y_tr.view(-1)
    y_val = y_val.view(-1)
    y_te = y_te.view(-1)
    pos = float(y_tr.sum().item())
    neg = float(len(y_tr) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)

    head = make_probe_head(d_in, mlp_hidden_layers, mlp_hidden_dim).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs), eta_min=lr * 1e-3)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    X_tr_d = X_tr.to(device)
    X_val_d = X_val.to(device)
    X_te_d = X_te.to(device)
    y_tr_d = y_tr.to(device)
    y_val_d = y_val.to(device)
    y_te_d = y_te.to(device)

    def _auroc(y_true, logits):
        y_true = y_true.detach().cpu().numpy().ravel()
        prob = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        if len(np.unique(y_true)) < 2:
            return float("nan")
        return float(roc_auc_score(y_true, prob))

    best_val_auroc = -1.0
    best_state = None
    bad = 0

    for ep in range(epochs):
        head.train()
        n = X_tr_d.shape[0]
        perm = torch.randperm(n, device=device)
        bs = min(512, n)
        for s in range(0, n, bs):
            idx = perm[s : s + bs]
            logit = head(X_tr_d[idx]).squeeze(-1)
            loss = crit(logit, y_tr_d[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        head.eval()
        with torch.no_grad():
            lv = head(X_val_d).squeeze(-1)
            va = _auroc(y_val_d, lv)
        if not np.isfinite(va):
            va = -1.0
        if va > best_val_auroc + 1e-6:
            best_val_auroc = va
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        lt = head(X_te_d).squeeze(-1)
        prob = torch.sigmoid(lt).cpu().numpy().ravel()
    yt = y_te.numpy().ravel()
    if len(np.unique(yt)) < 2:
        auroc = float("nan")
        auprc = float("nan")
    else:
        auroc = float(roc_auc_score(yt, prob))
        auprc = float(average_precision_score(yt, prob))
    test_bin = binary_acc_stratified(yt, prob)

    with torch.no_grad():
        X_all = torch.cat([X_tr_d, X_val_d, X_te_d], dim=0)
        y_all = torch.cat([y_tr_d, y_val_d, y_te_d], dim=0)
        lt_all = head(X_all).squeeze(-1)
        prob_all = torch.sigmoid(lt_all).cpu().numpy().ravel()
    yt_all = y_all.cpu().numpy().ravel()
    cohort_bin = binary_acc_stratified(yt_all, prob_all)

    return {
        "test_auroc": auroc,
        "test_auprc": auprc,
        "test_accuracy": test_bin["accuracy"],
        "test_acc_on_true_positive": test_bin["acc_on_true_positive"],
        "test_acc_on_true_negative": test_bin["acc_on_true_negative"],
        "test_balanced_accuracy": test_bin["balanced_accuracy"],
        "test_n_pos": test_bin["n_pos"],
        "test_n_neg": test_bin["n_neg"],
        "cohort_accuracy": cohort_bin["accuracy"],
        "cohort_acc_on_true_positive": cohort_bin["acc_on_true_positive"],
        "cohort_acc_on_true_negative": cohort_bin["acc_on_true_negative"],
        "cohort_balanced_accuracy": cohort_bin["balanced_accuracy"],
        "cohort_n_pos": cohort_bin["n_pos"],
        "cohort_n_neg": cohort_bin["n_neg"],
        "best_val_auroc": float(best_val_auroc) if np.isfinite(best_val_auroc) else None,
        "epochs_ran": ep + 1,
        "train_pos_rate": float(pos / max(pos + neg, 1.0)),
        "mlp_hidden_layers": mlp_hidden_layers,
    }


def load_group_a(proj_dim, drop_path, retfound_path, device):
    m = FundusContrastModel(proj_dim=proj_dim, drop_path_rate=drop_path)
    m.load_pretrained(_abs(retfound_path))
    m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_group_b(proj_dim, drop_path, retfound_path, contrast_ckpt, device):
    ckpt = torch.load(_abs(contrast_ckpt), map_location="cpu")
    ca = ckpt.get("args") or {}
    pdim = int(ca.get("proj_dim", proj_dim))
    dp = float(ca.get("drop_path", drop_path))
    m = FundusContrastModel(proj_dim=pdim, drop_path_rate=dp)
    m.load_state_dict(ckpt["fundus_model"], strict=True)
    m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    merged = load_merged_clinical(args.fundus_csv, args.stage2_csv)
    df_tr = merged[merged["split"] == "train"].reset_index(drop=True)
    df_va = merged[merged["split"] == "val"].reset_index(drop=True)
    df_te = merged[merged["split"] == "test"].reset_index(drop=True)
    if len(df_tr) < 10 or len(df_va) < 2 or len(df_te) < 2:
        raise ValueError("train/val/test 样本过少")

    probe_desc = (
        "linear" if args.mlp_hidden_layers == 0 else f"mlp_{args.mlp_hidden_layers}hidden_relu"
    )
    report = {
        "fundus_csv": args.fundus_csv,
        "stage2_csv": args.stage2_csv,
        "contrastive_ckpt": args.contrastive_ckpt,
        "lvef_col": LVEF_COL,
        "chd_col": CHD_COL,
        "n_train": len(df_tr),
        "n_val": len(df_va),
        "n_test": len(df_te),
        "mlp_hidden_layers": args.mlp_hidden_layers,
        "mlp_hidden_dim": args.mlp_hidden_dim,
        "probe_type": probe_desc,
        "note": "probe 在 train 上训练，val 早停，指标在 test 上报告；encoder 冻结。",
        "runs": [],
    }

    groups = [
        ("A_retfound", lambda: load_group_a(args.proj_dim, args.drop_path, args.retfound_ckpt, device)),
        ("B_contrastive", lambda: load_group_b(args.proj_dim, args.drop_path, args.retfound_ckpt, args.contrastive_ckpt, device)),
    ]

    for gname, loader_fn in groups:
        model = loader_fn()
        for rep in ("cls", "proj"):
            print(f"\n=== 特征提取 {gname}  {rep} ===")
            Xt, ylt, yct, Xv, ylv, ycv, Xe, yle, yce = extract_all_splits_both_tasks(
                df_tr, df_va, df_te, model, rep, device, args.batch_size, args.num_workers,
            )
            torch.cuda.empty_cache()

            print(f"  LVEF 训练 probe（{probe_desc}）…")
            lvef_res = train_lvef_probe(
                Xt, ylt, Xv, ylv, Xe, yle, device,
                args.epochs_lvef, args.lr, args.weight_decay, args.seed, args.patience,
                args.mlp_hidden_layers, args.mlp_hidden_dim,
            )
            print(f"  CHD 训练 probe（{probe_desc}）…")
            chd_res = train_chd_probe(
                Xt, yct, Xv, ycv, Xe, yce, device,
                args.epochs_chd, args.lr, args.weight_decay, args.seed, args.patience,
                args.mlp_hidden_layers, args.mlp_hidden_dim,
            )

            report["runs"].append(
                {
                    "group": gname,
                    "representation": rep,
                    "lvef": lvef_res,
                    "chd": chd_res,
                }
            )
            print(
                f"  [{gname} {rep}] LVEF test MAE={lvef_res['test_mae']:.4f}  Pearson={lvef_res['test_pearson_r']:.4f}  |  "
                f"CHD test AUROC={chd_res['test_auroc']:.4f}  AUPRC={chd_res['test_auprc']:.4f}"
            )
        del model
        torch.cuda.empty_cache()

    out = args.output_json or os.path.join(ROOT, "output_dir", "linear_probe_clinical.json")
    os.makedirs(os.path.dirname(_abs(out)) or ".", exist_ok=True)
    with open(_abs(out), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {_abs(out)}")


if __name__ == "__main__":
    main()
