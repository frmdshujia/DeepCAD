"""
加载 end2end 保存的 best.pth，在 test 集上计算扩展指标。

分类：ROC-AUC、PR-AUC（average precision）、ACC、Sensitivity、Specificity（阈值 0.5 on sigmoid）
回归：R²、Pearson r、Spearman ρ（均在原始量纲上）

用法：
  python contrastive_pretrain/eval_end2end_best_metrics.py \\
    --ckpt_glob 'output_dir/end2end_conservative_ckpt_part*/**/best.pth' \\
    --output_csv output_dir/end2end_best_metrics_test.csv
"""

from __future__ import annotations

import argparse
import glob
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
from sklearn.metrics import average_precision_score, confusion_matrix, r2_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from contrastive_pretrain.end2end_stage2_ceiling import (
    End2EndFundus,
    FundusSingleTargetDataset,
    build_encoder,
)
from contrastive_pretrain.linear_probe_stage2_sweep import load_merged_full

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


class Args:
    """与 build_encoder / DataLoader 兼容的轻量 namespace"""

    def __init__(self):
        self.proj_dim = 256
        self.drop_path = 0.1
        self.retfound_ckpt = "RETFound_cfp_weights.pth"
        self.contrastive_ckpt = ""
        self.batch_size = 8
        self.num_workers = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fundus_csv", type=str, default="contrastive_pretrain/preprocessed_data/fundus_table.csv")
    p.add_argument("--stage2_csv", type=str, default="contrastive_pretrain/preprocessed_data/stage2_cmr.csv")
    p.add_argument(
        "--ckpt_glob",
        type=str,
        default="output_dir/end2end_conservative_ckpt_part*/**/best.pth",
        help="匹配 best.pth 的 glob（相对项目根）；若使用 --ckpts 则忽略此项",
    )
    p.add_argument(
        "--ckpts",
        nargs="+",
        default=None,
        help="显式列出若干 best.pth 路径（相对项目根或绝对路径），优先于 --ckpt_glob",
    )
    p.add_argument("--contrastive_ckpt", type=str, default="output_dir/contrast_finetune_e2e_20260412_174935/checkpoint_best.pth")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--output_csv", type=str, default="output_dir/end2end_best_metrics_test.csv")
    return p.parse_args()


def _binary_metrics(y: np.ndarray, prob: np.ndarray, thr: float = 0.5):
    y = y.astype(np.float64)
    pred = (prob >= thr).astype(np.int64)
    acc = float(np.mean(pred == y))
    if len(np.unique(y)) < 2:
        roc = float("nan")
        pr = float("nan")
    else:
        roc = float(roc_auc_score(y, prob))
        pr = float(average_precision_score(y, prob))
    try:
        tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    except ValueError:
        sen = spe = float("nan")
        return roc, pr, acc, sen, spe
    sen = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
    spe = float(tn / (tn + fp)) if (tn + fp) > 0 else float("nan")
    return roc, pr, acc, sen, spe


def eval_one_ckpt(path: str, merged: pd.DataFrame, device: torch.device, base_args: Args) -> dict:
    payload = torch.load(_abs(path), map_location="cpu")
    if "model" not in payload:
        raise ValueError(f"无 model 键: {path}")
    init = payload.get("init", "")
    target_col = payload.get("target_col", "")
    kind = payload.get("kind", "")
    if not init or not target_col or not kind:
        raise ValueError(f"checkpoint meta 不完整: {path}")

    df_tr = merged[merged["split"] == "train"].dropna(subset=[target_col]).reset_index(drop=True)
    df_te = merged[merged["split"] == "test"].dropna(subset=[target_col]).reset_index(drop=True)
    y_tr = pd.to_numeric(df_tr[target_col], errors="coerce").values
    y_te = pd.to_numeric(df_te[target_col], errors="coerce").values

    base_args.contrastive_ckpt = base_args.contrastive_ckpt or ""
    enc = build_encoder(init, base_args, device)
    model = End2EndFundus(enc).to(device)
    model.load_state_dict(payload["model"], strict=True)
    model.eval()

    test_ds = FundusSingleTargetDataset(df_te["fundus_image_path"].tolist(), y_te, False)
    # 仅用 train 估计回归标准化（与训练一致）
    if kind == "reg":
        y_tr_f = torch.tensor(y_tr, dtype=torch.float32)
        y_mean = y_tr_f.mean()
        y_std = y_tr_f.std().clamp_min(1e-6)
    else:
        y_mean = y_std = None

    test_loader = DataLoader(
        test_ds, batch_size=base_args.batch_size, shuffle=False, num_workers=base_args.num_workers, pin_memory=True
    )
    preds = []
    ys = []
    with torch.no_grad():
        for images, yb in test_loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds.append(logits.cpu())
            ys.append(yb)
    pred = torch.cat(preds, dim=0).numpy().ravel()
    yt = torch.cat(ys, dim=0).numpy().ravel()

    row = {
        "ckpt_path": path,
        "init": init,
        "target_col": target_col,
        "kind": kind,
        "n_test": len(yt),
        "epoch_saved": payload.get("epoch", ""),
        "val_loss_at_save": payload.get("val_loss", ""),
    }

    if kind == "binary":
        prob = 1.0 / (1.0 + np.exp(-pred))
        roc, pr, acc, sen, spe = _binary_metrics(yt, prob)
        row["roc_auc"] = roc
        row["pr_auc"] = pr
        row["acc"] = acc
        row["sensitivity"] = sen
        row["specificity"] = spe
        row["r2"] = ""
        row["pearson_r"] = ""
        row["spearman_r"] = ""
    else:
        pred_orig = pred * float(y_std) + float(y_mean)
        r2 = float(r2_score(yt, pred_orig))
        pr, _ = pearsonr(yt, pred_orig)
        sp, _ = spearmanr(yt, pred_orig)
        row["roc_auc"] = ""
        row["pr_auc"] = ""
        row["acc"] = ""
        row["sensitivity"] = ""
        row["specificity"] = ""
        row["r2"] = r2
        row["pearson_r"] = float(pr)
        row["spearman_r"] = float(sp)

    return row


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    merged = load_merged_full(args.fundus_csv, args.stage2_csv)
    if args.ckpts:
        paths = [_abs(p) for p in args.ckpts]
    else:
        pattern = _abs(args.ckpt_glob)
        paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        print("未找到 checkpoint", file=sys.stderr)
        sys.exit(1)

    base = Args()
    base.contrastive_ckpt = args.contrastive_ckpt

    rows = []
    for p in paths:
        try:
            rows.append(eval_one_ckpt(p, merged, device, base))
            print(f"OK {p}")
        except Exception as e:
            print(f"FAIL {p}: {e}", file=sys.stderr)
            rows.append({"ckpt_path": p, "error": str(e)})

    out = _abs(args.output_csv)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print(f"Wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
