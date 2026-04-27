#!/usr/bin/env python3
"""
在预计算的 CLS embedding 上只训练线性分类头（Mode1），无 backbone 前向，极快。

依赖 extract_stage1_embeddings.py 生成的 .pt（含 label__{col} 与 split）。
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score
from timm.models.layers import trunc_normal_

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _append_csv(path: str, row: dict, fields: list):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    ex = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(f, fieldnames=fields)
            if not ex:
                w.writeheader()
            w.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def auc_safe(y, p):
    y = np.asarray(y).astype(np.int64)
    p = np.asarray(p).astype(np.float64)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def auprc_safe(y, p):
    if len(y) == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--embedding_pt", type=str, required=True)
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=16384)
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument(
        "--loss",
        type=str,
        default="bce_smooth",
        choices=["bce_smooth", "bce_hard", "bce_pos_weight"],
        help="bce_smooth: BCE + 标签平滑; bce_hard: 硬 0/1; bce_pos_weight: 类别不平衡加权（训练集统计）",
    )
    p.add_argument(
        "--pos_weight",
        type=str,
        default="auto",
        help='仅 loss=bce_pos_weight：正类权重，"auto"=N_neg/N_pos（全训练集）或正数如 2.5',
    )
    p.add_argument(
        "--hparam_tag",
        type=str,
        default="",
        help="可选：写入 metrics 与打印，便于 sweep 区分套餐名",
    )
    p.add_argument(
        "--delivery_final_csv",
        type=str,
        default="",
        help="非空则追加一行汇总；sweep 时建议每套餐单独 csv",
    )
    p.add_argument(
        "--skip_delivery",
        action="store_true",
        help="不写 delivery_final_csv",
    )
    return p.parse_args()


def binary_soft(y: torch.Tensor, eps: float) -> torch.Tensor:
    yf = y.float()
    return yf * (1.0 - eps) + (1.0 - yf) * eps


def _print_hparams(args: argparse.Namespace, extra: dict[str, Any]) -> None:
    d = {
        "hparam_tag": args.hparam_tag or "(default)",
        "embedding_pt": args.embedding_pt,
        "target_col": args.target_col,
        "output_dir": args.output_dir,
        "gpu": args.gpu,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "smoothing": args.smoothing,
        "loss": args.loss,
        "pos_weight": args.pos_weight,
        **extra,
    }
    print("=" * 72)
    print("[mode1-emb] CURRENT HYPERPARAMETERS (linear probe)")
    print(json.dumps(d, indent=2, ensure_ascii=False))
    print("=" * 72)


def main():
    args = parse_args()
    # 父进程（如 run_mode1_emb_train_parallel）已设 CUDA_VISIBLE_DEVICES 时勿覆盖，
    # 否则会把多卡并行误绑回物理 GPU0。
    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pack = torch.load(args.embedding_pt, map_location="cpu")
    emb = pack["embeddings"].float()
    if emb.dtype == torch.float16:
        emb = emb.float()
    key = f"label__{args.target_col}"
    if key not in pack:
        raise KeyError(f"embedding 包中无 {key}，请用新版 extract_stage1_embeddings.py 重新导出")
    ya = np.asarray(pack[key], dtype=np.float64)
    splits = np.array(pack["split"])
    valid = np.isfinite(ya) & np.isin(ya, [0.0, 1.0])
    emb = emb[torch.tensor(valid)]
    y_all = torch.tensor(ya[valid], dtype=torch.long)
    splits = splits[valid]
    tr = splits == "train"
    va = splits == "val"
    te = splits == "test"

    X_tr, y_tr = emb[tr], y_all[tr]
    X_va, y_va = emb[va], y_all[va]
    X_te, y_te = emb[te], y_all[te]
    n_pos_tr = int((y_tr == 1).sum().item())
    n_neg_tr = int((y_tr == 0).sum().item())
    if args.loss == "bce_pos_weight":
        if str(args.pos_weight).lower() == "auto":
            pw = float(n_neg_tr) / float(max(n_pos_tr, 1))
        else:
            pw = float(args.pos_weight)
        pos_w_t = torch.tensor(pw, device=device, dtype=torch.float32)
    else:
        pw = None
        pos_w_t = None

    _print_hparams(
        args,
        {
            "init_source": pack.get("init_source", ""),
            "n_train_pos": n_pos_tr,
            "n_train_neg": n_neg_tr,
            "pos_weight_resolved": pw,
            "device": str(device),
        },
    )
    print(
        f"[mode1-emb] {args.target_col} train/val/test={len(y_tr)}/{len(y_va)}/{len(y_te)} D={X_tr.shape[1]}"
    )

    head = nn.Linear(X_tr.shape[1], 1, bias=True).to(device)
    trunc_normal_(head.weight, std=2e-5)
    nn.init.constant_(head.bias, 0)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def run_epoch(X, y, train: bool):
        head.train(train)
        total_loss = 0.0
        n = len(X)
        if n == 0:
            return 0.0
        idx = torch.randperm(n) if train else torch.arange(n)
        bs = min(args.batch_size, n)
        for s in range(0, n, bs):
            sel = idx[s : s + bs]
            xb = X[sel].to(device)
            yb = y[sel].to(device)
            logit = head(xb).squeeze(-1)
            if args.loss == "bce_smooth":
                y_smooth = binary_soft(yb, args.smoothing)
                loss = F.binary_cross_entropy_with_logits(logit, y_smooth)
            elif args.loss == "bce_hard":
                loss = F.binary_cross_entropy_with_logits(logit, yb.float())
            else:
                assert pos_w_t is not None
                loss = F.binary_cross_entropy_with_logits(
                    logit, yb.float(), pos_weight=pos_w_t
                )
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            total_loss += loss.item() * len(sel)
        return total_loss / n

    best_auc = -1.0
    best_state = None
    for ep in range(args.epochs):
        run_epoch(X_tr, y_tr, True)
        head.eval()
        with torch.no_grad():
            pv = []
            nv = len(X_va)
            for s in range(0, nv, args.batch_size):
                xb = X_va[s : s + args.batch_size].to(device)
                pv.append(torch.sigmoid(head(xb).squeeze(-1)).cpu().numpy())
            pv = np.concatenate(pv) if nv else np.array([])
        vauc = auc_safe(y_va.numpy(), pv) if nv else float("nan")
        if not np.isnan(vauc) and vauc > best_auc:
            best_auc = vauc
            best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}
        if ep % 10 == 0 or ep == args.epochs - 1:
            print(f"  epoch {ep} val_auroc={vauc:.4f}")

    if best_state is not None:
        head.load_state_dict(best_state)

    head.eval()
    with torch.no_grad():
        def probs(X):
            out = []
            n = len(X)
            for s in range(0, n, args.batch_size):
                xb = X[s : s + args.batch_size].to(device)
                out.append(torch.sigmoid(head(xb).squeeze(-1)).cpu().numpy())
            return np.concatenate(out) if n else np.array([])

        p_va = probs(X_va)
        p_te = probs(X_te)

    t_auc = auc_safe(y_te.numpy(), p_te)
    t_pr = auprc_safe(y_te.numpy(), p_te)
    v_auc = auc_safe(y_va.numpy(), p_va)
    v_pr = auprc_safe(y_va.numpy(), p_va)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "best_val_auroc": best_auc}, out_dir / "head_only.pt")
    hparams = {
        "hparam_tag": args.hparam_tag,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "smoothing": args.smoothing,
        "loss": args.loss,
        "pos_weight": args.pos_weight,
        "pos_weight_resolved": pw,
    }
    with open(out_dir / "metrics_mode1_emb.json", "w") as f:
        json.dump(
            {
                "target_col": args.target_col,
                "val_auroc": v_auc,
                "val_auprc": v_pr,
                "test_auroc": t_auc,
                "test_auprc": t_pr,
                "n_train": int(len(y_tr)),
                "n_val": int(len(y_va)),
                "n_test": int(len(y_te)),
                "hparams": hparams,
            },
            f,
            indent=2,
        )
    (out_dir / "DONE").write_text("mode1_emb\n", encoding="utf-8")
    print(
        f"[mode1-emb] done test_auroc={t_auc:.4f} test_auprc={t_pr:.4f} -> {out_dir}"
    )

    if args.delivery_final_csv and not args.skip_delivery:
        dpath = (
            args.delivery_final_csv
            if os.path.isabs(args.delivery_final_csv)
            else str(REPO_ROOT / args.delivery_final_csv)
        )
        row = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "target_col": args.target_col,
            "mode": "1_emb",
            "embedding_init": pack.get("init_source", ""),
            "hparam_tag": args.hparam_tag,
            "val_auroc": v_auc,
            "val_auprc": v_pr,
            "test_auroc": t_auc,
            "test_auprc": t_pr,
            "output_dir": str(out_dir),
        }
        fields = [
            "timestamp",
            "target_col",
            "mode",
            "embedding_init",
            "hparam_tag",
            "val_auroc",
            "val_auprc",
            "test_auroc",
            "test_auprc",
            "output_dir",
        ]
        _append_csv(dpath, row, fields)


if __name__ == "__main__":
    main()
