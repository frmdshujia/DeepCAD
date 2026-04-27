"""
end2end_stage2_ceiling.py
「天花板」实验：解冻 FundusContrastModel（ViT backbone + 投影头）+ 线性任务头，
对 stage2 中病史/CMR 衍生列端到端 fine-tune，在 test 上报告指标。

初始化：
  - retfound：RETFound 预训练 backbone + 随机 proj_head（与线性探针 A 一致）
  - contrastive：checkpoint_best 中的 fundus_model

与冻结探针对比，用于估计「当前监督信号下眼底表征还能抬多少」。

用法（单目标）：
  python contrastive_pretrain/end2end_stage2_ceiling.py \\
    --init retfound --target_col composite_ischemic_hd

批量（与 linear_probe_stage2_sweep 同一套 discover 规则）：
  python contrastive_pretrain/end2end_stage2_ceiling.py --run_all_inits --output_csv output_dir/end2end_ceiling_AB.csv
"""

from __future__ import annotations

import argparse
import csv
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

from contrastive_pretrain.linear_probe_stage2_sweep import discover_target_columns, load_merged_full
from contrastive_pretrain.models_contrast import FundusContrastModel

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fundus_csv", type=str, default="contrastive_pretrain/preprocessed_data/fundus_table.csv")
    p.add_argument("--stage2_csv", type=str, default="contrastive_pretrain/preprocessed_data/stage2_cmr.csv")
    p.add_argument("--retfound_ckpt", type=str, default="RETFound_cfp_weights.pth")
    p.add_argument("--contrastive_ckpt", type=str, default="")
    p.add_argument("--init", type=str, choices=["retfound", "contrastive"], help="单跑时使用")
    p.add_argument("--target_col", type=str, default="", help="单目标列名；与 --run_all_inits 互斥")
    p.add_argument("--run_all_inits", action="store_true", help="对 discover 的全部目标 × (retfound, contrastive) 批量运行")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="端到端训练显存占用高，ViT-L 建议 8–16（视 GPU）",
    )
    p.add_argument("--num_workers", type=int, default=0, help="DataLoader workers；0 最稳（避免多进程+I/O 卡住）")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr_backbone", type=float, default=2e-5, help="ViT backbone 学习率")
    p.add_argument("--lr_head_group", type=float, default=2e-4, help="proj_head + 任务头学习率")
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help="梯度累积步数；有效 batch = batch_size × accumulation_steps，显存按 batch_size 计",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--max_missing_frac", type=float, default=0.35)
    p.add_argument("--min_train", type=int, default=80)
    p.add_argument("--output_csv", type=str, default="output_dir/end2end_ceiling_AB.csv")
    p.add_argument("--output_json", type=str, default="", help="单跑时可选写 JSON")
    p.add_argument(
        "--target_list",
        type=str,
        default="",
        help="与 --run_all_inits 合用：逗号分隔列名，只跑这些目标（须在 merged 表中存在）",
    )
    p.add_argument(
        "--targets_file",
        type=str,
        default="",
        help="每行一个列名；# 行为注释。与 target_list 二选一，targets_file 优先",
    )
    p.add_argument(
        "--ckpt_dir",
        type=str,
        default="",
        help="若设置：每个 epoch 验证后打印指标；当 val 变好时保存 best 权重；该 run 结束后再写一份最终 best",
    )
    p.add_argument(
        "--save_epoch_checkpoints",
        action="store_true",
        help="与 --ckpt_dir 合用：每个 epoch 末额外保存一份整模型，磁盘占用大（ViT-L 可达数百 MB/文件）",
    )
    return p.parse_args()


def _infer_target_kind(merged: pd.DataFrame, col: str):
    """当列不在 discover 结果里时，推断 binary / regression。"""
    if col not in merged.columns:
        return None
    s = pd.to_numeric(merged[col], errors="coerce").dropna()
    if len(s) < 10:
        return None
    u = set(float(x) for x in np.unique(s.values))
    if len(u) <= 2 and u <= {0.0, 1.0}:
        return "binary"
    if len(u) < 5:
        return None
    return "reg"


def resolve_targets_subset(
    merged: pd.DataFrame, discovered: list, targets_file: str, target_list: str
) -> list:
    """返回与 discover 同结构的列表；顺序与输入文件/列表一致。"""
    wanted: list[str] = []
    if targets_file:
        path = _abs(targets_file)
        with open(path) as f:
            for ln in f:
                ln = ln.strip()
                if not ln or ln.startswith("#"):
                    continue
                wanted.append(ln)
    elif target_list.strip():
        wanted = [x.strip() for x in target_list.split(",") if x.strip()]
    else:
        return discovered

    dmap = {t["name"]: t for t in discovered}
    out = []
    for w in wanted:
        if w in dmap:
            out.append(dmap[w])
            continue
        k = _infer_target_kind(merged, w)
        if k:
            miss = float(pd.to_numeric(merged[w], errors="coerce").isna().mean())
            out.append({"name": w, "kind": k, "missing_frac": miss})
            print(f"[resolve_targets] {w} 不在 discover 集合内，已推断 kind={k}")
        else:
            print(f"[resolve_targets] 跳过未知/无效列: {w}")
    return out


class FundusSingleTargetDataset(Dataset):
    def __init__(self, paths: list, y: np.ndarray, train_aug: bool):
        self.paths = paths
        self.y = y.astype(np.float32)
        self.train_aug = train_aug
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        if train_aug:
            self.t = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3 / 4, 4 / 3)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            self.t = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.t(img), torch.tensor(self.y[i], dtype=torch.float32)


class End2EndFundus(nn.Module):
    """backbone → proj_head（无 L2）→ 线性任务头"""

    def __init__(self, encoder: FundusContrastModel):
        super().__init__()
        self.encoder = encoder
        pdim = encoder.proj_head.net[-1].out_features
        self.task_head = nn.Linear(pdim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder.backbone.forward_features(x)
        h = self.encoder.proj_head(feat)
        return self.task_head(h).squeeze(-1)


def build_encoder(init: str, args, device):
    if init == "retfound":
        m = FundusContrastModel(proj_dim=args.proj_dim, drop_path_rate=args.drop_path)
        m.load_pretrained(_abs(args.retfound_ckpt))
        m.to(device)
        return m
    if init == "contrastive":
        if not args.contrastive_ckpt:
            raise ValueError("init=contrastive 需要 --contrastive_ckpt")
        ckpt = torch.load(_abs(args.contrastive_ckpt), map_location="cpu")
        ca = ckpt.get("args") or {}
        pdim = int(ca.get("proj_dim", args.proj_dim))
        dp = float(ca.get("drop_path", args.drop_path))
        m = FundusContrastModel(proj_dim=pdim, drop_path_rate=dp)
        m.load_state_dict(ckpt["fundus_model"], strict=True)
        m.to(device)
        return m
    raise ValueError(init)


def _both_classes(y: torch.Tensor) -> bool:
    u = set(float(t.item()) for t in y)
    return u >= {0.0, 1.0}


def _safe_ckpt_tag(s: str) -> str:
    t = "".join(c if (c.isalnum() or c in "._-") else "_" for c in str(s))
    return t[:120] if len(t) > 120 else t


def _save_end2end_ckpt(path: str, model: nn.Module, meta: dict):
    os.makedirs(os.path.dirname(_abs(path)) or ".", exist_ok=True)
    payload = {"model": model.state_dict(), **meta}
    torch.save(payload, _abs(path))


def train_end2end_one(
    init: str,
    target_col: str,
    kind: str,
    merged: pd.DataFrame,
    args,
    device: torch.device,
) -> dict:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    df_tr = merged[merged["split"] == "train"].dropna(subset=[target_col]).reset_index(drop=True)
    df_va = merged[merged["split"] == "val"].dropna(subset=[target_col]).reset_index(drop=True)
    df_te = merged[merged["split"] == "test"].dropna(subset=[target_col]).reset_index(drop=True)

    if len(df_tr) < args.min_train or len(df_va) < 8 or len(df_te) < 8:
        return {"skipped": True, "reason": "too_few_after_dropna", "init": init, "target_col": target_col}

    y_tr = pd.to_numeric(df_tr[target_col], errors="coerce").values
    y_va = pd.to_numeric(df_va[target_col], errors="coerce").values
    y_te = pd.to_numeric(df_te[target_col], errors="coerce").values

    if kind == "binary":
        y_tr_t = torch.tensor(y_tr, dtype=torch.float32)
        y_va_t = torch.tensor(y_va, dtype=torch.float32)
        y_te_t = torch.tensor(y_te, dtype=torch.float32)
        if not _both_classes(y_tr_t) or not _both_classes(y_va_t) or not _both_classes(y_te_t):
            return {"skipped": True, "reason": "not_both_classes_in_split", "init": init, "target_col": target_col}
        pos = float(y_tr_t.sum().item())
        neg = float(len(y_tr_t) - pos)
        pos_weight = torch.tensor([neg / max(pos, 1.0)], device=device)
        crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        y_mean = y_std = None
    else:
        y_tr_f = torch.tensor(y_tr, dtype=torch.float32)
        y_va_f = torch.tensor(y_va, dtype=torch.float32)
        y_te_f = torch.tensor(y_te, dtype=torch.float32)
        y_mean = y_tr_f.mean()
        y_std = y_tr_f.std().clamp_min(1e-6)
        y_tr_n = (y_tr_f - y_mean) / y_std
        y_va_n = (y_va_f - y_mean) / y_std
        y_te_n = (y_te_f - y_mean) / y_std
        crit = nn.MSELoss()

    enc = build_encoder(init, args, device)
    model = End2EndFundus(enc).to(device)

    opt = torch.optim.AdamW(
        [
            {"params": model.encoder.backbone.parameters(), "lr": args.lr_backbone, "weight_decay": args.weight_decay},
            {
                "params": list(model.encoder.proj_head.parameters()) + list(model.task_head.parameters()),
                "lr": args.lr_head_group,
                "weight_decay": args.weight_decay,
            },
        ]
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs), eta_min=args.lr_backbone * 0.01)

    train_ds = FundusSingleTargetDataset(df_tr["fundus_image_path"].tolist(), y_tr, True)
    val_ds = FundusSingleTargetDataset(df_va["fundus_image_path"].tolist(), y_va, False)
    # drop_last：小 batch 时若最后一步只有 1 张图，proj_head 里 BatchNorm 在 train 模式会报错
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    best_val = float("inf")
    best_state = None
    bad = 0

    acc = max(1, int(args.accumulation_steps))
    for ep in range(args.epochs):
        model.train()
        loss_ep = 0.0
        n_seen = 0
        opt.zero_grad()
        for step, (images, yb) in enumerate(train_loader):
            images = images.to(device, non_blocking=True)
            yb = yb.to(device)
            logits = model(images)
            if kind == "binary":
                loss = crit(logits, yb)
            else:
                ybn = (yb - y_mean.to(device)) / y_std.to(device)
                loss = crit(logits, ybn)
            loss = loss / acc
            loss.backward()
            loss_ep += loss.item() * acc * images.size(0)
            n_seen += images.size(0)
            if (step + 1) % acc == 0 or (step + 1) == len(train_loader):
                if args.clip_grad > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
                opt.step()
                opt.zero_grad()
        sched.step()
        loss_ep /= max(n_seen, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for images, yb in val_loader:
                images = images.to(device, non_blocking=True)
                yb = yb.to(device)
                logits = model(images)
                if kind == "binary":
                    v = crit(logits, yb)
                else:
                    ybn = (yb - y_mean.to(device)) / y_std.to(device)
                    v = crit(logits, ybn)
                val_loss += v.item() * images.size(0)
                n_val += images.size(0)
        val_loss /= max(n_val, 1)

        improved = val_loss < best_val - 1e-7
        if improved:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1

        print(
            f"    [epoch {ep + 1}/{args.epochs}] train_loss={loss_ep:.6f}  val_loss={val_loss:.6f}  "
            f"best_val={best_val:.6f}  bad_epoch={bad}" + ("  *best*" if improved else ""),
            flush=True,
        )

        ckroot = (getattr(args, "ckpt_dir", None) or "").strip()
        if ckroot:
            tag = f"{init}__{_safe_ckpt_tag(target_col)}"
            base = os.path.join(_abs(ckroot), tag)
            os.makedirs(base, exist_ok=True)
            if getattr(args, "save_epoch_checkpoints", False):
                _save_end2end_ckpt(
                    os.path.join(base, f"epoch_{ep + 1:03d}.pth"),
                    model,
                    {
                        "init": init,
                        "target_col": target_col,
                        "kind": kind,
                        "epoch": ep + 1,
                        "train_loss": float(loss_ep),
                        "val_loss": float(val_loss),
                    },
                )
            if improved:
                _save_end2end_ckpt(
                    os.path.join(base, "best.pth"),
                    model,
                    {
                        "init": init,
                        "target_col": target_col,
                        "kind": kind,
                        "epoch": ep + 1,
                        "train_loss": float(loss_ep),
                        "val_loss": float(val_loss),
                    },
                )

        if not improved and bad >= args.patience:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(device)
    model.eval()

    test_ds = FundusSingleTargetDataset(df_te["fundus_image_path"].tolist(), y_te, False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

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

    out: dict = {
        "skipped": False,
        "init": init,
        "target_col": target_col,
        "kind": kind,
        "n_train": int(len(df_tr)),
        "n_val": int(len(df_va)),
        "n_test": int(len(df_te)),
        "best_val_loss": float(best_val),
        "epochs_ran": ep + 1,
    }

    if kind == "binary":
        prob = 1.0 / (1.0 + np.exp(-pred))
        if len(np.unique(yt)) < 2:
            out["test_auroc"] = float("nan")
            out["test_auprc"] = float("nan")
        else:
            out["test_auroc"] = float(roc_auc_score(yt, prob))
            out["test_auprc"] = float(average_precision_score(yt, prob))
        out["test_mae"] = ""
        out["test_pearson_r"] = ""
    else:
        pred_orig = pred * float(y_std) + float(y_mean)
        out["test_mae"] = float(np.mean(np.abs(pred_orig - yt)))
        out["test_pearson_r"] = float(pearsonr(yt, pred_orig)[0])
        out["test_auroc"] = ""
        out["test_auprc"] = ""

    return out


def append_csv_row(path: str, row: dict, fieldnames: list[str]):
    os.makedirs(os.path.dirname(_abs(path)) or ".", exist_ok=True)
    exists = os.path.isfile(_abs(path))
    with open(_abs(path), "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            w.writeheader()
        w.writerow(row)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    merged = load_merged_full(args.fundus_csv, args.stage2_csv)
    discovered = discover_target_columns(merged, args.max_missing_frac)
    targets = resolve_targets_subset(merged, discovered, args.targets_file, args.target_list)
    tmap = {t["name"]: t["kind"] for t in targets}
    if not targets:
        raise SystemExit("无可用目标列（检查 --target_list / --targets_file 或 discover 规则）")

    fieldnames = [
        "init",
        "target_col",
        "kind",
        "skipped",
        "skip_reason",
        "n_train",
        "n_val",
        "n_test",
        "epochs_ran",
        "best_val_loss",
        "test_mae",
        "test_pearson_r",
        "test_auroc",
        "test_auprc",
    ]

    if args.run_all_inits:
        if not args.contrastive_ckpt:
            args.contrastive_ckpt = "output_dir/contrast_finetune_e2e_20260412_174935/checkpoint_best.pth"
        out_csv = args.output_csv
        if os.path.isfile(_abs(out_csv)):
            os.remove(_abs(out_csv))
        for init in ("retfound", "contrastive"):
            for tinfo in targets:
                name = tinfo["name"]
                kind = tinfo["kind"]
                print(f"\n=== end2end init={init}  target={name} ({kind}) ===")
                r = train_end2end_one(init, name, kind, merged, args, device)
                row = {k: "" for k in fieldnames}
                row.update(
                    {
                        "init": init,
                        "target_col": name,
                        "kind": kind,
                        "skipped": r.get("skipped", False),
                        "skip_reason": r.get("reason", ""),
                        "n_train": r.get("n_train", ""),
                        "n_val": r.get("n_val", ""),
                        "n_test": r.get("n_test", ""),
                        "epochs_ran": r.get("epochs_ran", ""),
                        "best_val_loss": r.get("best_val_loss", ""),
                        "test_mae": r.get("test_mae", ""),
                        "test_pearson_r": r.get("test_pearson_r", ""),
                        "test_auroc": r.get("test_auroc", ""),
                        "test_auprc": r.get("test_auprc", ""),
                    }
                )
                append_csv_row(out_csv, row, fieldnames)
                if not r.get("skipped"):
                    if kind == "binary":
                        print(
                            f"  → test AUROC={r['test_auroc']:.4f}  AUPRC={r['test_auprc']:.4f}  "
                            f"epochs={r['epochs_ran']}  val_loss={r['best_val_loss']:.4f}"
                        )
                    else:
                        print(
                            f"  → test MAE={r['test_mae']:.4f}  r={r['test_pearson_r']:.4f}  "
                            f"epochs={r['epochs_ran']}  val_loss={r['best_val_loss']:.4f}"
                        )
                else:
                    print(f"  → skipped: {r.get('reason')}")
                torch.cuda.empty_cache()
        print(f"\nDone. Wrote {_abs(out_csv)}")
        return

    if not args.target_col or not args.init:
        raise SystemExit("请指定 --target_col 与 --init，或使用 --run_all_inits")
    if args.init == "contrastive" and not args.contrastive_ckpt:
        raise SystemExit("init=contrastive 需要 --contrastive_ckpt")
    kind = tmap.get(args.target_col)
    if kind is None:
        raise SystemExit(f"目标列不在 discover 列表或不符合条件: {args.target_col}")

    r = train_end2end_one(args.init, args.target_col, kind, merged, args, device)
    print(json.dumps(r, indent=2, default=str))
    if args.output_json:
        with open(_abs(args.output_json), "w") as f:
            json.dump(r, f, indent=2, default=str)
        print(f"Wrote {_abs(args.output_json)}")


if __name__ == "__main__":
    main()
