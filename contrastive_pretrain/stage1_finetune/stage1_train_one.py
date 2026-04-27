#!/usr/bin/env python3
"""
单任务、单卡：stage1 表二分类微调（独立脚本，不影响 main_finetune / 对比学习里的损失）。

--loss_type paper_bce（默认）：
  与 RETFound Methods 描述一致：BCEWithLogits + 标签平滑软标签，无 pos_weight。
--loss_type bce_pos_weight：
  保留与 linear_probe 等一致的「正类加权」：nn.BCEWithLogitsLoss(pos_weight=…)，
  硬标签 0/1；pos_weight 可由训练集 neg/pos 自动估计（--pos_weight 0）或手动指定。

说明：仓库内 FocalLoss、main_finetune、contrastive_pretrain 等未因本脚本改动而删除。
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
from timm.models.layers import trunc_normal_

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models_vit  # noqa: E402
import util.lr_decay as lrd  # noqa: E402
import util.lr_sched as lr_sched  # noqa: E402
import util.misc as misc  # noqa: E402
from util.misc import NativeScalerWithGradNormCount as NativeScaler  # noqa: E402
from util.pos_embed import interpolate_pos_embed  # noqa: E402

from contrastive_pretrain.stage1_finetune.stage1_dataset import Stage1BinaryDataset  # noqa: E402
from contrastive_pretrain.stage1_finetune.stage1_paths import (  # noqa: E402
    COL_LEFT,
    COL_RIGHT,
    filter_target_valid,
    prepare_image_frame,
    split_subset,
)


def _append_csv_locked(path: str, row: dict, fieldnames: list):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                w.writeheader()
            w.writerow(row)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def load_pretrained_vit(model: nn.Module, ckpt_path: str, source: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if source == "retfound":
        checkpoint_model = ckpt["model"]
    else:
        raw = ckpt.get("fundus_model")
        if raw is None:
            raise KeyError(f"{ckpt_path} 缺少 fundus_model")
        checkpoint_model = {}
        for k, v in raw.items():
            if k.startswith("backbone."):
                checkpoint_model[k.replace("backbone.", "")] = v
    state_dict = model.state_dict()
    for k in ["head.weight", "head.bias"]:
        if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
            del checkpoint_model[k]
    interpolate_pos_embed(model, checkpoint_model)
    msg = model.load_state_dict(checkpoint_model, strict=False)
    print(msg)
    trunc_normal_(model.head.weight, std=2e-5)
    if model.head.bias is not None:
        nn.init.constant_(model.head.bias, 0)


def apply_finetune_mode(model: nn.Module, mode: int) -> None:
    if mode == 3:
        for p in model.parameters():
            p.requires_grad = True
        return
    for p in model.parameters():
        p.requires_grad = False
    for p in model.head.parameters():
        p.requires_grad = True
    if mode == 2:
        n_blocks = len(model.blocks)
        k = max(1, n_blocks // 3)
        for blk in model.blocks[n_blocks - k :]:
            for p in blk.parameters():
                p.requires_grad = True


def build_optimizer(model: nn.Module, mode: int, args: argparse.Namespace):
    wd = args.weight_decay
    if mode == 1:
        params = list(model.head.parameters())
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=wd)
    param_groups = lrd.param_groups_lrd(
        model,
        wd,
        no_weight_decay_list=model.no_weight_decay(),
        layer_decay=args.layer_decay,
    )
    return torch.optim.AdamW(param_groups, lr=args.lr)


def binary_targets_smoothed(y_long: torch.Tensor, smoothing: float) -> torch.Tensor:
    """二分类标签平滑：y=1→1-ε，y=0→ε（与 CE 版平滑一致的两类特例）。"""
    y = y_long.float()
    return y * (1.0 - smoothing) + (1.0 - y) * smoothing


def train_one_epoch_bce(
    model: nn.Module,
    data_loader,
    optimizer,
    device: torch.device,
    epoch: int,
    loss_scaler,
    args: SimpleNamespace,
):
    """paper_bce：BCE + 标签平滑；bce_pos_weight：BCEWithLogitsLoss(pos_weight)，硬标签。"""
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter
    optimizer.zero_grad()
    smoothing = float(args.smoothing)
    loss_type = getattr(args, "loss_type", "paper_bce")
    bce_crit = getattr(args, "bce_criterion", None)

    for data_iter_step, (*_, samples, targets) in enumerate(
        metric_logger.log_every(data_loader, print_freq, header)
    ):
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(
                optimizer, data_iter_step / len(data_loader) + epoch, args
            )

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).long()

        with torch.cuda.amp.autocast():
            logits = model(samples)
            logit = logits.squeeze(-1)
            if loss_type == "bce_pos_weight":
                assert bce_crit is not None
                y_hard = targets.float()
                loss = bce_crit(logit, y_hard)
            else:
                y_smooth = binary_targets_smoothed(targets, smoothing)
                loss = F.binary_cross_entropy_with_logits(logit, y_smooth)

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss = loss / accum_iter
        loss_scaler(
            loss,
            optimizer,
            clip_grad=args.clip_grad,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=(data_iter_step + 1) % accum_iter == 0,
        )
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        max_lr = max(g["lr"] for g in optimizer.param_groups)
        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)

    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def forward_probs(model, loader, device):
    model.eval()
    ys, ps = [], []
    for batch in loader:
        *_, images, targets = batch
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            logits = model(images)
            logit = logits.float().squeeze(-1)
            prob = torch.sigmoid(logit).cpu().numpy()
        ys.append(targets.numpy())
        ps.append(prob)
    if not ys:
        return np.array([]), np.array([])
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def binary_metrics(y_true: np.ndarray, p: np.ndarray, thr: float):
    y_pred = (p >= thr).astype(np.int64)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    sens = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    spec = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    npv = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
    acc = accuracy_score(y_true, y_pred) if len(y_true) else float("nan")
    return {
        "acc": acc,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "thr": thr,
    }


def youden_threshold(y_true: np.ndarray, p: np.ndarray, n_grid: int = 101):
    if len(np.unique(y_true)) < 2:
        return 0.5
    qs = np.linspace(0.0, 1.0, n_grid)
    best_t, best_j = 0.5, -1.0
    for t in qs:
        m = binary_metrics(y_true, p, t)
        j = m["sensitivity"] + m["specificity"] - 1.0
        if not math.isnan(j) and j > best_j:
            best_j, best_t = j, t
    return float(best_t)


def auc_safe(y, p):
    if len(y) == 0 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def auprc_safe(y, p):
    if len(y) == 0:
        return float("nan")
    return float(average_precision_score(y, p))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--stage1_csv",
        type=str,
        default="contrastive_pretrain/preprocessed_data/stage1_fundus_downstream_finetuning_with_image_paths.csv",
    )
    p.add_argument(
        "--fundus_root",
        type=str,
        default="/data/home/home6/fundus_data/UKB/fundus_images",
        help="UKB 眼底 PNG 根目录（含 eid_21015_* / eid_21016_*）",
    )
    p.add_argument("--target_col", type=str, required=True)
    p.add_argument(
        "--init_source",
        type=str,
        choices=["retfound", "controlled", "no_residual"],
        required=True,
    )
    p.add_argument(
        "--init_ckpt",
        type=str,
        default="",
        help="覆盖默认 init 权重路径；空则使用 init_source 预设",
    )
    p.add_argument("--mode", type=int, choices=[1, 2, 3], required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--gpu", type=int, default=0)

    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--warmup_epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--lr", type=float, default=None, help="默认按 mode 自动设定")
    p.add_argument("--layer_decay", type=float, default=0.75)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument(
        "--loss_type",
        type=str,
        choices=["paper_bce", "bce_pos_weight"],
        default="paper_bce",
        help="paper_bce=原文式 BCE+标签平滑(无pos_weight)；bce_pos_weight=正类加权 BCE（硬标签，与 linear_probe 一致）",
    )
    p.add_argument(
        "--pos_weight",
        type=float,
        default=0.0,
        help="仅 loss_type=bce_pos_weight：正类权重；0 表示用训练集 n_neg/n_pos 自动估计",
    )
    p.add_argument("--smoothing", type=float, default=0.1)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--balance_train", action="store_true")
    p.add_argument("--resume", action="store_true", help="从 output_dir/checkpoint_last.pth 续训")
    p.add_argument(
        "--delivery_csv",
        type=str,
        default="",
        help="每 epoch 追加一行 val 指标（实时交付）",
    )
    p.add_argument(
        "--delivery_final_csv",
        type=str,
        default="",
        help="任务完成后追加一行最终 test 指标",
    )
    p.add_argument(
        "--narrow_task_csv",
        type=str,
        default="",
        help="每任务单表（负采样等）：需列 fundus_path, label, split；标签列会映射为 --target_col",
    )
    return p.parse_args()


def default_ckpt(init_source: str) -> str:
    root = REPO_ROOT
    if init_source == "retfound":
        return str(root / "RETFound_cfp_weights.pth")
    if init_source == "controlled":
        return str(
            root
            / "output_dir/exp_stratified_e2e_controlled_20260418_115523/checkpoint_best.pth"
        )
    if init_source == "no_residual":
        return str(
            root
            / "output_dir/exp_stratified_e2e_no_covariate_residual_20260418_173147/checkpoint_best.pth"
        )
    raise ValueError(init_source)


def default_lr(mode: int) -> float:
    if mode == 1:
        return 1e-4
    if mode == 2:
        return 5e-5
    return 5e-4


def print_default_ckpt_report() -> None:
    print("[default_ckpt] 预置 init 权重路径（default_ckpt 解析结果）")
    for src in ["retfound", "controlled", "no_residual"]:
        p = default_ckpt(src)
        ok = os.path.isfile(p)
        print(f"  init_source={src!r}")
        print(f"    path   = {p}")
        print(f"    exists = {ok}")


def main():
    args = parse_args()
    lr = args.lr if args.lr is not None else default_lr(args.mode)
    args.lr = lr

    if not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    misc.init_distributed_mode(
        SimpleNamespace(
            dist_on_itp=False,
            distributed=False,
            rank=0,
            world_size=1,
            gpu=0,
            dist_url="env://",
        )
    )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    fundus_root = args.fundus_root
    if args.narrow_task_csv.strip():
        narrow_p = Path(args.narrow_task_csv.strip())
        if not narrow_p.is_absolute():
            narrow_p = REPO_ROOT / narrow_p
        if not narrow_p.is_file():
            raise FileNotFoundError(f"narrow_task_csv 不存在: {narrow_p}")
        df = pd.read_csv(narrow_p, low_memory=False)
        for req in ("fundus_path", "label", "split"):
            if req not in df.columns:
                raise ValueError(f"narrow_task_csv 需含列 fundus_path, label, split；缺 {req}")
        df = df.copy()
        df["fundus_image_path"] = df["fundus_path"].astype(str)
        df[args.target_col] = pd.to_numeric(df["label"], errors="coerce")
        df = filter_target_valid(df, args.target_col)
    else:
        csv_path = REPO_ROOT / args.stage1_csv
        df = pd.read_csv(csv_path, low_memory=False)
        need = {"eid", "instance", "split", args.target_col, COL_LEFT, COL_RIGHT}
        miss = need - set(df.columns)
        if miss:
            raise ValueError(f"CSV 缺少列: {miss}")

        df = filter_target_valid(df, args.target_col)
        df = prepare_image_frame(df, fundus_root)
    exist_mask = df["fundus_image_path"].apply(lambda p: os.path.isfile(str(p)))
    n_miss = int((~exist_mask).sum())
    if n_miss:
        df = df.loc[exist_mask].reset_index(drop=True)
        print(f"[stage1] 跳过磁盘缺失图像: {n_miss}")
    if len(df) == 0:
        raise RuntimeError("无有效样本（检查 fundus_root 与 21015/21016 文件）")

    tr = split_subset(df, "train")
    va = split_subset(df, "val")
    te = split_subset(df, "test")
    print(
        f"[stage1] N train/val/test={len(tr)}/{len(va)}/{len(te)} target={args.target_col}"
    )

    ds_tr = Stage1BinaryDataset(
        tr,
        args.target_col,
        is_train=True,
        balance_train=args.balance_train,
        seed=args.seed,
    )
    ds_va = Stage1BinaryDataset(va, args.target_col, is_train=False)
    ds_te = Stage1BinaryDataset(te, args.target_col, is_train=False)

    loader_tr = torch.utils.data.DataLoader(
        ds_tr,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=len(ds_tr) > args.batch_size,
    )
    loader_va = torch.utils.data.DataLoader(
        ds_va,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    loader_te = torch.utils.data.DataLoader(
        ds_te,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = models_vit.vit_large_patch16(
        img_size=224,
        num_classes=1,
        drop_path_rate=0.1,
        global_pool=False,
    )
    ckpt_path = args.init_ckpt.strip() or default_ckpt(args.init_source)
    ckpt_path = str(REPO_ROOT / ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
    load_pretrained_vit(model, ckpt_path, args.init_source)
    apply_finetune_mode(model, args.mode)
    model.to(device)

    optimizer = build_optimizer(model, args.mode, args)
    loss_scaler = NativeScaler()

    y_tr = tr[args.target_col].values
    n_pos = int((y_tr == 1).sum())
    n_neg = int((y_tr == 0).sum())
    bce_criterion = None
    pw_used = None
    if args.loss_type == "bce_pos_weight":
        if args.pos_weight and args.pos_weight > 0:
            pw_used = float(args.pos_weight)
        else:
            pw_used = float(n_neg) / max(n_pos, 1)
        bce_criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pw_used], device=device)
        )
        print(
            f"[stage1] loss_type=bce_pos_weight pos_weight={pw_used:.6g} "
            f"(train n_pos={n_pos}, n_neg={n_neg})"
        )
        if args.smoothing > 0:
            print(
                "[stage1] 提示: bce_pos_weight 使用硬标签，--smoothing 在该模式下不参与损失计算"
            )
    else:
        print(
            f"[stage1] loss_type=paper_bce label_smoothing={args.smoothing} "
            f"(train n_pos={n_pos}, n_neg={n_neg})"
        )

    train_ns = SimpleNamespace(
        accum_iter=1,
        clip_grad=None,
        lr=args.lr,
        min_lr=args.min_lr,
        epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        smoothing=args.smoothing,
        loss_type=args.loss_type,
        bce_criterion=bce_criterion,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_dump = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}
    if args.loss_type == "paper_bce":
        cfg_dump["loss_note"] = (
            "stage1 only: BCEWithLogits + soft labels (label smoothing), no pos_weight; "
            "num_classes=1. Does not modify main_finetune/contrastive losses."
        )
    else:
        cfg_dump["loss_note"] = (
            f"stage1 only: BCEWithLogitsLoss(pos_weight={pw_used}) hard labels; "
            "same spirit as linear_probe_clinical. Does not modify main_finetune/contrastive."
        )
    with open(out_dir / "train_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg_dump, f, indent=2, default=str)

    log_path = out_dir / "training_log.csv"
    start_epoch = 0
    last_path = out_dir / "checkpoint_last.pth"
    best_path = out_dir / "checkpoint_best.pth"

    if args.resume and last_path.is_file():
        chk = torch.load(last_path, map_location="cpu")
        model.load_state_dict(chk["model"])
        optimizer.load_state_dict(chk["optimizer"])
        loss_scaler.load_state_dict(chk["scaler"])
        start_epoch = int(chk["epoch"]) + 1
        if "best_auc" in chk:
            best_auc = float(chk["best_auc"])
            best_epoch = int(chk.get("best_epoch", -1))
        else:
            best_auc = -1.0
            best_epoch = -1
        print(f"[resume] 从 epoch {start_epoch} 继续 best_auc={best_auc}")
    else:
        best_auc = -1.0
        best_epoch = -1

    live_fields = [
        "timestamp",
        "target_col",
        "init_source",
        "mode",
        "epoch",
        "train_loss",
        "val_auroc",
        "val_auprc",
        "lr",
    ]
    final_fields = [
        "timestamp",
        "target_col",
        "init_source",
        "mode",
        "best_epoch",
        "n_train",
        "n_val",
        "n_test",
        "val_auroc",
        "val_auprc",
        "test_auroc",
        "test_auprc",
        "youden_thr",
        "test_sensitivity",
        "test_specificity",
        "test_ppv",
        "test_npv",
        "output_dir",
    ]

    def save_last(epoch_done: int):
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": loss_scaler.state_dict(),
                "epoch": epoch_done,
                "best_auc": best_auc,
                "best_epoch": best_epoch,
            },
            last_path,
        )

    for epoch in range(start_epoch, args.epochs):
        train_stats = train_one_epoch_bce(
            model,
            loader_tr,
            optimizer,
            device,
            epoch,
            loss_scaler,
            train_ns,
        )
        y_val, p_val = forward_probs(model, loader_va, device)
        v_auc = auc_safe(y_val, p_val)
        v_pr = auprc_safe(y_val, p_val)
        lr_now = max(g["lr"] for g in optimizer.param_groups)

        row_log = {
            "epoch": epoch,
            "train_loss": train_stats.get("loss", ""),
            "val_auroc": v_auc,
            "val_auprc": v_pr,
            "lr": lr_now,
        }
        print(
            f"[epoch {epoch}] loss={train_stats.get('loss', float('nan')):.4f} "
            f"val_auroc={v_auc:.4f} val_auprc={v_pr:.4f} lr={lr_now:.2e}"
        )

        with open(log_path, "a", newline="", encoding="utf-8") as lf:
            w = csv.DictWriter(lf, fieldnames=list(row_log.keys()))
            if not log_path.exists() or log_path.stat().st_size == 0:
                w.writeheader()
            w.writerow(row_log)

        if args.delivery_csv:
            _append_csv_locked(
                args.delivery_csv,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "target_col": args.target_col,
                    "init_source": args.init_source,
                    "mode": args.mode,
                    **row_log,
                },
                live_fields,
            )

        if not math.isnan(v_auc) and v_auc > best_auc:
            best_auc = v_auc
            best_epoch = epoch
            torch.save({"model": model.state_dict()}, best_path)
            print(f"  -> 保存 best val AUROC={best_auc:.4f} @epoch {epoch}")

        save_last(epoch)

    # ---- test with Youden threshold from val ----
    if best_path.is_file():
        wdict = torch.load(best_path, map_location="cpu")["model"]
    elif last_path.is_file():
        wdict = torch.load(last_path, map_location="cpu")["model"]
    else:
        raise RuntimeError("无可用 checkpoint")
    model.load_state_dict(wdict)
    y_val, p_val = forward_probs(model, loader_va, device)
    thr = youden_threshold(y_val, p_val)
    y_te, p_te = forward_probs(model, loader_te, device)
    t_auc = auc_safe(y_te, p_te)
    t_pr = auprc_safe(y_te, p_te)
    m = binary_metrics(y_te, p_te, thr)
    v_auc_final = auc_safe(y_val, p_val)
    v_pr_final = auprc_safe(y_val, p_val)

    final_row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target_col": args.target_col,
        "init_source": args.init_source,
        "mode": args.mode,
        "best_epoch": best_epoch,
        "n_train": len(ds_tr),
        "n_val": len(ds_va),
        "n_test": len(ds_te),
        "val_auroc": v_auc_final,
        "val_auprc": v_pr_final,
        "test_auroc": t_auc,
        "test_auprc": t_pr,
        "youden_thr": thr,
        "test_sensitivity": m["sensitivity"],
        "test_specificity": m["specificity"],
        "test_ppv": m["ppv"],
        "test_npv": m["npv"],
        "output_dir": str(out_dir),
    }
    print("[test]", final_row)

    if args.delivery_final_csv:
        _append_csv_locked(args.delivery_final_csv, final_row, final_fields)

    Path(out_dir / "DONE").write_text("ok\n", encoding="utf-8")


if __name__ == "__main__":
    if "--print_default_ckpts" in sys.argv:
        print_default_ckpt_report()
    else:
        main()
