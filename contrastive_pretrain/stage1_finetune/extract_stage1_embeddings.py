#!/usr/bin/env python3
"""
冻结 ViT backbone，无梯度前向，把每张图的 CLS embedding (1024) 导出为 .pt。
同一套 embedding 可供全部下游任务复用（Mode1 线性头训练），每个 init 各导一份。

用法示例：
  python contrastive_pretrain/stage1_finetune/extract_stage1_embeddings.py \\
    --init_source retfound --gpu 0 --batch_size 128 \\
    --output_pt output_dir/stage1_emb_cache/emb_retfound.pt
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import models_vit  # noqa: E402
import util.misc as misc  # noqa: E402
from timm.models.layers import trunc_normal_  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from contrastive_pretrain.stage1_finetune.stage1_dataset import Stage1BinaryDataset  # noqa: E402
from contrastive_pretrain.stage1_finetune.stage1_paths import (  # noqa: E402
    COL_LEFT,
    COL_RIGHT,
    COMPOSITE_TASKS,
    ICD_PREVALENT_TASKS,
    prepare_image_frame,
)
from contrastive_pretrain.stage1_finetune.stage1_train_one import (  # noqa: E402
    default_ckpt,
    load_pretrained_vit,
)


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
    )
    p.add_argument(
        "--init_source",
        type=str,
        choices=["retfound", "controlled", "no_residual"],
        required=True,
    )
    p.add_argument("--init_ckpt", type=str, default="")
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--output_pt", type=str, required=True)
    p.add_argument("--fp16_storage", action="store_true", help="保存为 float16 减小体积")
    return p.parse_args()


@torch.no_grad()
def run_extract():
    args = parse_args()
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

    csv_path = REPO_ROOT / args.stage1_csv
    df0 = pd.read_csv(csv_path, low_memory=False)
    need = {"eid", "instance", "split", COL_LEFT, COL_RIGHT}
    miss = need - set(df0.columns)
    if miss:
        raise ValueError(f"CSV 缺少列: {miss}")

    # 不因某一任务标签过滤行：保证 embedding 覆盖所有后续任务可能用到的样本
    df = prepare_image_frame(df0, args.fundus_root)
    exist_mask = df["fundus_image_path"].apply(lambda p: os.path.isfile(str(p)))
    n_miss = int((~exist_mask).sum())
    if n_miss:
        df = df.loc[exist_mask].reset_index(drop=True)
        print(f"[extract] 跳过缺失图像: {n_miss}")
    n = len(df)
    print(f"[extract] 行数 N={n} init={args.init_source}")

    df = df.copy()
    df["_emb_dummy_target"] = 0
    ds = Stage1BinaryDataset(
        df, "_emb_dummy_target", is_train=False, input_size=224, balance_train=False
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model = models_vit.vit_large_patch16(
        img_size=224, num_classes=1, drop_path_rate=0.0, global_pool=False
    )
    ckpt_path = args.init_ckpt.strip() or default_ckpt(args.init_source)
    ckpt_path = str(REPO_ROOT / ckpt_path) if not os.path.isabs(ckpt_path) else ckpt_path
    load_pretrained_vit(model, ckpt_path, args.init_source)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.to(device)

    chunks = []
    paths_order = []
    for batch in tqdm(loader, desc="embedding"):
        paths, images, _y = batch
        paths_order.extend(paths)
        images = images.to(device, non_blocking=True)
        with torch.cuda.amp.autocast():
            z = model.forward_features(images)
        chunks.append(z.float().cpu())

    emb = torch.cat(chunks, dim=0)
    assert emb.shape[0] == n, (emb.shape[0], n)
    if args.fp16_storage:
        emb = emb.half()

    out = {
        "embeddings": emb,
        "fundus_image_path": df["fundus_image_path"].astype(str).tolist(),
        "split": df["split"].astype(str).tolist(),
        "eid": df["eid"].values,
        "init_source": args.init_source,
        "num_rows": int(n),
        "embed_dim": int(emb.shape[1]),
    }
    label_cols = list(COMPOSITE_TASKS) + list(ICD_PREVALENT_TASKS)
    for c in label_cols:
        if c in df.columns:
            out[f"label__{c}"] = pd.to_numeric(df[c], errors="coerce").values

    os.makedirs(os.path.dirname(args.output_pt) or ".", exist_ok=True)
    torch.save(out, args.output_pt)
    sz_mb = os.path.getsize(args.output_pt) / (1024 * 1024)
    print(f"[extract] 已保存 {args.output_pt} ({sz_mb:.1f} MiB) shape={tuple(emb.shape)}")


if __name__ == "__main__":
    run_extract()
