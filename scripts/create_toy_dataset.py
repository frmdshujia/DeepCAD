#!/usr/bin/env python3
"""
生成用于快速自检的玩具数据集。

该脚本会在指定根目录下创建 retina / mri / splits 三个子目录，
并随机生成一小批视网膜图像（RGB JPG）与对应的 MRI 切片张量（.pt），
同时写出 train / val 两个 CSV，可直接用于 `train_stage1.py` 的连通性测试。

注意：脚本依赖 numpy、torch、Pillow。若当前环境尚未安装，请先按照
`docs/ENVIRONMENT_SETUP.md` 或 `docs/QUICKSTART.md` 进行依赖安装。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 DeepCAD 玩具数据集")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data"),
        help="数据根目录（默认: ./data）",
    )
    parser.add_argument(
        "--num-train",
        type=int,
        default=4,
        help="训练集样本数量（默认: 4）",
    )
    parser.add_argument(
        "--num-val",
        type=int,
        default=2,
        help="验证集样本数量（默认: 2）",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="视网膜图像与 MRI 切片的空间尺寸（默认: 224）",
    )
    parser.add_argument(
        "--num-mri-slices",
        type=int,
        default=3,
        help="每个样本生成的 MRI 切片数量（默认: 3）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机数种子（默认: 42）",
    )
    return parser.parse_args()


def _make_dirs(root: Path) -> Tuple[Path, Path, Path]:
    retinal_dir = root / "retinal"
    mri_dir = root / "mri"
    splits_dir = root / "splits"
    for path in (retinal_dir, mri_dir, splits_dir):
        path.mkdir(parents=True, exist_ok=True)
    return retinal_dir, mri_dir, splits_dir


def _make_subject_ids(prefix: str, count: int) -> List[str]:
    return [f"{prefix}_{i:02d}" for i in range(1, count + 1)]


def _save_retinal_image(path: Path, rng: np.random.Generator, size: int) -> None:
    arr = (rng.random((size, size, 3)) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="JPEG")


def _save_mri_tensor(path: Path, rng: np.random.Generator, slices: int, size: int) -> None:
    tensor = torch.from_numpy(rng.random((slices, size, size), dtype=np.float32))
    torch.save(tensor, path)


def _write_csv(csv_path: Path, rows: List[Tuple[str, str, str, int]]) -> None:
    header = "subject_id,retinal_path,mri_paths,label\n"
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(header)
        for subject_id, retinal_rel, mri_rel, label in rows:
            # mri_paths 支持 JSON / CSV / 单路径，这里直接写 JSON 列表便于扩展
            mri_field = json.dumps([mri_rel])
            f.write(f"{subject_id},{retinal_rel},{mri_field},{label}\n")


def main() -> None:
    args = _parse_args()
    rng = np.random.default_rng(args.seed)

    retinal_dir, mri_dir, splits_dir = _make_dirs(args.root)

    train_ids = _make_subject_ids("train_subj", args.num_train)
    val_ids = _make_subject_ids("val_subj", args.num_val)

    all_rows: List[Tuple[str, str, str, int]] = []

    for subject_id in train_ids + val_ids:
        label = int(rng.integers(0, 2))
        retinal_filename = f"{subject_id}.jpg"
        mri_filename = f"{subject_id}_slices.pt"

        _save_retinal_image(retinal_dir / retinal_filename, rng, args.image_size)
        _save_mri_tensor(mri_dir / mri_filename, rng, args.num_mri_slices, args.image_size)

        all_rows.append(
            (
                subject_id,
                f"retinal/{retinal_filename}",
                f"mri/{mri_filename}",
                label,
            )
        )

    train_rows = all_rows[: len(train_ids)]
    val_rows = all_rows[len(train_ids) :]

    _write_csv(splits_dir / "train_toy.csv", train_rows)
    _write_csv(splits_dir / "val_toy.csv", val_rows)

    print(
        f"[OK] 已在 {args.root.resolve()} 下生成 {len(train_rows)} 条训练样本、"
        f"{len(val_rows)} 条验证样本。"
    )
    print("可通过 scripts/train_stage1.py 与 --*_base_path data 选项进行快速连通性测试。")


if __name__ == "__main__":
    main()

