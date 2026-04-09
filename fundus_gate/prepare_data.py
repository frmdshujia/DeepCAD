"""
眼底图像门控模型 - 数据准备脚本
功能：
  - 收集眼底正样本（来自UKB和SDPP数据集）
  - 下载CIFAR-100作为自然图像负样本
  - 转换ACDC心脏MRI npy文件为PNG负样本
  - 生成train/val划分的CSV文件
"""

import os
import csv
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image

# ─── 配置 ────────────────────────────────────────────────────────────────────
FUNDUS_DIRS = [
    "/data/home/shujia/CHD/data(docker)/UKB/pictures/positive",
    "/data/home/shujia/CHD/data(docker)/UKB/pictures/negative",
    "/data/home/shujia/CHD/data(docker)/SDPP_CTA/pictures/positive",
    "/data/home/shujia/CHD/data(docker)/SDPP_CTA/pictures/negative",
]

ACDC_DIR = "/data/home/shujia/dataset/CardiacMRI/public_set/ACDC"

OUTPUT_DIR = "/data/home/shujia/CHD/model_train/RETFound_MAE-main/fundus_gate"
NEG_SAVE_DIR = os.path.join(OUTPUT_DIR, "negative_samples")
CIFAR_SAVE_DIR = os.path.join(NEG_SAVE_DIR, "cifar100")
ACDC_SAVE_DIR = os.path.join(NEG_SAVE_DIR, "acdc_mri")

# 最终采样数量（正负各取多少进行训练）
MAX_FUNDUS = 10000        # 从所有眼底中随机抽取
MAX_NEG_CIFAR = 5000      # CIFAR-100 负样本数
MAX_NEG_ACDC = 2000       # ACDC MRI 负样本数
VAL_RATIO = 0.15          # 验证集比例
SEED = 42


def collect_fundus_images(max_count=MAX_FUNDUS):
    """从多个眼底目录收集图像路径"""
    all_paths = []
    for d in FUNDUS_DIRS:
        if not os.path.exists(d):
            print(f"  [跳过] 目录不存在: {d}")
            continue
        imgs = [
            str(p) for p in Path(d).rglob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
        ]
        print(f"  {d}: {len(imgs)} 张")
        all_paths.extend(imgs)

    print(f"眼底图像总计: {len(all_paths)} 张")
    random.shuffle(all_paths)
    if len(all_paths) > max_count:
        all_paths = all_paths[:max_count]
        print(f"  随机抽取 {max_count} 张作为正样本")
    return all_paths


def download_cifar100_as_images(save_dir, max_count=MAX_NEG_CIFAR):
    """下载CIFAR-100并保存为PNG图像（负样本）"""
    os.makedirs(save_dir, exist_ok=True)

    # 检查是否已有足够图像
    existing = list(Path(save_dir).glob("*.png"))
    if len(existing) >= max_count:
        print(f"CIFAR-100 已有 {len(existing)} 张图像，跳过下载")
        return [str(p) for p in existing[:max_count]]

    print(f"正在下载/加载 CIFAR-100 ...")
    try:
        import torchvision.datasets as dsets
        import torchvision.transforms as T

        cache_dir = os.path.join(OUTPUT_DIR, ".cifar_cache")
        dataset = dsets.CIFAR100(root=cache_dir, train=True, download=True)
        test_ds = dsets.CIFAR100(root=cache_dir, train=False, download=True)

        saved_paths = []
        indices = list(range(len(dataset) + len(test_ds)))
        random.shuffle(indices)

        for i, idx in enumerate(indices[:max_count]):
            if idx < len(dataset):
                img_arr, _ = dataset[idx]
            else:
                img_arr, _ = test_ds[idx - len(dataset)]

            # torchvision CIFAR100 直接返回PIL Image
            if not isinstance(img_arr, Image.Image):
                img_arr = Image.fromarray(img_arr)

            save_path = os.path.join(save_dir, f"cifar100_{i:06d}.png")
            img_arr.save(save_path)
            saved_paths.append(save_path)

            if (i + 1) % 1000 == 0:
                print(f"  已保存 {i+1}/{max_count} 张CIFAR-100图像")

        print(f"CIFAR-100 保存完成，共 {len(saved_paths)} 张")
        return saved_paths

    except Exception as e:
        print(f"[警告] CIFAR-100 下载失败: {e}")
        return []


def convert_acdc_npy_to_images(acdc_dir, save_dir, max_count=MAX_NEG_ACDC):
    """将ACDC心脏MRI的npy文件转换为PNG图像（负样本）"""
    os.makedirs(save_dir, exist_ok=True)

    existing = list(Path(save_dir).glob("*.png"))
    if len(existing) >= max_count:
        print(f"ACDC MRI 已有 {len(existing)} 张图像，跳过转换")
        return [str(p) for p in existing[:max_count]]

    if not os.path.exists(acdc_dir):
        print(f"[跳过] ACDC目录不存在: {acdc_dir}")
        return []

    npy_files = [f for f in Path(acdc_dir).glob("*data*.npy")]
    if not npy_files:
        print(f"[跳过] 未找到ACDC数据文件")
        return []

    print(f"正在转换 {len(npy_files)} 个ACDC npy文件 ...")
    saved_paths = []
    img_idx = 0

    for npy_file in npy_files:
        if img_idx >= max_count:
            break
        try:
            data = np.load(str(npy_file))  # shape: (N, H, W) or (N, H, W, C)
            if data.ndim == 3:
                # (N, H, W) - 灰度切片
                for slice_i in range(data.shape[0]):
                    if img_idx >= max_count:
                        break
                    slice_img = data[slice_i]
                    # 归一化到0-255
                    vmin, vmax = slice_img.min(), slice_img.max()
                    if vmax > vmin:
                        slice_img = ((slice_img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                    else:
                        slice_img = np.zeros_like(slice_img, dtype=np.uint8)

                    img = Image.fromarray(slice_img).convert("RGB")
                    save_path = os.path.join(save_dir, f"acdc_{img_idx:06d}.png")
                    img.save(save_path)
                    saved_paths.append(save_path)
                    img_idx += 1

            elif data.ndim == 4:
                # (N, H, W, C) or (N, C, H, W)
                for slice_i in range(data.shape[0]):
                    if img_idx >= max_count:
                        break
                    slice_img = data[slice_i]
                    if slice_img.shape[0] in (1, 3):  # (C, H, W)
                        slice_img = slice_img.transpose(1, 2, 0)
                    if slice_img.shape[-1] == 1:
                        slice_img = slice_img[:, :, 0]

                    vmin, vmax = slice_img.min(), slice_img.max()
                    if vmax > vmin:
                        slice_img = ((slice_img - vmin) / (vmax - vmin) * 255).astype(np.uint8)
                    else:
                        slice_img = np.zeros_like(slice_img, dtype=np.uint8)

                    img = Image.fromarray(slice_img).convert("RGB")
                    save_path = os.path.join(save_dir, f"acdc_{img_idx:06d}.png")
                    img.save(save_path)
                    saved_paths.append(save_path)
                    img_idx += 1
        except Exception as e:
            print(f"  [警告] 处理 {npy_file.name} 失败: {e}")

    print(f"ACDC MRI 转换完成，共 {len(saved_paths)} 张")
    return saved_paths


def create_csv_splits(fundus_paths, neg_paths, output_dir, val_ratio=VAL_RATIO):
    """生成 train.csv 和 val.csv"""
    random.seed(SEED)

    # 合并并打乱
    all_fundus = [(p, 1) for p in fundus_paths]
    all_neg = [(p, 0) for p in neg_paths]

    random.shuffle(all_fundus)
    random.shuffle(all_neg)

    # 划分
    def split(items, ratio):
        n_val = max(1, int(len(items) * ratio))
        return items[n_val:], items[:n_val]

    fundus_train, fundus_val = split(all_fundus, val_ratio)
    neg_train, neg_val = split(all_neg, val_ratio)

    train_data = fundus_train + neg_train
    val_data = fundus_val + neg_val

    random.shuffle(train_data)
    random.shuffle(val_data)

    def write_csv(path, data):
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows(data)
        print(f"  已写入: {path}  ({len(data)} 条)")

    write_csv(os.path.join(output_dir, "train.csv"), train_data)
    write_csv(os.path.join(output_dir, "val.csv"), val_data)

    pos_train = sum(1 for _, l in train_data if l == 1)
    neg_train_n = sum(1 for _, l in train_data if l == 0)
    pos_val = sum(1 for _, l in val_data if l == 1)
    neg_val_n = sum(1 for _, l in val_data if l == 0)

    print(f"\n数据集统计:")
    print(f"  训练集: {len(train_data)} (正样本={pos_train}, 负样本={neg_train_n})")
    print(f"  验证集: {len(val_data)}  (正样本={pos_val}, 负样本={neg_val_n})")


def main():
    parser = argparse.ArgumentParser(description="眼底门控模型数据准备")
    parser.add_argument("--max_fundus", type=int, default=MAX_FUNDUS)
    parser.add_argument("--max_neg_cifar", type=int, default=MAX_NEG_CIFAR)
    parser.add_argument("--max_neg_acdc", type=int, default=MAX_NEG_ACDC)
    parser.add_argument("--skip_cifar", action="store_true", help="跳过CIFAR-100下载")
    parser.add_argument("--skip_acdc", action="store_true", help="跳过ACDC转换")
    args = parser.parse_args()

    random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("步骤1: 收集眼底正样本")
    print("=" * 60)
    fundus_paths = collect_fundus_images(args.max_fundus)

    print("\n" + "=" * 60)
    print("步骤2: 准备负样本")
    print("=" * 60)
    neg_paths = []

    if not args.skip_cifar:
        print("\n[CIFAR-100 自然图像]")
        cifar_paths = download_cifar100_as_images(CIFAR_SAVE_DIR, args.max_neg_cifar)
        neg_paths.extend(cifar_paths)
    else:
        print("[跳过] CIFAR-100")
        existing = list(Path(CIFAR_SAVE_DIR).glob("*.png"))
        neg_paths.extend([str(p) for p in existing])
        print(f"  使用已有CIFAR图像: {len(existing)} 张")

    if not args.skip_acdc:
        print("\n[ACDC 心脏MRI]")
        acdc_paths = convert_acdc_npy_to_images(ACDC_DIR, ACDC_SAVE_DIR, args.max_neg_acdc)
        neg_paths.extend(acdc_paths)
    else:
        print("[跳过] ACDC MRI转换")
        existing = list(Path(ACDC_SAVE_DIR).glob("*.png"))
        neg_paths.extend([str(p) for p in existing])
        print(f"  使用已有ACDC图像: {len(existing)} 张")

    print(f"\n负样本总计: {len(neg_paths)} 张")

    if len(neg_paths) == 0:
        print("[错误] 没有负样本，请检查数据路径")
        return

    print("\n" + "=" * 60)
    print("步骤3: 生成train/val CSV")
    print("=" * 60)
    create_csv_splits(fundus_paths, neg_paths, OUTPUT_DIR)

    print("\n数据准备完成！")
    print(f"CSV文件保存在: {OUTPUT_DIR}/train.csv 和 val.csv")


if __name__ == "__main__":
    main()
