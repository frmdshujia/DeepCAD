"""
眼底图像门控模型 - 训练脚本
模型：EfficientNet-B0（轻量级二分类）
标签：1=眼底图像，0=非眼底图像
"""

import os
import csv
import time
import random
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

# ─── 日志设置 ─────────────────────────────────────────────────────────────────
def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


# ─── 数据集 ───────────────────────────────────────────────────────────────────
class FundusGateDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append((row["path"], int(row["label"])))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label


# ─── 数据增强 ─────────────────────────────────────────────────────────────────
def get_transforms(img_size=224, train=True):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomCrop(img_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ─── 模型 ─────────────────────────────────────────────────────────────────────
def build_model(arch="resnet18", pretrained=True):
    """构建二分类模型（优先使用本地缓存权重）"""
    import torchvision.models as tvm

    # 本地缓存权重路径映射
    _LOCAL_WEIGHTS = {
        "resnet18": "/data/home/shujia/.cache/torch/hub/checkpoints/resnet18-f37072fd.pth",
        "resnet50": "/data/home/shujia/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth",
    }

    arch_lower = arch.lower()

    # 优先尝试 torchvision 内置模型（利用本地缓存）
    tv_models = {
        "resnet18": (tvm.resnet18, lambda m: setattr(m, "fc", nn.Linear(m.fc.in_features, 1))),
        "resnet50": (tvm.resnet50, lambda m: setattr(m, "fc", nn.Linear(m.fc.in_features, 1))),
    }

    if arch_lower in tv_models:
        model_fn, head_replacer = tv_models[arch_lower]
        local_ckpt = _LOCAL_WEIGHTS.get(arch_lower)

        if pretrained and local_ckpt and os.path.exists(local_ckpt):
            # 从本地加载预训练权重
            model = model_fn(pretrained=False)
            state = torch.load(local_ckpt, map_location="cpu")
            model.load_state_dict(state)
            print(f"使用本地预训练权重: {local_ckpt}")
        elif pretrained:
            model = model_fn(pretrained=True)
            print(f"使用 torchvision {arch_lower}（在线下载预训练权重）")
        else:
            model = model_fn(pretrained=False)
            print(f"使用 torchvision {arch_lower}（随机初始化）")

        head_replacer(model)
        return model

    # 回退到 timm
    if TIMM_AVAILABLE:
        model = timm.create_model(arch, pretrained=pretrained, num_classes=1)
        print(f"使用 timm 模型: {arch}")
        return model

    raise ValueError(f"不支持的模型架构: {arch}")


# ─── 训练/验证循环 ─────────────────────────────────────────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True, logger=None):
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.float().to(device)

            logits = model(imgs).squeeze(1)
            loss = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = (torch.sigmoid(logits) >= 0.5).long()
            correct += (preds == labels.long()).sum().item()
            total += len(labels)
            total_loss += loss.item() * len(labels)

    avg_loss = total_loss / total if total > 0 else 0
    acc = correct / total if total > 0 else 0
    return avg_loss, acc


# ─── 主函数 ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="眼底门控模型训练")
    parser.add_argument("--train_csv", default="train.csv")
    parser.add_argument("--val_csv", default="val.csv")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--arch", default="efficientnet_b0",
                        help="模型架构 (efficientnet_b0 / mobilenetv3_small_100)")
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pos_weight", type=float, default=1.0,
                        help="正样本权重（若负样本多可设>1）")
    parser.add_argument("--no_pretrain", action="store_true")
    args = parser.parse_args()

    # 路径处理（相对于脚本所在目录）
    script_dir = os.path.dirname(os.path.abspath(__file__))
    train_csv = args.train_csv if os.path.isabs(args.train_csv) else os.path.join(script_dir, args.train_csv)
    val_csv = args.val_csv if os.path.isabs(args.val_csv) else os.path.join(script_dir, args.val_csv)
    output_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(script_dir, args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    # 日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(output_dir, f"train_{timestamp}.log")
    logger = setup_logger(log_path)
    logger.info(f"参数: {vars(args)}")

    # 随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # 数据集
    train_tf = get_transforms(args.img_size, train=True)
    val_tf = get_transforms(args.img_size, train=False)

    train_ds = FundusGateDataset(train_csv, train_tf)
    val_ds = FundusGateDataset(val_csv, val_tf)
    logger.info(f"训练集: {len(train_ds)} 样本  验证集: {len(val_ds)} 样本")

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        prefetch_factor=2 if args.num_workers > 0 else None,
        persistent_workers=True if args.num_workers > 0 else False,
    )

    # 模型
    model = build_model(args.arch, pretrained=not args.no_pretrain)
    model = model.to(device)

    # 损失函数（BCEWithLogitsLoss 支持正样本权重）
    pos_weight = torch.tensor([args.pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 优化器 + 调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练
    best_val_acc = 0.0
    best_ckpt_path = os.path.join(output_dir, "best_fundus_gate.pth")

    logger.info("=" * 60)
    logger.info("开始训练")
    logger.info("=" * 60)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"lr={lr_now:.2e} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
            f"time={elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "arch": args.arch,
                "img_size": args.img_size,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, best_ckpt_path)
            logger.info(f"  ✓ 保存最优模型 (val_acc={val_acc:.4f}) -> {best_ckpt_path}")

    # 保存最终模型
    final_ckpt_path = os.path.join(output_dir, "last_fundus_gate.pth")
    torch.save({
        "epoch": args.epochs,
        "arch": args.arch,
        "img_size": args.img_size,
        "state_dict": model.state_dict(),
        "val_acc": val_acc,
    }, final_ckpt_path)

    logger.info("=" * 60)
    logger.info(f"训练完成！最优验证准确率: {best_val_acc:.4f}")
    logger.info(f"最优模型: {best_ckpt_path}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
