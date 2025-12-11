#!/usr/bin/env python3
"""
DeepCAD Stage I 表征可视化脚本

功能：
1. 从训练好的 DeepCADStageI 检查点导出眼底/MRI 投影向量
2. 运行 t-SNE 将高维嵌入降至 2D，方便观察对比学习效果
3. 同时保存绘图 PNG 和包含元信息的 CSV，便于后续分析

使用示例：
python scripts/evaluate_tsne.py \
    --checkpoint checkpoints/stage1/best_model.pth \
    --data_csv data/splits/val_toy.csv \
    --retinal_pretrained checkpoints/pretrained/retfound/RETFound_cfp_weights.pth \
    --mri_pretrained checkpoints/pretrained/medsam/medsam_vit_b.pth \
    --retinal_base_path data \
    --mri_base_path data \
    --output_dir outputs/tsne_stage1 \
    --severity_column severity_score
"""

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

# 添加项目根路径，便于模块导入
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from datasets import RetinaCardiacDataset  # noqa: E402
from models.deepcad_stage1 import DeepCADStageI  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepCAD Stage I t-SNE 可视化")
    parser.add_argument("--checkpoint", type=str, required=True, help="训练好的模型检查点路径")
    parser.add_argument("--data_csv", type=str, required=True, help="用于可视化的数据 CSV（建议使用验证/测试集）")
    parser.add_argument("--retinal_pretrained", type=str, required=True, help="RETFound 预训练权重路径")
    parser.add_argument("--mri_pretrained", type=str, required=True, help="MedSAM 预训练权重路径")
    parser.add_argument("--retinal_base_path", type=str, default=None, help="眼底图像基础路径")
    parser.add_argument("--mri_base_path", type=str, default=None, help="MRI 数据基础路径")
    parser.add_argument("--retinal_img_size", type=int, default=224, help="眼底图像尺寸")
    parser.add_argument("--mri_img_size", type=int, default=224, help="MRI 图像尺寸（将被上采样至 MedSAM 需求）")
    parser.add_argument("--max_mri_slices", type=int, default=10, help="MRI 最大切片数（与训练时保持一致）")
    parser.add_argument("--batch_size", type=int, default=8, help="推理批次大小")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader worker 数量")
    parser.add_argument("--device", type=str, default=None, help="推理设备（默认自动检测）")
    parser.add_argument("--limit", type=int, default=None, help="最多可视化多少个受试者（None 表示全部）")
    parser.add_argument("--severity_column", type=str, default=None, help="CSV 中记录严重程度的列名（可选）")
    parser.add_argument("--output_dir", type=str, required=True, help="结果输出目录")
    parser.add_argument("--tsne_perplexity", type=float, default=30.0, help="t-SNE perplexity 参数")
    parser.add_argument("--tsne_lr", type=float, default=200.0, help="t-SNE 学习率")
    parser.add_argument("--tsne_metric", type=str, default="euclidean", help="t-SNE 距离度量")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def load_metadata(csv_path: str, severity_column: Optional[str]) -> Dict[str, Dict[str, Optional[float]]]:
    """读取 CSV，返回 subject -> metadata 映射"""
    df = pd.read_csv(csv_path)
    metadata: Dict[str, Dict[str, Optional[float]]] = {}
    for _, row in df.iterrows():
        subject = str(row["subject_id"])
        entry = {"label": int(row["label"]) if "label" in row else None}
        if severity_column and severity_column in row:
            entry["severity"] = row[severity_column]
        else:
            entry["severity"] = None
        metadata[subject] = entry
    return metadata


def create_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = RetinaCardiacDataset(
        data_csv=args.data_csv,
        retinal_img_size=args.retinal_img_size,
        mri_img_size=args.mri_img_size,
        train=False,
        retinal_augmentation="light",
        mri_augmentation="light",
        max_mri_slices=args.max_mri_slices,
        mri_slice_type="representative",
        live_loading=True,
        retinal_base_path=args.retinal_base_path,
        mri_base_path=args.mri_base_path,
    )
    if args.limit:
        dataset.df = dataset.df.head(args.limit)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )


def build_model(args: argparse.Namespace, device: torch.device) -> DeepCADStageI:
    model = DeepCADStageI(
        retinal_pretrained_path=args.retinal_pretrained,
        retinal_img_size=args.retinal_img_size,
        retinal_freeze_backbone=False,
        mri_pretrained_path=args.mri_pretrained,
        mri_img_size=args.mri_img_size,
        mri_pooling_type="attention",
        mri_freeze_backbone=False,
        latent_dim=128,
    )
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval()


def collect_embeddings(
    model: DeepCADStageI,
    dataloader: DataLoader,
    device: torch.device,
    metadata: Dict[str, Dict[str, Optional[float]]],
) -> pd.DataFrame:
    retinal_embeds: List[np.ndarray] = []
    mri_embeds: List[np.ndarray] = []
    modalities: List[str] = []
    subjects: List[str] = []
    labels: List[Optional[int]] = []
    severities: List[Optional[float]] = []

    with torch.no_grad():
        for batch in dataloader:
            x_R = batch["x_R"].to(device)
            x_C = batch["x_C"].to(device)
            subject_ids = batch["subject_id"]
            batch_labels = batch["y"].cpu().numpy().tolist()

            outputs = model(x_R, x_C)
            z_R = outputs["z_R"].cpu().numpy()
            z_C = outputs["z_C"].cpu().numpy()

            for idx, subject in enumerate(subject_ids):
                subject_str = str(subject)
                label = batch_labels[idx]
                severity = metadata.get(subject_str, {}).get("severity") if metadata else None

                retinal_embeds.append(z_R[idx])
                mri_embeds.append(z_C[idx])

                subjects.extend([subject_str, subject_str])
                labels.extend([label, label])
                severities.extend([severity, severity])
                modalities.extend(["retinal", "mri"])

    embeddings = np.vstack([np.stack(retinal_embeds), np.stack(mri_embeds)])
    df = pd.DataFrame({
        "tsne_input_idx": np.arange(len(embeddings)),
        "subject_id": subjects,
        "label": labels,
        "severity": severities,
        "modality": modalities,
    })
    df["tsne_x"] = np.nan
    df["tsne_y"] = np.nan
    return embeddings, df


def run_tsne(embeddings: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        learning_rate=args.tsne_lr,
        metric=args.tsne_metric,
        init="pca",
        random_state=args.seed,
        n_iter=1000,
    )
    return tsne.fit_transform(embeddings)


def plot_results(df: pd.DataFrame, output_dir: str) -> None:
    plt.figure(figsize=(8, 6))
    for modality, marker in [("retinal", "o"), ("mri", "^")]:
        subset = df[df["modality"] == modality]
        plt.scatter(
            subset["tsne_x"],
            subset["tsne_y"],
            label=modality,
            alpha=0.7,
            s=30,
            marker=marker,
        )
    plt.legend()
    plt.title("t-SNE by Modality")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tsne_modality.png"), dpi=300)
    plt.close()

    if df["severity"].notna().any():
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(
            df["tsne_x"],
            df["tsne_y"],
            c=df["severity"],
            cmap="viridis",
            alpha=0.75,
            s=30,
        )
        plt.colorbar(scatter, label="Severity")
        plt.title("t-SNE colored by Severity")
        plt.xlabel("Dimension 1")
        plt.ylabel("Dimension 2")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "tsne_severity.png"), dpi=300)
        plt.close()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    os.makedirs(args.output_dir, exist_ok=True)

    metadata = load_metadata(args.data_csv, args.severity_column) if args.severity_column else {}
    dataloader = create_dataloader(args)
    model = build_model(args, device)

    embeddings, df = collect_embeddings(model, dataloader, device, metadata)
    tsne_coords = run_tsne(embeddings, args)
    df.loc[:, "tsne_x"] = tsne_coords[:, 0]
    df.loc[:, "tsne_y"] = tsne_coords[:, 1]

    df.to_csv(os.path.join(args.output_dir, "tsne_embeddings.csv"), index=False)
    plot_results(df, args.output_dir)
    print(f"[OK] t-SNE 结果已保存至 {args.output_dir}")


if __name__ == "__main__":
    main()

