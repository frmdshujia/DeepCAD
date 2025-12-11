#!/usr/bin/env python3
"""
Grad-CAM 和跨模态可视化脚本
"""

import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deepcad_stage1 import DeepCADStageI
from datasets import RetinaCardiacDataset
from explainability import VitGradCAM
from explainability.cross_modal_viz import visualize_cross_modal, visualize_batch
from utils.checkpoint import load_checkpoint


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepCAD Grad-CAM 可视化')
    
    # 模型参数
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='模型检查点路径')
    parser.add_argument('--retinal_pretrained', type=str, default=None,
                       help='RETFound预训练权重路径（如果检查点中没有）')
    parser.add_argument('--mri_pretrained', type=str, default=None,
                       help='MedSAM预训练权重路径（如果检查点中没有）')
    
    # 数据参数
    parser.add_argument('--data_csv', type=str, default=None,
                       help='数据CSV路径（用于批量可视化）')
    parser.add_argument('--retinal_image', type=str, default=None,
                       help='单个视网膜图像路径')
    parser.add_argument('--mri_paths', type=str, nargs='+', default=None,
                       help='MRI切片路径列表')
    parser.add_argument('--retinal_base_path', type=str, default=None,
                       help='视网膜图像基础路径')
    parser.add_argument('--mri_base_path', type=str, default=None,
                       help='MRI数据基础路径')
    
    # 可视化参数
    parser.add_argument('--num_samples', type=int, default=4,
                       help='批量可视化时的样本数量')
    parser.add_argument('--save_dir', type=str, default='visualizations',
                       help='保存目录')
    parser.add_argument('--target_type', type=str, default='similarity',
                       choices=['similarity', 'embedding', 'auto'],
                       help='Grad-CAM目标类型')
    
    # 其他参数
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu，默认自动选择）')
    parser.add_argument('--img_size', type=int, default=224,
                       help='图像尺寸')
    
    return parser.parse_args()


def load_model(checkpoint_path: str, args) -> DeepCADStageI:
    """加载模型"""
    print(f"加载模型检查点: {checkpoint_path}")
    
    checkpoint = load_checkpoint(checkpoint_path)
    
    # 创建模型（需要从检查点或参数中获取配置）
    model = DeepCADStageI(
        retinal_pretrained_path=args.retinal_pretrained,
        retinal_img_size=args.img_size,
        mri_pretrained_path=args.mri_pretrained,
        mri_img_size=args.img_size,
        latent_dim=128  # 默认值，可以从检查点中读取
    )
    
    # 加载权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print("模型加载完成")
    return model


def load_single_image(image_path: str, img_size: int = 224) -> torch.Tensor:
    """加载单个图像"""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def visualize_single(
    model: DeepCADStageI,
    retinal_image_path: str,
    mri_paths: list,
    args
):
    """可视化单个样本"""
    print("加载图像...")
    
    # 加载视网膜图像
    retinal_image = load_single_image(retinal_image_path, args.img_size)
    
    # 加载MRI切片（简化版本，实际应该使用数据集加载器）
    # 这里假设MRI已经预处理为张量
    print("注意: MRI切片加载需要根据实际数据格式实现")
    
    # 创建 Grad-CAM
    print("生成 Grad-CAM...")
    gradcam = VitGradCAM(model, use_cuda=(args.device == "cuda"))
    
    try:
        cam = gradcam.generate_cam(
            retinal_image.to(args.device),
            target_type=args.target_type
        )
        
        # 可视化
        save_path = os.path.join(args.save_dir, "gradcam_single.png")
        os.makedirs(args.save_dir, exist_ok=True)
        
        # 加载原始图像用于叠加
        original_image = Image.open(retinal_image_path).convert('RGB')
        original_image = original_image.resize((args.img_size, args.img_size))
        original_np = np.array(original_image)
        
        # 叠加热图
        overlay = gradcam.overlay_heatmap(original_np, cam)
        
        # 保存
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(original_image)
        plt.title('Original Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(overlay)
        plt.title('Grad-CAM Overlay')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存: {save_path}")
        plt.close()
        
    except Exception as e:
        print(f"生成 Grad-CAM 失败: {e}")
        import traceback
        traceback.print_exc()


def visualize_from_dataset(
    model: DeepCADStageI,
    data_csv: str,
    args
):
    """从数据集批量可视化"""
    from torch.utils.data import DataLoader
    
    print(f"从数据集加载: {data_csv}")
    
    # 创建数据集
    dataset = RetinaCardiacDataset(
        data_csv=data_csv,
        retinal_img_size=args.img_size,
        mri_img_size=args.img_size,
        train=False,  # 不使用数据增强
        live_loading=True,
        retinal_base_path=args.retinal_base_path,
        mri_base_path=args.mri_base_path
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0  # 可视化时使用单线程
    )
    
    # 批量可视化
    visualize_batch(
        model=model,
        dataloader=dataloader,
        num_samples=args.num_samples,
        device=args.device,
        save_dir=args.save_dir
    )


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("DeepCAD Grad-CAM 可视化")
    print("=" * 60)
    print(f"设备: {args.device}")
    print()
    
    # 加载模型
    model = load_model(args.checkpoint, args)
    model.to(args.device)
    model.eval()
    
    # 可视化
    if args.data_csv:
        # 从数据集批量可视化
        visualize_from_dataset(model, args.data_csv, args)
    elif args.retinal_image:
        # 单个图像可视化
        visualize_single(model, args.retinal_image, args.mri_paths, args)
    else:
        print("错误: 必须提供 --data_csv 或 --retinal_image")
        return
    
    print("\n可视化完成！")


if __name__ == "__main__":
    main()

