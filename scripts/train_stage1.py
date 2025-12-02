#!/usr/bin/env python3
"""
DeepCAD Stage I 训练脚本
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deepcad_stage1 import DeepCADStageI
from datasets import create_dataloaders
from losses import CrossModalContrastiveLoss
from trainers.stage1_trainer import Stage1Trainer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepCAD Stage I 训练')
    
    # 数据参数
    parser.add_argument('--train_csv', type=str, required=True,
                       help='训练集CSV路径')
    parser.add_argument('--val_csv', type=str, default=None,
                       help='验证集CSV路径（可选）')
    parser.add_argument('--test_csv', type=str, default=None,
                       help='测试集CSV路径（可选）')
    parser.add_argument('--retinal_base_path', type=str, default=None,
                       help='视网膜图像基础路径')
    parser.add_argument('--mri_base_path', type=str, default=None,
                       help='MRI数据基础路径')
    
    # 模型参数
    parser.add_argument('--retinal_pretrained', type=str, default=None,
                       help='RETFound预训练权重路径（可选，HuggingFace Hub ID或本地路径）')
    parser.add_argument('--mri_pretrained', type=str, default=None,
                       help='MedSAM预训练权重路径（可选）')
    parser.add_argument('--retinal_img_size', type=int, default=224,
                       help='视网膜图像尺寸')
    parser.add_argument('--mri_img_size', type=int, default=224,
                       help='MRI图像尺寸')
    parser.add_argument('--max_mri_slices', type=int, default=10,
                       help='最大MRI切片数量')
    parser.add_argument('--mri_pooling_type', type=str, default='attention',
                       choices=['attention', 'learnable_weighted', 'mean', 'max'],
                       help='MRI池化类型')
    parser.add_argument('--latent_dim', type=int, default=128,
                       help='共享潜在空间维度')
    parser.add_argument('--projection_hidden_dim', type=int, default=None,
                       help='投影头隐藏层维度（默认使用编码器输出维度）')
    parser.add_argument('--projection_num_layers', type=int, default=2,
                       help='投影头层数')
    
    # 训练参数
    parser.add_argument('--batch_size', type=int, default=32,
                       help='批次大小')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='对比损失温度参数')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='优化器类型')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='学习率调度器类型')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='预热轮数（用于cosine调度器）')
    parser.add_argument('--freeze_encoders', action='store_true',
                       help='冻结编码器参数（只训练投影头）')
    
    # 数据加载参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--pin_memory', action='store_true', default=True,
                       help='是否固定内存')
    parser.add_argument('--retinal_augmentation', type=str, default='medium',
                       choices=['light', 'medium', 'strong'],
                       help='视网膜图像增强强度')
    parser.add_argument('--mri_augmentation', type=str, default='medium',
                       choices=['light', 'medium', 'strong'],
                       help='MRI图像增强强度')
    
    # 保存和日志
    parser.add_argument('--save_dir', type=str, default='checkpoints/stage1',
                       help='检查点保存目录')
    parser.add_argument('--log_dir', type=str, default='logs/stage1',
                       help='日志目录')
    parser.add_argument('--resume_from', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--save_interval', type=int, default=5,
                       help='检查点保存间隔（每N个epoch）')
    parser.add_argument('--log_interval', type=int, default=10,
                       help='日志记录间隔（每N个批次）')
    
    # 设备
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu，默认自动选择）')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    return parser.parse_args()


def create_optimizer(model: nn.Module, args) -> optim.Optimizer:
    """创建优化器"""
    if args.optimizer.lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer.lower() == 'sgd':
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"未知的优化器类型: {args.optimizer}")
    
    return optimizer


def create_scheduler(optimizer: optim.Optimizer, args, num_epochs: int):
    """创建学习率调度器"""
    if args.scheduler.lower() == 'none':
        return None
    elif args.scheduler.lower() == 'cosine':
        from torch.optim.lr_scheduler import CosineAnnealingLR
        return CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif args.scheduler.lower() == 'step':
        from torch.optim.lr_scheduler import StepLR
        return StepLR(optimizer, step_size=num_epochs // 3, gamma=0.1)
    elif args.scheduler.lower() == 'plateau':
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        return ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    else:
        raise ValueError(f"未知的调度器类型: {args.scheduler}")


def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设置设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("DeepCAD Stage I 训练")
    print("=" * 60)
    print(f"设备: {device}")
    print(f"随机种子: {args.seed}")
    print()
    
    # 创建数据加载器
    print("加载数据集...")
    train_loader, val_loader, test_loader = create_dataloaders(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        retinal_img_size=args.retinal_img_size,
        mri_img_size=args.mri_img_size,
        retinal_augmentation=args.retinal_augmentation,
        mri_augmentation=args.mri_augmentation,
        max_mri_slices=args.max_mri_slices,
        live_loading=True,
        retinal_base_path=args.retinal_base_path,
        mri_base_path=args.mri_base_path
    )
    
    print(f"训练集: {len(train_loader.dataset)} 样本")
    if val_loader:
        print(f"验证集: {len(val_loader.dataset)} 样本")
    if test_loader:
        print(f"测试集: {len(test_loader.dataset)} 样本")
    print()
    
    # 创建模型
    print("创建模型...")
    model = DeepCADStageI(
        retinal_pretrained_path=args.retinal_pretrained,
        retinal_img_size=args.retinal_img_size,
        retinal_freeze_backbone=args.freeze_encoders,
        mri_pretrained_path=args.mri_pretrained,
        mri_img_size=args.mri_img_size,
        mri_pooling_type=args.mri_pooling_type,
        mri_freeze_backbone=args.freeze_encoders,
        latent_dim=args.latent_dim,
        projection_hidden_dim=args.projection_hidden_dim,
        projection_num_layers=args.projection_num_layers
    )
    
    if args.freeze_encoders:
        model.freeze_encoders()
        print("编码器已冻结，只训练投影头")
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"可训练参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print()
    
    # 创建损失函数
    criterion = CrossModalContrastiveLoss(tau=args.temperature)
    
    # 创建优化器
    optimizer = create_optimizer(model, args)
    
    # 创建学习率调度器
    scheduler = create_scheduler(optimizer, args, args.num_epochs)
    
    # 创建训练器
    trainer = Stage1Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        log_dir=args.log_dir,
        save_dir=args.save_dir,
        log_interval=args.log_interval,
        save_interval=args.save_interval
    )
    
    # 开始训练
    trainer.train(
        num_epochs=args.num_epochs,
        resume_from=args.resume_from
    )
    
    print("\n训练完成！")
    print(f"最佳模型保存在: {os.path.join(args.save_dir, 'best_model.pth')}")


if __name__ == "__main__":
    main()

