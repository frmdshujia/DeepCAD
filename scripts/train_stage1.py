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
    parser.add_argument('--subject_column', type=str, default='subject_id',
                       help='CSV中受试者ID列名')
    parser.add_argument('--retinal_column', type=str, default='retinal_path',
                       help='CSV中视网膜图像路径列名（可为单路径或列表字符串）')
    parser.add_argument('--mri_paths_column', type=str, default='mri_paths',
                       help='CSV中心脏MRI路径列名（可为单路径或列表字符串；当未使用独立npy列时生效）')
    parser.add_argument('--label_column', type=str, default='label',
                       help='CSV中CAD标签列名')
    parser.add_argument('--grade_column', type=str, default=None,
                       help='CSV中疾病严重程度列名（可选）')
    parser.add_argument('--t1map_npy_column', type=str, default=None,
                       help='CSV中T1MAP序列npy路径列名（例如 T1MAP_B1B2B3_npy，可选）')
    parser.add_argument('--short_axis_npy_column', type=str, default=None,
                       help='CSV中短轴序列npy路径列名（例如 short_axis_ED_mid_ES_npy，可选）')
    parser.add_argument('--retinal_base_path', type=str, default=None,
                       help='视网膜图像基础路径')
    parser.add_argument('--mri_base_path', type=str, default=None,
                       help='MRI数据基础路径')
    
    # 模型参数
    parser.add_argument('--retinal_pretrained', type=str, required=True,
                       help='RETFound预训练权重路径（必填，可为HuggingFace Hub ID或本地路径）')
    parser.add_argument('--mri_pretrained', type=str, required=True,
                       help='MedSAM预训练权重路径（必填）')
    parser.add_argument('--retinal_img_size', type=int, default=224,
                       help='视网膜图像尺寸')
    parser.add_argument('--mri_img_size', type=int, default=224,
                       help='MRI图像尺寸')
    parser.add_argument('--max_mri_slices', type=int, default=6,
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
                       help='批次大小') # batch 越大，负样本越多，对比信号越丰富
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='学习率')
    '''
    对 ViT/大模型微调来说：
    1e-5 是非常常见、比较“温和”的正则强度。如果你完全从零训练，可能会用到 1e-4 或更大；
    这里是用大规模预训练模型（RETFound + MedSAM），1e-5 很合适。
    '''
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='权重衰减')
    '''
    先用 0.1；如果发现：
    loss 很难降、梯度爆炸或训练不稳定，可以试稍微大一点：0.15 ~ 0.2。
    loss 能降，但对对齐效果不满意，可以试 0.07 或 0.05，更强调区分度。
    '''
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='对比损失温度参数')
    parser.add_argument('--optimizer', type=str, default='adam',
                       choices=['adam', 'adamw', 'sgd'],
                       help='优化器类型')
    parser.add_argument('--scheduler', type=str, default='cosine',
                       choices=['cosine', 'step', 'plateau', 'none'],
                       help='学习率调度器类型')
    '''
    当前代码中还没有真正用到（create_scheduler 里没有处理 warmup）
    '''
    parser.add_argument('--warmup_epochs', type=int, default=5,
                       help='预热轮数（用于cosine调度器）')
    parser.add_argument('--freeze_encoders', action='store_true',
                       help='冻结编码器参数（只训练投影头）')
    parser.add_argument('--training_mode', type=str, default='grade',
                       choices=['subject', 'grade', 'mixed', 'mixed_clinical'],
                       help='训练模式选择')
    
    # 数据加载参数
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载器工作进程数')
    parser.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                       help='启用 DataLoader 的固定内存')
    parser.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                       help='禁用 DataLoader 的固定内存')
    parser.set_defaults(pin_memory=True)
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
    # 梯度累积
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                       help='梯度累积步数（>1 时，相当于放大等效 batch size）')
    # memory bank / 队列负样本
    parser.add_argument('--use_queue', action='store_true',
                       help='是否启用 memory bank / 队列负样本扩展')
    parser.add_argument('--queue_size', type=int, default=16384,
                       help='memory bank 队列长度（负样本容量 K）')
    
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
        mri_base_path=args.mri_base_path,
        subject_column=args.subject_column,
        retinal_column=args.retinal_column,
        mri_paths_column=args.mri_paths_column,
        label_column=args.label_column,
        grade_column=args.grade_column,
        t1map_npy_column=args.t1map_npy_column,
        short_axis_npy_column=args.short_axis_npy_column
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
    
    # 创建损失函数（正样本定义与 training_mode 绑定）
    criterion = CrossModalContrastiveLoss(
        tau=args.temperature,
        training_mode=args.training_mode
    )
    
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
        save_interval=args.save_interval,
        training_mode=args.training_mode,
        use_amp=True,
        grad_accum_steps=args.grad_accum_steps,
        use_queue=args.use_queue,
        queue_size=args.queue_size,
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

