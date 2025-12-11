#!/usr/bin/env python3
"""
模型调试脚本
用于诊断和验证模型配置
"""

import argparse
import os
import sys
import torch

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.deepcad_stage1 import DeepCADStageI
from datasets import create_dataloaders
from losses import CrossModalContrastiveLoss
from utils.debug import (
    diagnose_training_issue,
    print_diagnosis,
    validate_model_architecture,
    check_tensor_shapes
)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='DeepCAD 模型调试工具')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='模型检查点路径（可选）')
    parser.add_argument('--train_csv', type=str, default=None,
                       help='训练集CSV路径（用于诊断）')
    parser.add_argument('--retinal_pretrained', type=str, default=None,
                       help='RETFound预训练权重路径')
    parser.add_argument('--mri_pretrained', type=str, default=None,
                       help='MedSAM预训练权重路径')
    parser.add_argument('--device', type=str, default=None,
                       help='设备（cuda/cpu，默认自动选择）')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='测试批次大小')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    # 设置设备
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("DeepCAD 模型调试工具")
    print("=" * 60)
    print(f"设备: {device}\n")
    
    # 创建模型
    print("1. 创建模型...")
    try:
        model = DeepCADStageI(
            retinal_pretrained_path=args.retinal_pretrained,
            mri_pretrained_path=args.mri_pretrained,
            latent_dim=128
        )
        print("  ✓ 模型创建成功")
    except Exception as e:
        print(f"  ❌ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 加载检查点（如果有）
    if args.checkpoint:
        print(f"\n2. 加载检查点: {args.checkpoint}")
        try:
            from utils.checkpoint import load_checkpoint
            checkpoint = load_checkpoint(args.checkpoint)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("  ✓ 检查点加载成功")
        except Exception as e:
            print(f"  ❌ 检查点加载失败: {e}")
            import traceback
            traceback.print_exc()
            return
    
    # 验证模型架构
    print("\n3. 验证模型架构...")
    arch_check = validate_model_architecture(model)
    all_ok = True
    for key, value in arch_check.items():
        if isinstance(value, bool):
            status = "✓" if value else "❌"
            print(f"  {status} {key}: {value}")
            if not value:
                all_ok = False
        else:
            print(f"  ℹ️  {key}: {value}")
    
    if not all_ok:
        print("\n  ⚠️  模型架构验证发现问题")
    
    # 测试前向传播
    print("\n4. 测试前向传播...")
    try:
        batch_size = args.batch_size
        x_R = torch.randn(batch_size, 3, 224, 224)
        x_C = torch.randn(batch_size, 5, 1, 224, 224)  # 5个MRI切片
        
        model.eval()
        with torch.no_grad():
            outputs = model(x_R.to(device), x_C.to(device))
        
        print(f"  ✓ 前向传播成功")
        print(f"    h_R 形状: {outputs['h_R'].shape}")
        print(f"    h_C 形状: {outputs['h_C'].shape}")
        print(f"    z_R 形状: {outputs['z_R'].shape}")
        print(f"    z_C 形状: {outputs['z_C'].shape}")
        
        # 检查张量形状
        shapes = check_tensor_shapes(model, x_R.to(device), x_C.to(device))
        if 'error' not in shapes:
            print(f"    ✓ 所有张量形状正确")
        else:
            print(f"    ❌ 张量形状错误: {shapes['error']}")
    
    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 诊断训练问题（如果有数据）
    if args.train_csv:
        print("\n5. 诊断训练配置...")
        try:
            train_loader, _, _ = create_dataloaders(
                train_csv=args.train_csv,
                batch_size=args.batch_size,
                num_workers=0,  # 调试时使用单线程
                retinal_img_size=224,
                mri_img_size=224
            )
            
            criterion = CrossModalContrastiveLoss(tau=0.1)
            
            diagnosis = diagnose_training_issue(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                device=device
            )
            
            print_diagnosis(diagnosis)
            
        except Exception as e:
            print(f"  ❌ 诊断失败: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("调试完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

