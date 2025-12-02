#!/usr/bin/env python3
"""
数据准备示例脚本
帮助您创建符合格式要求的 CSV 文件
"""

import argparse
import os
import pandas as pd
from pathlib import Path


def create_example_csv(output_path: str, num_samples: int = 10):
    """
    创建示例 CSV 文件
    
    Args:
        output_path: 输出 CSV 路径
        num_samples: 示例样本数量
    """
    data = []
    
    for i in range(1, num_samples + 1):
        subject_id = f"subj_{i:03d}"
        
        # 假设的数据路径（需要根据实际情况修改）
        retinal_path = f"retinal/{subject_id}.jpg"
        mri_path = f"mri/{subject_id}_cine.nii.gz"
        
        # 随机标签（实际应该使用真实标签）
        import random
        label = random.randint(0, 1)
        
        data.append({
            'subject_id': subject_id,
            'retinal_path': retinal_path,
            'mri_paths': mri_path,
            'label': label
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"示例 CSV 已创建: {output_path}")
    print(f"包含 {len(df)} 个样本")
    print("\n前5行:")
    print(df.head())
    print("\n请根据实际数据路径修改 CSV 文件！")


def validate_csv(csv_path: str):
    """
    验证 CSV 文件格式
    
    Args:
        csv_path: CSV 文件路径
    """
    print(f"验证 CSV 文件: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # 检查必需的列
        required_columns = ['subject_id', 'retinal_path', 'mri_paths', 'label']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"❌ 缺少必需的列: {missing_columns}")
            return False
        
        print(f"✓ CSV 格式正确")
        print(f"  样本数量: {len(df)}")
        print(f"  标签分布:")
        print(df['label'].value_counts().to_string())
        
        # 检查路径（如果提供了基础路径）
        print(f"\n前5个样本:")
        print(df.head().to_string())
        
        return True
        
    except Exception as e:
        print(f"❌ CSV 验证失败: {e}")
        return False


def split_data(csv_path: str, train_ratio: float = 0.7, val_ratio: float = 0.15):
    """
    划分数据集
    
    Args:
        csv_path: 原始 CSV 路径
        train_ratio: 训练集比例
        val_ratio: 验证集比例
    """
    df = pd.read_csv(csv_path)
    
    # 随机打乱
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # 计算划分点
    n_total = len(df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    
    # 划分
    train_df = df[:n_train]
    val_df = df[n_train:n_train + n_val]
    test_df = df[n_train + n_val:]
    
    # 保存
    base_path = os.path.splitext(csv_path)[0]
    train_path = f"{base_path}_train.csv"
    val_path = f"{base_path}_val.csv"
    test_path = f"{base_path}_test.csv"
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"数据集划分完成:")
    print(f"  训练集: {train_path} ({len(train_df)} 样本)")
    print(f"  验证集: {val_path} ({len(val_df)} 样本)")
    print(f"  测试集: {test_path} ({len(test_df)} 样本)")


def main():
    parser = argparse.ArgumentParser(description='数据准备工具')
    parser.add_argument('--create_example', type=str, default=None,
                       help='创建示例 CSV 文件路径')
    parser.add_argument('--validate', type=str, default=None,
                       help='验证 CSV 文件路径')
    parser.add_argument('--split', type=str, default=None,
                       help='划分数据集 CSV 路径')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='示例样本数量')
    
    args = parser.parse_args()
    
    if args.create_example:
        create_example_csv(args.create_example, args.num_samples)
    
    if args.validate:
        validate_csv(args.validate)
    
    if args.split:
        split_data(args.split)


if __name__ == "__main__":
    main()

