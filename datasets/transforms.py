"""
数据变换和增强模块
为视网膜图像和心脏MRI分别定义变换管道
"""

import torch
from torchvision import transforms
from typing import Optional, Callable


def get_retinal_transforms(
    img_size: int = 224,
    train: bool = True,
    augmentation_strength: str = "medium"
) -> transforms.Compose:
    """
    获取视网膜图像的变换管道
    
    Args:
        img_size: 输出图像尺寸
        train: 是否为训练模式（训练时应用增强，验证时不应用）
        augmentation_strength: 增强强度 ("light", "medium", "strong")
    
    Returns:
        transforms.Compose: 变换管道
    """
    if not train:
        # 验证/测试时只进行标准化变换
        return transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # 训练时的增强策略
    if augmentation_strength == "light":
        return transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),
            transforms.RandomCrop(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif augmentation_strength == "medium":
        return transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),
            transforms.RandomCrop(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:  # strong
        return transforms.Compose([
            transforms.Resize(size=(img_size + 48, img_size + 48)),
            transforms.RandomCrop(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))], p=0.5),
            transforms.RandomApply([transforms.ElasticTransform(alpha=50.0, sigma=5.0)], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def get_mri_transforms(
    img_size: int = 224,
    train: bool = True,
    augmentation_strength: str = "medium"
) -> transforms.Compose:
    """
    获取心脏MRI图像的变换管道
    
    Args:
        img_size: 输出图像尺寸
        train: 是否为训练模式
        augmentation_strength: 增强强度
    
    Returns:
        transforms.Compose: 变换管道
    """
    if not train:
        # 验证/测试时只进行标准化变换
        return transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.Lambda(lambda x: x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),  # 添加通道维度如果是2D
            transforms.Normalize(mean=[0.5], std=[0.5])  # 简单的归一化
        ])
    
    # 训练时的增强策略（MRI增强通常比视网膜图像更保守）
    if augmentation_strength == "light":
        return transforms.Compose([
            transforms.Resize(size=(img_size + 16, img_size + 16)),
            transforms.RandomCrop(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Lambda(lambda x: x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    elif augmentation_strength == "medium":
        return transforms.Compose([
            transforms.Resize(size=(img_size + 16, img_size + 16)),
            transforms.RandomCrop(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.Lambda(lambda x: x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    else:  # strong
        return transforms.Compose([
            transforms.Resize(size=(img_size + 32, img_size + 32)),
            transforms.RandomCrop(size=(img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            transforms.Lambda(lambda x: x.float() if isinstance(x, torch.Tensor) else torch.tensor(x, dtype=torch.float32)),
            transforms.Lambda(lambda x: x.unsqueeze(0) if x.dim() == 2 else x),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


class ToTensor2D:
    """将2D numpy数组或PIL图像转换为torch.Tensor"""
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            return pic
        import numpy as np
        from PIL import Image
        
        if isinstance(pic, np.ndarray):
            return torch.from_numpy(pic).float()
        elif isinstance(pic, Image.Image):
            return transforms.functional.to_tensor(pic)
        else:
            raise TypeError(f"Type {type(pic)} not supported")

