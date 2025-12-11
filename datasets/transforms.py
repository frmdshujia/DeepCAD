"""
数据变换和增强模块
为视网膜图像和心脏MRI分别定义变换管道

视网膜数据增强参考 RETFound 实现
MRI 数据增强参考 MMCL-Tabular-Imaging-main 实现
"""

import torch
from torchvision import transforms
from typing import Optional, Callable
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def get_retinal_transforms(
    img_size: int = 224,
    train: bool = True,
    augmentation_strength: str = "medium"
) -> transforms.Compose:
    """
    获取视网膜图像的变换管道（参考 RETFound 实现）
    
    Args:
        img_size: 输出图像尺寸
        train: 是否为训练模式（训练时应用增强，验证时不应用）
        augmentation_strength: 增强强度 ("light", "medium", "strong")
    
    Returns:
        transforms.Compose: 变换管道
    """
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    
    if not train:
        # 验证/测试时参考 RETFound 的实现
        crop_pct = 224 / 256 if img_size <= 224 else 1.0
        size = int(img_size / crop_pct)
        return transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    
    # 训练时的增强策略（参考 RETFound，使用 timm 的 create_transform）
    # 根据增强强度调整参数
    if augmentation_strength == "light":
        # 轻度增强：较小的 color_jitter，较弱的 auto_augment，较低的 random erase 概率
        return create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.2,
            auto_augment="rand-m5-mstd0.5-inc1",  # 较弱的 RandAugment
            interpolation='bicubic',
            re_prob=0.1,  # Random Erasing 概率较低
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
    elif augmentation_strength == "medium":
        # 中等增强：RETFound 默认配置
        return create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.4,
            auto_augment="rand-m9-mstd0.5-inc1",  # RETFound 默认的 RandAugment
            interpolation='bicubic',
            re_prob=0.25,  # RETFound 默认的 Random Erasing 概率
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )
    else:  # strong
        # 强度增强：更强的 color_jitter，更强的 auto_augment，更高的 random erase 概率
        return create_transform(
            input_size=img_size,
            is_training=True,
            color_jitter=0.6,
            auto_augment="rand-m12-mstd0.5-inc1",  # 更强的 RandAugment
            interpolation='bicubic',
            re_prob=0.4,  # 更高的 Random Erasing 概率
            re_mode='pixel',
            re_count=1,
            mean=mean,
            std=std,
        )


def get_mri_transforms(
    img_size: int = 224,
    train: bool = True,
    augmentation_strength: str = "medium"
) -> transforms.Compose:
    """
    获取心脏MRI图像的变换管道（参考 MMCL-Tabular-Imaging-main 实现）
    
    注意：
    - 此函数期望 PIL Image 输入（灰度图像，mode='L'）
    - 输出 Tensor 值范围在 [0, 1]，符合 MedSAM 的输入要求
    - MedSAM encoder 会在内部将单通道复制为 3 通道，并上采样到 1024x1024
    
    Args:
        img_size: 输出图像尺寸（MedSAM 会在 encoder 内部上采样到 1024x1024）
        train: 是否为训练模式
        augmentation_strength: 增强强度 ("light", "medium", "strong")
    
    Returns:
        transforms.Compose: 变换管道，输出形状为 (1, H, W) 的 Tensor，值范围 [0, 1]
    """
    if not train:
        # 验证/测试时只进行标准化变换
        return transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),  # PIL Image -> Tensor (1, H, W) for grayscale
            transforms.Lambda(lambda x: x.float()),  # 确保是 float 类型
        ])
    
    # 训练时的增强策略：以“医学友好”的几何变换为主，避免颜色抖动
    if augmentation_strength == "light":
        # 轻度增强：小角度旋转 + 适度裁剪，不改变对比度/亮度
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.9, 1.0)),
            transforms.ToTensor(),  # PIL Image -> Tensor (1, H, W) for grayscale
            transforms.Lambda(lambda x: x.float()),  # 确保是 float 类型
        ])
    elif augmentation_strength == "medium":
        # 中等增强：稍大的旋转和裁剪范围，仍不改变强度分布
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=25),
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
            transforms.ToTensor(),  # PIL Image -> Tensor (1, H, W) for grayscale
            transforms.Lambda(lambda x: x.float()),  # 确保是 float 类型
        ])
    else:  # strong
        # 强增强：增加旋转幅度和裁剪范围，可选加入轻微模糊
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=35),
            transforms.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0)),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            transforms.ToTensor(),  # PIL Image -> Tensor (1, H, W) for grayscale
            transforms.Lambda(lambda x: x.float()),  # 确保是 float 类型
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

