"""
数据加载工具函数
用于加载视网膜图像和MRI切片
"""

import os
import json
import numpy as np
import torch
from typing import List, Union, Optional
from PIL import Image
import nibabel as nib


def load_retinal_image(image_path: str) -> Image.Image:
    """
    加载视网膜眼底照片
    
    Args:
        image_path: 图像文件路径（支持JPEG, PNG等格式）
    
    Returns:
        PIL.Image: 加载的图像
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Retinal image not found: {image_path}")
    
    image = Image.open(image_path).convert('RGB')
    return image


def load_mri_slices(
    mri_paths: Union[str, List[str]],
    slice_indices: Optional[List[int]] = None,
    max_slices: int = 10
) -> torch.Tensor:
    """
    加载心脏MRI切片
    
    Args:
        mri_paths: MRI文件路径或路径列表
                  - 如果是字符串，可能是单个NIfTI文件或包含切片的目录
                  - 如果是列表，是多个切片文件的路径
        slice_indices: 要加载的切片索引（如果为None，则加载所有或代表性切片）
        max_slices: 最大切片数量
    
    Returns:
        torch.Tensor: 形状为 (num_slices, H, W) 的MRI切片张量
    """
    if isinstance(mri_paths, str):
        # 单个路径：可能是NIfTI文件或目录
        if os.path.isdir(mri_paths):
            # 目录：加载目录中的所有图像文件
            slice_files = sorted([f for f in os.listdir(mri_paths) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.nii', '.nii.gz'))])
            mri_paths = [os.path.join(mri_paths, f) for f in slice_files]
        else:
            # 单个文件
            mri_paths = [mri_paths]
    
    slices = []
    
    for path in mri_paths:
        if not os.path.exists(path):
            continue
            
        # 根据文件扩展名选择加载方式
        if path.lower().endswith(('.nii', '.nii.gz')):
            # NIfTI格式
            nii = nib.load(path)
            data = nii.get_fdata()
            
            # 如果是3D数据，提取中间切片或指定切片
            if data.ndim == 3:
                if slice_indices is not None:
                    for idx in slice_indices:
                        if 0 <= idx < data.shape[2]:
                            slices.append(data[:, :, idx])
                else:
                    # 提取中间切片
                    mid_slice = data.shape[2] // 2
                    slices.append(data[:, :, mid_slice])
            elif data.ndim == 2:
                slices.append(data)
            else:
                raise ValueError(f"Unsupported NIfTI data shape: {data.shape}")
        else:
            # 图像格式（PNG, JPEG等）
            img = Image.open(path).convert('L')  # 转换为灰度
            img_array = np.array(img)
            slices.append(img_array)
    
    if len(slices) == 0:
        raise ValueError(f"No valid MRI slices found in paths: {mri_paths}")
    
    # 限制切片数量
    if len(slices) > max_slices:
        # 均匀采样
        indices = np.linspace(0, len(slices) - 1, max_slices, dtype=int)
        slices = [slices[i] for i in indices]
    
    # 转换为torch.Tensor
    slices_tensor = torch.stack([torch.from_numpy(s).float() for s in slices])
    
    return slices_tensor


def load_mri_from_nifti(
    nifti_path: str,
    slice_type: str = "mid",
    num_slices: int = 3
) -> torch.Tensor:
    """
    从NIfTI文件加载代表性MRI切片
    
    Args:
        nifti_path: NIfTI文件路径
        slice_type: 切片类型 ("mid", "representative", "all")
        num_slices: 要提取的切片数量
    
    Returns:
        torch.Tensor: 形状为 (num_slices, H, W) 的切片张量
    """
    if not os.path.exists(nifti_path):
        raise FileNotFoundError(f"NIfTI file not found: {nifti_path}")
    
    nii = nib.load(nifti_path)
    data = nii.get_fdata()
    
    if data.ndim == 2:
        return torch.from_numpy(data).float().unsqueeze(0)
    elif data.ndim == 3:
        if slice_type == "mid":
            # 提取中间切片
            mid_idx = data.shape[2] // 2
            return torch.from_numpy(data[:, :, mid_idx]).float().unsqueeze(0)
        elif slice_type == "representative":
            # 均匀采样代表性切片
            indices = np.linspace(0, data.shape[2] - 1, num_slices, dtype=int)
            slices = [data[:, :, idx] for idx in indices]
            return torch.stack([torch.from_numpy(s).float() for s in slices])
        else:  # all
            # 返回所有切片（可能很多）
            slices = [data[:, :, i] for i in range(data.shape[2])]
            return torch.stack([torch.from_numpy(s).float() for s in slices])
    else:
        raise ValueError(f"Unsupported NIfTI data dimensionality: {data.ndim}")


def parse_mri_paths(mri_path_str: str) -> List[str]:
    """
    解析MRI路径字符串（可能是JSON列表或逗号分隔的路径）
    
    Args:
        mri_path_str: 路径字符串
    
    Returns:
        List[str]: 路径列表
    """
    # 尝试解析为JSON
    try:
        paths = json.loads(mri_path_str)
        if isinstance(paths, list):
            return paths
    except:
        pass
    
    # 尝试按逗号分割
    if ',' in mri_path_str:
        return [p.strip() for p in mri_path_str.split(',')]
    
    # 单个路径
    return [mri_path_str]

