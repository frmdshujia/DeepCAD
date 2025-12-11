"""
数据加载工具函数
用于加载视网膜图像和MRI切片

MRI 数据加载参考 MMCL-Tabular-Imaging-main 实现
"""

import os
import json
import numpy as np
import torch
from typing import List, Union, Optional
from PIL import Image
import nibabel as nib
from torchvision.io import read_image


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
    加载心脏MRI切片（参考 MMCL-Tabular-Imaging-main 实现）
    
    MMCL 的处理方式：
    1. 如果路径是 .pt 文件，直接 torch.load() 加载预处理的 Tensor
    2. 如果是图像文件，使用 torchvision.io.read_image() 加载
    3. 如果是 NIfTI 文件，提取中间切片（参考 MMCL 的预处理方式）
    
    注意：返回的 Tensor 值范围在 [0, 1]，符合 MedSAM 的输入要求
    MedSAM 期望输入是 [0, 1] 范围的 3 通道图像（会在 encoder 内部处理）
    
    Args:
        mri_paths: MRI文件路径或路径列表
                  - 如果是字符串，可能是：
                    - .pt 文件（预处理的 Tensor，MMCL 方式）
                    - 单个 NIfTI 文件
                    - 包含切片的目录
                  - 如果是列表，是多个切片文件的路径
        slice_indices: 要加载的切片索引（仅用于 NIfTI，如果为None，则提取中间切片）
                      - 参考 MMCL：提取中间 z 轴切片 im[:,:,im.shape[2]//2]
        max_slices: 最大切片数量
    
    Returns:
        torch.Tensor: 形状为 (num_slices, H, W) 的MRI切片张量，值范围 [0, 1]
    """
    if isinstance(mri_paths, str):
        # 单个路径
        if os.path.isdir(mri_paths):
            # 目录：加载目录中的所有图像文件
            slice_files = sorted([f for f in os.listdir(mri_paths) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.pt', '.nii', '.nii.gz'))])
            mri_paths = [os.path.join(mri_paths, f) for f in slice_files]
        else:
            # 单个文件
            mri_paths = [mri_paths]
    
    slices = []
    
    for path in mri_paths:
        if not os.path.exists(path):
            continue
        
        path_lower = path.lower()
        
        # 1. 预处理的 .pt 文件（MMCL 的主要方式）
        # 优势：
        # - 加载速度快（避免每次训练重新处理 NIfTI）
        # - 数据格式统一（所有样本预处理为相同格式）
        # - 内存效率高（可按需加载）
        # - 便于版本控制和复现
        if path_lower.endswith('.pt'):
            try:
                # 直接加载预处理的 Tensor（MMCL 方式）
                data = torch.load(path, map_location='cpu')
                if isinstance(data, torch.Tensor):
                    # 如果是单个 Tensor，检查形状
                    if data.dim() == 3:
                        # MMCL 中通常是 (3, H, W)，其中 3 个通道代表：
                        # - Channel 0: ES (end-systolic) 切片
                        # - Channel 1: mid beat 切片
                        # - Channel 2: ED (end-diastolic) 切片
                        # 这是 3 张不同的切片，不是单通道复制！
                        if data.shape[0] > 1:
                            # 多通道：提取每个通道作为独立的切片
                            # 这样可以在后续处理中分别应用增强
                            for c in range(data.shape[0]):
                                slices.append(data[c].numpy())
                        else:
                            slices.append(data[0].numpy())
                    elif data.dim() == 2:
                        slices.append(data.numpy())
                    else:
                        raise ValueError(f"Unsupported Tensor shape: {data.shape}")
                elif isinstance(data, dict):
                    # 如果是字典（MMCL 的 all_subjects 格式），需要指定键
                    raise ValueError("Dictionary format not supported. Please provide a specific key or use a list of paths.")
                else:
                    raise ValueError(f"Unsupported data type in .pt file: {type(data)}")
            except Exception as e:
                raise ValueError(f"Failed to load .pt file {path}: {e}")
        
        # 2. NIfTI 格式（参考 MMCL 的预处理方式）
        elif path_lower.endswith(('.nii', '.nii.gz')):
            nii = nib.load(path)
            data = nii.get_fdata()
            
            # 如果是3D数据，提取中间切片
            # 注意：MMCL 每个样本使用 3 张切片（ES, mid, ED），不是 1 张！
            # 这里只提取中间切片，如果需要 MMCL 的完整方式，请使用 load_ukb_cardiac_slices()
            if data.ndim == 3:
                if slice_indices is not None:
                    for idx in slice_indices:
                        if 0 <= idx < data.shape[2]:
                            slices.append(data[:, :, idx])
                else:
                    # 提取中间 z 轴切片（MMCL 方式：mid_heart_slice = im[:,:,im.shape[2]//2]）
                    # MMCL 会从 3 个不同的 NIfTI 文件中各提取一张中间切片，堆叠成 (3, H, W)
                    mid_slice = data.shape[2] // 2
                    slices.append(data[:, :, mid_slice])
            elif data.ndim == 2:
                slices.append(data)
            else:
                raise ValueError(f"Unsupported NIfTI data shape: {data.shape}")
        
        # 3. 图像格式（使用 torchvision.io.read_image，MMCL 方式）
        else:
            try:
                # 使用 torchvision.io.read_image（MMCL 方式）
                img_tensor = read_image(path)  # 返回 (C, H, W)，值范围 [0, 255]
                # 归一化到 [0, 1]（MMCL 方式：im = im / 255）
                img_tensor = img_tensor.float() / 255.0
                
                # 如果是多通道，提取每个通道作为切片；如果是单通道，直接使用
                if img_tensor.shape[0] > 1:
                    for c in range(img_tensor.shape[0]):
                        slices.append(img_tensor[c].numpy())
                else:
                    slices.append(img_tensor[0].numpy())
            except Exception as e:
                # 如果 read_image 失败，回退到 PIL（向后兼容）
                try:
                    img = Image.open(path).convert('L')
                    img_array = np.array(img, dtype=np.float32) / 255.0
                    slices.append(img_array)
                except Exception as e2:
                    raise ValueError(f"Failed to load image {path}: {e}, {e2}")
    
    if len(slices) == 0:
        raise ValueError(f"No valid MRI slices found in paths: {mri_paths}")
    
    # 限制切片数量（均匀采样）
    if len(slices) > max_slices:
        indices = np.linspace(0, len(slices) - 1, max_slices, dtype=int)
        slices = [slices[i] for i in indices]
    
    # 转换为torch.Tensor（已经是 float 类型，范围 [0, 1]）
    slices_tensor = torch.stack([torch.from_numpy(s).float() for s in slices])
    
    return slices_tensor


def load_ukb_cardiac_slices(
    subject_folder: str,
    cycle_positions: List[str] = None
) -> torch.Tensor:
    """
    从 UKB 心脏 MRI 数据加载关键切片（参考 MMCL 的实现方式）
    
    MMCL 的处理方式：
    - 提取三个关键时间点的切片：ES (end-systolic), mid beat, ED (end-diastolic)
    - 从每个 NIfTI 文件中提取中间 z 轴切片：im[:,:,im.shape[2]//2]
    - 对于完整周期 (sa.nii.gz)，提取中间心跳切片
    - Padding 成正方形
    - 堆叠成 (3, H, W) 的 Tensor
    
    Args:
        subject_folder: 受试者文件夹路径，应包含：
                       - sa_ES.nii.gz (end-systolic)
                       - sa.nii.gz (完整周期，用于提取 mid beat)
                       - sa_ED.nii.gz (end-diastolic)
        cycle_positions: 要加载的周期位置列表，默认 ['sa_ES.nii.gz', 'sa.nii.gz', 'sa_ED.nii.gz']
    
    Returns:
        torch.Tensor: 形状为 (3, H, W) 的 Tensor，值范围 [0, 1]
    """
    if cycle_positions is None:
        cycle_positions = ['sa_ES.nii.gz', 'sa.nii.gz', 'sa_ED.nii.gz']
    
    import numpy as np
    
    def get_mid_beat_slice(im, es_slice):
        """提取中间心跳切片（参考 MMCL 的实现）"""
        thresh = (1.0, 99.0)
        best_overlap_es = 0
        best_i_es = None
        
        # 假设第4维是时间维度
        if im.ndim != 4:
            return None
            
        for i in range(min(50, im.shape[3])):
            im_slice = im[:, :, im.shape[2]//2, i]
            overlap_es = (es_slice == im_slice).sum()
            if overlap_es > best_overlap_es:
                best_overlap_es = overlap_es
                best_i_es = i
        
        if best_i_es is None:
            return None
        
        val_l, val_h = np.percentile(im, thresh)
        im_slice = im[:, :, im.shape[2]//2, best_i_es]
        im_slice = np.clip(im_slice, val_l, val_h)
        
        try:
            assert np.allclose(im_slice, es_slice, rtol=1e-3)
        except:
            return None
        
        mid_beat_i = best_i_es // 2
        mid_beat_slice = im[:, :, im.shape[2]//2, mid_beat_i]
        mid_beat_slice = np.clip(mid_beat_slice, val_l, val_h)
        return mid_beat_slice
    
    slices = []
    es_slice = None
    
    for cycle_pos in cycle_positions:
        path = os.path.join(subject_folder, cycle_pos)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing file: {path}")
        
        nii = nib.load(path)
        im = nii.get_fdata()
        
        # 检查 z 轴切片数量（MMCL 要求至少 7 个）
        if im.ndim >= 3 and im.shape[2] <= 7:
            raise ValueError(f"Too few z-axis slices: {im.shape[2]}")
        
        # 提取中间 z 轴切片
        if cycle_pos == 'sa.nii.gz' and im.ndim == 4:
            # 完整周期：提取中间心跳切片
            mid_heart_slice = get_mid_beat_slice(im, es_slice)
            if mid_heart_slice is None:
                raise ValueError(f"Failed to extract mid beat slice from {path}")
        else:
            # ES 或 ED：提取中间 z 轴切片
            mid_heart_slice = im[:, :, im.shape[2]//2]
        
        # 保存 ES slice 用于后续匹配
        if cycle_pos == 'sa_ES.nii.gz':
            es_slice = mid_heart_slice.copy()
        
        # Padding 成正方形（MMCL 方式）
        if mid_heart_slice.shape[1] > mid_heart_slice.shape[0]:
            pad_size = (mid_heart_slice.shape[1] - mid_heart_slice.shape[0]) // 2
            mid_heart_slice = np.pad(
                mid_heart_slice,
                ((pad_size, pad_size), (0, 0)),
                'constant',
                constant_values=0
            )
        elif mid_heart_slice.shape[0] > mid_heart_slice.shape[1]:
            pad_size = (mid_heart_slice.shape[0] - mid_heart_slice.shape[1]) // 2
            mid_heart_slice = np.pad(
                mid_heart_slice,
                ((0, 0), (pad_size, pad_size)),
                'constant',
                constant_values=0
            )
        
        assert mid_heart_slice.shape[0] == mid_heart_slice.shape[1], \
            f"Shape mismatch after padding: {mid_heart_slice.shape}"
        
        # 归一化到 [0, 1] 范围（符合 MedSAM 要求）
        slice_min, slice_max = mid_heart_slice.min(), mid_heart_slice.max()
        if slice_max > slice_min:
            mid_heart_slice = (mid_heart_slice - slice_min) / (slice_max - slice_min)
        else:
            mid_heart_slice = np.zeros_like(mid_heart_slice)
        
        slices.append(torch.from_numpy(mid_heart_slice).float())
    
    # 堆叠成 (3, H, W)
    stacked = torch.stack(slices)
    
    # 特殊处理：如果形状是 (3, 208, 208)，padding 到 (3, 210, 210)（MMCL 方式）
    if stacked.shape == (3, 208, 208):
        stacked = torch.nn.functional.pad(stacked, (1, 1, 1, 1), mode='constant', value=0)
    
    return stacked


def load_ukb_cardiac_with_t1(
    subject_folder: str,
    include_t1: bool = True,
    t1_cycle_positions: List[str] = None,
    fallback_to_cine: bool = False
) -> torch.Tensor:
    """
    加载 UKB 心脏 MRI 数据（Cine + T1 Map）- 方案A优化版本
    
    每个样本包含 6 张切片（如果包含T1 Map）：
    - Cine 序列（3 张）：ES, mid beat, ED（参考 MMCL）
    - T1 Map 序列（3 张）：对应的时间点
    
    处理策略（方案A）：
    - 每张切片单独处理（单通道）
    - 堆叠成 (6, H, W) 或 (3, H, W)
    - MedSAM encoder 会在内部将每张切片复制为 3 通道
    - 每张切片独立通过 MedSAM，然后池化融合
    
    Args:
        subject_folder: 受试者文件夹路径，应包含：
                       - sa_ES.nii.gz, sa.nii.gz, sa_ED.nii.gz（Cine序列）
                       - t1_ES.nii.gz, t1_mid.nii.gz, t1_ED.nii.gz（T1 Map，可选）
        include_t1: 是否包含 T1 Map 切片
        t1_cycle_positions: T1 Map 的周期位置列表
                          - 默认: ['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz']
                          - 如果只有一个文件: ['t1.nii.gz']
        fallback_to_cine: 如果T1 Map文件不存在，是否使用Cine切片作为占位符
                         False: 抛出错误
                         True: 使用对应的Cine切片
    
    Returns:
        torch.Tensor: 
            - 如果 include_t1=True 且成功加载: (6, H, W)
            - 如果 include_t1=False 或 T1 Map 不可用: (3, H, W)
            - 值范围: [0, 1]，符合 MedSAM 输入要求
    
    Raises:
        FileNotFoundError: 如果必需的Cine文件不存在
        ValueError: 如果 include_t1=True 但T1 Map文件不存在且 fallback_to_cine=False
    """
    # 加载 Cine 切片（参考 MMCL）
    cine_slices = load_ukb_cardiac_slices(subject_folder)  # (3, H, W)
    
    if not include_t1:
        return cine_slices  # (3, H, W)
    
    # 加载 T1 Map 切片
    if t1_cycle_positions is None:
        # 尝试常见的 T1 Map 文件命名
        t1_cycle_positions = ['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz']
    
    t1_slices = []
    missing_files = []
    
    for i, t1_pos in enumerate(t1_cycle_positions):
        t1_path = os.path.join(subject_folder, t1_pos)
        
        if os.path.exists(t1_path):
            try:
                # 加载 T1 Map 切片（提取中间 z 轴切片）
                nii = nib.load(t1_path)
                im = nii.get_fdata()
                
                if im.ndim == 3:
                    # 提取中间 z 轴切片（参考 MMCL）
                    mid_slice = im[:, :, im.shape[2]//2]
                elif im.ndim == 2:
                    mid_slice = im
                else:
                    raise ValueError(f"Unsupported T1 Map shape: {im.shape}")
                
                # Padding 成正方形（参考 MMCL）
                if mid_slice.shape[1] > mid_slice.shape[0]:
                    pad_size = (mid_slice.shape[1] - mid_slice.shape[0]) // 2
                    remainder = (mid_slice.shape[1] - mid_slice.shape[0]) % 2
                    mid_slice = np.pad(
                        mid_slice,
                        ((pad_size, pad_size + remainder), (0, 0)),
                        'constant',
                        constant_values=0
                    )
                elif mid_slice.shape[0] > mid_slice.shape[1]:
                    pad_size = (mid_slice.shape[0] - mid_slice.shape[1]) // 2
                    remainder = (mid_slice.shape[0] - mid_slice.shape[1]) % 2
                    mid_slice = np.pad(
                        mid_slice,
                        ((0, 0), (pad_size, pad_size + remainder)),
                        'constant',
                        constant_values=0
                    )
                
                # 确保是正方形
                assert mid_slice.shape[0] == mid_slice.shape[1], \
                    f"T1 Map slice shape mismatch after padding: {mid_slice.shape}"
                
                # 归一化到 [0, 1]（符合 MedSAM 要求）
                slice_min, slice_max = mid_slice.min(), mid_slice.max()
                if slice_max > slice_min:
                    mid_slice = (mid_slice - slice_min) / (slice_max - slice_min)
                else:
                    mid_slice = np.zeros_like(mid_slice)
                
                t1_slices.append(torch.from_numpy(mid_slice).float())
                
            except Exception as e:
                if fallback_to_cine:
                    print(f"Warning: Failed to load T1 Map {t1_path}: {e}, using Cine slice as fallback")
                    t1_slices.append(cine_slices[i % 3].clone())
                else:
                    raise ValueError(f"Failed to load T1 Map {t1_path}: {e}")
        else:
            missing_files.append(t1_path)
            if fallback_to_cine:
                print(f"Warning: T1 Map file not found: {t1_path}, using Cine slice as fallback")
                t1_slices.append(cine_slices[i % 3].clone())
            else:
                raise FileNotFoundError(f"T1 Map file not found: {t1_path}")
    
    if len(t1_slices) == 0:
        if fallback_to_cine:
            print("Warning: No T1 Map slices loaded, returning Cine slices only")
            return cine_slices  # (3, H, W)
        else:
            raise ValueError(f"No T1 Map slices could be loaded from {subject_folder}")
    
    # 堆叠 T1 Map 切片
    t1_tensor = torch.stack(t1_slices)  # (3, H, W)
    
    # 确保 Cine 和 T1 Map 的空间尺寸一致
    if cine_slices.shape[1:] != t1_tensor.shape[1:]:
        # 如果尺寸不一致，resize T1 Map 到 Cine 的尺寸
        from torch.nn import functional as F
        target_size = cine_slices.shape[1:]
        t1_tensor = F.interpolate(
            t1_tensor.unsqueeze(1),  # (3, 1, H, W)
            size=target_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(1)  # (3, H, W)
    
    # 堆叠 Cine 和 T1 Map：顺序为 [Cine_ES, Cine_mid, Cine_ED, T1_ES, T1_mid, T1_ED]
    all_slices = torch.cat([cine_slices, t1_tensor], dim=0)  # (6, H, W)
    
    return all_slices


def load_mri_from_nifti(
    nifti_path: str,
    slice_type: str = "mid",
    num_slices: int = 3
) -> torch.Tensor:
    """
    从NIfTI文件加载代表性MRI切片（参考 MMCL-Tabular-Imaging-main 实现）
    
    MMCL 的预处理方式：提取中间心跳切片 (mid_heart_slice = im[:,:,im.shape[2]//2])
    
    Args:
        nifti_path: NIfTI文件路径
        slice_type: 切片类型 ("mid", "representative", "all")
                    - "mid": 提取中间切片（MMCL 方式）
                    - "representative": 均匀采样多个代表性切片
                    - "all": 返回所有切片
        num_slices: 要提取的切片数量（用于 "representative" 模式）
    
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

