"""
视网膜+心脏MRI数据集
用于DeepCAD Stage I训练
"""

import os
import ast
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, Optional, Union, List
from PIL import Image

from .transforms import get_retinal_transforms, get_mri_transforms
from .data_utils import (
    load_retinal_image,
    load_mri_slices,
    load_ukb_cardiac_slices,
    load_ukb_cardiac_with_t1,
    parse_mri_paths
)


class RetinaCardiacDataset(Dataset):
    """
    视网膜眼底照片 + 心脏MRI数据集
    
    对于每个受试者j：
    - 视网膜图像: x_j^R
    - 心脏MRI数据: x_j^C (代表性short-axis cine + Native T1切片)
    - 二分类CAD标签: y_j ∈ {0, 1}
    
    返回格式：
    {
        'x_R': retinal_tensor,      # 形状: (3, H, W) 或 (C, H, W)
        'x_C': mri_tensor,          # 形状: (num_slices, 1, H, W) 或 (num_slices, H, W)
        'y': cad_label,             # 形状: () 标量
        'subject_id': subject_id    # 字符串
    }
    """
    
    def __init__(
        self,
        data_csv: str,
        retinal_img_size: int = 224,
        mri_img_size: int = 224,
        train: bool = True,
        retinal_augmentation: str = "medium",
        mri_augmentation: str = "medium",
        max_mri_slices: int = 6,
        # 每个受试者最多使用多少张眼底图像（>1 时会从多张中均匀采样；1 张会复制）
        max_retinal_images: int = 2,
        mri_slice_type: str = "representative",
        live_loading: bool = True,
        retinal_base_path: Optional[str] = None,
        mri_base_path: Optional[str] = None,
        use_ukb_format: bool = False,
        include_t1: bool = False,
        t1_cycle_positions: Optional[List[str]] = None,
        # 列名配置（避免表头写死）
        subject_column: str = "subject_id",
        retinal_column: str = "retinal_path",
        mri_paths_column: str = "mri_paths",
        label_column: str = "label",
        grade_column: Optional[str] = None,
        # 分离的 MRI 序列列名（例如 T1MAP 与 short-axis ED/mid/ES 的 npy）
        t1map_npy_column: Optional[str] = None,
        short_axis_npy_column: Optional[str] = None,
    ):
        """
        初始化数据集
        
        Args:
            data_csv: CSV文件路径，包含以下列：
                     - subject_id: 受试者ID
                     - retinal_path: 视网膜图像路径（相对于retinal_base_path或绝对路径）
                     - mri_paths: MRI路径（可以是JSON字符串、逗号分隔的路径，或单个路径）
                                - 如果 use_ukb_format=True，这应该是受试者文件夹路径
                     - label: CAD标签 (0或1)
            retinal_img_size: 视网膜图像输出尺寸
            mri_img_size: MRI图像输出尺寸
            train: 是否为训练模式（影响数据增强）
            retinal_augmentation: 视网膜图像增强强度 ("light", "medium", "strong")
            mri_augmentation: MRI图像增强强度
            max_mri_slices: 最大MRI切片数量（用于非UKB格式）
            mri_slice_type: MRI切片类型 ("mid", "representative", "all")
            live_loading: 是否实时加载（True: 从磁盘加载，False: 预加载到内存）
            retinal_base_path: 视网膜图像的基础路径（如果CSV中的路径是相对路径）
            mri_base_path: MRI数据的基础路径（如果CSV中的路径是相对路径）
            use_ukb_format: 是否使用UKB格式（参考MMCL）
                          - True: 使用 load_ukb_cardiac_with_t1 加载
                          - False: 使用 load_mri_slices 加载
            include_t1: 是否包含T1 Map切片（仅当 use_ukb_format=True 时有效）
                       - True: 加载6张切片（3张Cine + 3张T1 Map）
                       - False: 加载3张切片（仅Cine）
            t1_cycle_positions: T1 Map的周期位置列表（仅当 include_t1=True 时有效）
                              - 默认: ['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz']
        """
        super(RetinaCardiacDataset, self).__init__()
        
        # 保存列名配置
        self.subject_column = subject_column
        self.retinal_column = retinal_column
        self.mri_paths_column = mri_paths_column
        self.label_column = label_column
        self.grade_column = grade_column
        self.t1map_npy_column = t1map_npy_column
        self.short_axis_npy_column = short_axis_npy_column

        # 加载数据索引（列名通过超参数配置）
        if not os.path.exists(data_csv):
            raise FileNotFoundError(f"Data CSV not found: {data_csv}")
        
        self.df = pd.read_csv(data_csv)
        # 基本必需列（label 列可选，以支持纯 subject-level 对比学习）
        required_columns = [
            self.subject_column,
            self.retinal_column,
        ]
        # MRI 列：如果显式提供了 npy 列名，则检查它们；
        # 否则回退到统一的 mri_paths_column
        if (self.t1map_npy_column is not None) or (self.short_axis_npy_column is not None):
            if self.t1map_npy_column is not None:
                required_columns.append(self.t1map_npy_column)
            if self.short_axis_npy_column is not None:
                required_columns.append(self.short_axis_npy_column)
        else:
            required_columns.append(self.mri_paths_column)
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        # 是否存在显式的标签列
        self.has_label = self.label_column in self.df.columns
        
        self.train = train
        self.live_loading = live_loading
        self.retinal_base_path = retinal_base_path
        self.mri_base_path = mri_base_path
        self.max_mri_slices = max_mri_slices
        self.max_retinal_images = max(1, int(max_retinal_images))
        self.mri_slice_type = mri_slice_type
        self.retinal_img_size = retinal_img_size
        self.mri_img_size = mri_img_size
        self.mri_augmentation = mri_augmentation
        self.use_ukb_format = use_ukb_format
        self.include_t1 = include_t1
        self.t1_cycle_positions = t1_cycle_positions
        
        # 如果使用UKB格式且包含T1，更新max_mri_slices
        if use_ukb_format and include_t1:
            self.expected_mri_slices = 6  # Cine (3) + T1 Map (3)
        elif use_ukb_format:
            self.expected_mri_slices = 3  # Cine only
        else:
            self.expected_mri_slices = max_mri_slices
        
        # 设置变换
        self.retinal_transform = get_retinal_transforms(
            img_size=retinal_img_size,
            train=train,
            augmentation_strength=retinal_augmentation
        )
        self.mri_transform = get_mri_transforms(
            img_size=mri_img_size,
            train=train,
            augmentation_strength=mri_augmentation
        )
        
        # 预加载数据（如果不需要实时加载）
        if not live_loading:
            self._preload_data()
        else:
            self.retinal_data = None
            self.mri_data = None
    
    def _preload_data(self):
        """预加载所有数据到内存"""
        print("Preloading data to memory...")
        self.retinal_data = []
        self.mri_data = []
        
        for idx in range(len(self.df)):
            # 预加载阶段仍按单张视网膜图像处理：如果存在多张，可在后续根据需要扩展
            retinal_path = self._get_retinal_path(idx)
            mri_paths = self._get_mri_paths(idx)
            
            # 加载视网膜图像
            retinal_img = load_retinal_image(retinal_path)
            self.retinal_data.append(retinal_img)
            
            # 加载MRI切片（根据格式选择不同的加载方式）
            if self.use_ukb_format:
                # 使用UKB格式（参考MMCL）
                subject_folder = mri_paths[0] if isinstance(mri_paths, list) else mri_paths
                if self.mri_base_path:
                    subject_folder = os.path.join(self.mri_base_path, subject_folder)
                
                try:
                    mri_slices = load_ukb_cardiac_with_t1(
                        subject_folder=subject_folder,
                        include_t1=self.include_t1,
                        t1_cycle_positions=self.t1_cycle_positions,
                        fallback_to_cine=False
                    )
                except (FileNotFoundError, ValueError) as e:
                    if self.include_t1:
                        print(f"Warning: Failed to load T1 Map for {subject_folder}: {e}")
                        print("Falling back to Cine slices only")
                        mri_slices = load_ukb_cardiac_slices(subject_folder)
                    else:
                        raise
            else:
                # 使用通用格式
                mri_slices = load_mri_slices(
                    mri_paths,
                    max_slices=self.max_mri_slices
                )
            self.mri_data.append(mri_slices)
        
        print(f"Preloaded {len(self.retinal_data)} samples")
    
    def _get_retinal_path(self, idx: int) -> str:
        """获取视网膜图像完整路径"""
        # 为了兼容旧逻辑，这里只返回第一张视网膜图像路径
        paths = self._get_retinal_paths(idx)
        path = paths[0]
        if os.path.isabs(path):
            return path
        elif self.retinal_base_path:
            return os.path.join(self.retinal_base_path, path)
        else:
            return path

    def _get_retinal_paths(self, idx: int) -> List[str]:
        """获取视网膜图像路径列表（支持列表/JSON/逗号分隔），并应用 max_retinal_images 规则"""
        path_value = self.df.iloc[idx][self.retinal_column]
        raw = str(path_value)

        # 优先尝试用 ast.literal_eval 解析类似 "['a.png', 'b.png']" 这种 Python 列表字符串
        paths: List[str] = []
        raw_strip = raw.strip()
        if raw_strip.startswith("[") and raw_strip.endswith("]"):
            try:
                parsed = ast.literal_eval(raw_strip)
                if isinstance(parsed, (list, tuple)):
                    paths = [str(p).strip().strip('"').strip("'") for p in parsed]
            except (ValueError, SyntaxError):
                # 解析失败则退回通用解析逻辑
                paths = []

        # 如果不是列表字符串，或上面解析失败，则退回到通用 parse_mri_paths
        if not paths:
            paths = parse_mri_paths(raw)
        # 过滤空字符串
        paths = [p for p in paths if isinstance(p, str) and p.strip() != ""]
        if len(paths) == 0:
            raise ValueError(f"No retinal paths found at index {idx} for column {self.retinal_column}")

        # 1) 如果多于 max_retinal_images，做均匀采样
        if len(paths) > self.max_retinal_images:
            indices = np.linspace(0, len(paths) - 1, self.max_retinal_images, dtype=int)
            paths = [paths[i] for i in indices]
        # 2) 如果只有 1 张且 max_retinal_images > 1，则复制
        elif len(paths) == 1 and self.max_retinal_images > 1:
            paths = paths * self.max_retinal_images
        
        # 处理相对路径
        if self.retinal_base_path:
            paths = [
                os.path.join(self.retinal_base_path, p) if not os.path.isabs(p) else p
                for p in paths
            ]
        
        return paths
    
    def _get_mri_paths(self, idx: int) -> List[str]:
        """获取MRI路径列表"""
        mri_paths_str = str(self.df.iloc[idx][self.mri_paths_column])
        paths = parse_mri_paths(mri_paths_str)
        
        # 处理相对路径
        if self.mri_base_path:
            paths = [os.path.join(self.mri_base_path, p) if not os.path.isabs(p) else p 
                    for p in paths]
        
        return paths
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        获取一个样本
        
        Returns:
            Dict包含:
                - 'x_R': 视网膜图像张量 (C, H, W)
                - 'x_C': MRI切片张量 (num_slices, 1, H, W) 或 (num_slices, H, W)
                - 'y': CAD标签 (标量)
                - 'subject_id': 受试者ID (字符串)
        """
        # 获取受试者ID和标签
        subject_id = str(self.df.iloc[idx][self.subject_column])
        if self.has_label:
            label = int(self.df.iloc[idx][self.label_column])
        else:
            # 无标签列时，使用占位标签 0（主要用于 subject-level 对比学习）
            label = 0
        grade_value = None
        if self.grade_column is not None and self.grade_column in self.df.columns:
            grade_value = self.df.iloc[idx][self.grade_column]
        
        # 加载视网膜图像（支持多张眼底同时输入）
        if self.live_loading:
            # 支持单张或多张视网膜图像
            retinal_paths = self._get_retinal_paths(idx)
            retinal_images = []
            for p in retinal_paths:
                try:
                    retinal_images.append(load_retinal_image(p))
                except FileNotFoundError:
                    continue
            if len(retinal_images) == 0:
                raise FileNotFoundError(f"No valid retinal images found for index {idx}: {retinal_paths}")
        else:
            # 预加载模式当前只缓存一张眼底，这里将其作为单一视图使用
            retinal_images = [self.retinal_data[idx]]

        # 对每张眼底图像应用相同的变换，并堆叠成 (num_retinal, C, H, W)
        retinal_tensors = []
        for img in retinal_images:
            retinal_tensors.append(self.retinal_transform(img))  # (3, H, W)
        x_R = torch.stack(retinal_tensors, dim=0)  # (num_retinal, 3, H, W)
        
        # 加载 MRI 切片
        raw_mri_slices: List[np.ndarray] = []
        if self.live_loading:
            if self.use_ukb_format:
                # 使用UKB格式（参考MMCL）：加载Cine + T1 Map
                mri_paths = self._get_mri_paths(idx)
                subject_folder = mri_paths[0] if isinstance(mri_paths, list) else mri_paths
                if self.mri_base_path:
                    subject_folder = os.path.join(self.mri_base_path, subject_folder)
                try:
                    mri_slices = load_ukb_cardiac_with_t1(
                        subject_folder=subject_folder,
                        include_t1=self.include_t1,
                        t1_cycle_positions=self.t1_cycle_positions,
                        fallback_to_cine=False  # 严格模式，T1 Map必须存在
                    )  # (6, H, W) 或 (3, H, W)
                except (FileNotFoundError, ValueError) as e:
                    if self.include_t1:
                        print(f"Warning: Failed to load T1 Map for {subject_folder}: {e}")
                        print("Falling back to Cine slices only")
                        mri_slices = load_ukb_cardiac_slices(subject_folder)  # (3, H, W)
                    else:
                        raise
                # 转为 numpy list，保持与 npy 分支一致
                for i in range(mri_slices.shape[0]):
                    raw_mri_slices.append(mri_slices[i].numpy())
            elif (self.t1map_npy_column is not None and self.t1map_npy_column in self.df.columns) or \
                 (self.short_axis_npy_column is not None and self.short_axis_npy_column in self.df.columns):
                # 使用分离的 npy 序列列（例如 T1MAP_B1B2B3_npy 与 short_axis_ED_mid_ES_npy）
                def _load_npy_as_slices(npy_path: str) -> List[np.ndarray]:
                    slices = []
                    if not isinstance(npy_path, str) or npy_path.strip() == "":
                        return slices
                    full_path = npy_path
                    if self.mri_base_path and not os.path.isabs(full_path):
                        full_path = os.path.join(self.mri_base_path, full_path)
                    if not os.path.exists(full_path):
                        return slices
                    arr = np.load(full_path)
                    # 期望形状为 (num_slices, H, W) 或 (H, W)
                    if arr.ndim == 2:
                        arr = arr[None, ...]
                    elif arr.ndim != 3:
                        raise ValueError(f"Unexpected npy MRI shape: {arr.shape} at {full_path}")
                    # 归一化到 [0, 1]（如果需要）
                    a_min, a_max = arr.min(), arr.max()
                    if a_max > a_min:
                        if a_max > 1.0 or a_min < 0.0:
                            arr = (arr - a_min) / (a_max - a_min)
                    else:
                        arr = np.zeros_like(arr)
                    for i in range(arr.shape[0]):
                        slices.append(arr[i])
                    return slices

                # 短轴序列
                if self.short_axis_npy_column is not None and self.short_axis_npy_column in self.df.columns:
                    sa_path = str(self.df.iloc[idx][self.short_axis_npy_column])
                    raw_mri_slices.extend(_load_npy_as_slices(sa_path))
                # T1MAP 序列
                if self.t1map_npy_column is not None and self.t1map_npy_column in self.df.columns:
                    t1_path = str(self.df.iloc[idx][self.t1map_npy_column])
                    raw_mri_slices.extend(_load_npy_as_slices(t1_path))

                if len(raw_mri_slices) == 0:
                    raise ValueError(f"No MRI slices loaded from npy columns at index {idx}")
            else:
                # 使用通用格式：从 mri_paths_column 加载
                mri_paths = self._get_mri_paths(idx)
                mri_slices = load_mri_slices(
                    mri_paths,
                    max_slices=self.max_mri_slices
                )
                for i in range(mri_slices.shape[0]):
                    raw_mri_slices.append(mri_slices[i].numpy())
        else:
            # 预加载模式：mri_data 已经是 Tensor (num_slices, H, W)
            mri_slices = self.mri_data[idx]
            for i in range(mri_slices.shape[0]):
                raw_mri_slices.append(mri_slices[i].numpy())
        
        # 对每个 MRI 切片应用变换（方案A：每张切片单独处理）
        # 
        # 注意：MRI 有多个切片，需要循环处理每个切片，然后堆叠
        # 这与视网膜图像不同（视网膜是单个图像，可以直接应用 transform）
        # 
        # 数据流程说明（符合 MedSAM 输入要求，方案A）：
        # 1. load_ukb_cardiac_with_t1() 或 load_mri_slices() 返回 [0, 1] 范围的 Tensor
        #    - UKB格式（包含T1）: (6, H, W) - [Cine_ES, Cine_mid, Cine_ED, T1_ES, T1_mid, T1_ED]
        #    - UKB格式（仅Cine）: (3, H, W) - [Cine_ES, Cine_mid, Cine_ED]
        #    - 通用格式: (num_slices, H, W)
        # 2. 转换为 PIL Image（需要 0-255 范围，因为 torchvision transforms 期望 PIL Image）
        # 3. 应用 transform（ToTensor() 会转换回 [0, 1] 范围）
        # 4. 堆叠所有切片: (num_slices, 1, H, W)
        # 5. MedSAM encoder 会：
        #    - 每张切片独立处理
        #    - 将单通道复制为 3 通道（x.repeat(1, 3, 1, 1)）
        #    - 上采样到 1024x1024（MedSAM 的标准输入尺寸）
        #    - 每张切片通过 MedSAM encoder，然后池化融合
        #    - 输入已经是 [0, 1] 范围，符合 MedSAM 要求
        # 
        # raw_mri_slices 中每个元素形状约为 (H, W)，值范围约为 [0, 1]
        transformed_slices = []
        for slice_img in raw_mri_slices:
            # 转换为PIL Image（需要先归一化到0-255范围）
            # 注意：load_mri_slices 已经返回 [0, 1] 范围，所以直接乘以 255
            # 但如果数据是其他范围（如 NIfTI 原始值），需要先归一化
            slice_min, slice_max = slice_img.min(), slice_img.max()
            if slice_max > slice_min:
                # 如果数据已经是 [0, 1] 范围，直接乘以 255
                # 否则先归一化到 [0, 1] 再乘以 255
                if slice_max <= 1.0:
                    slice_img = (slice_img * 255).astype(np.uint8)
                else:
                    slice_img = ((slice_img - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
            else:
                slice_img = np.zeros_like(slice_img, dtype=np.uint8)
            
            slice_pil = Image.fromarray(slice_img, mode='L')
            
            # 应用变换（使用统一的 self.mri_transform）
            # transform 会输出 [0, 1] 范围的 Tensor，符合 MedSAM 要求
            transformed_slice = self._apply_mri_transform(slice_pil)
            transformed_slices.append(transformed_slice)
        
        # 堆叠所有切片: (num_slices, 1, H, W) 或 (num_slices, H, W)
        x_C = torch.stack(transformed_slices)
        
        # 如果切片没有通道维度，添加一个
        if x_C.dim() == 3:
            x_C = x_C.unsqueeze(1)  # (num_slices, 1, H, W)
        
        sample = {
            'x_R': x_R,
            'x_C': x_C,
            'y': torch.tensor(label, dtype=torch.long),
            'subject_id': subject_id
        }

        # 如果存在 grade 列，则一并返回（供其他训练模式使用）
        if grade_value is not None and not (isinstance(grade_value, float) and np.isnan(grade_value)):
            try:
                grade_tensor = torch.tensor(float(grade_value), dtype=torch.float32)
                sample['grade'] = grade_tensor
            except Exception:
                # 若无法转为 float，则跳过该字段
                pass

        return sample
    
    def _apply_mri_transform(self, slice_pil: Image.Image) -> torch.Tensor:
        """
        对单个MRI切片应用变换
        
        直接使用 self.mri_transform，该 transform 已经配置好所有增强策略
        
        注意：
        - 输入：PIL Image（灰度，mode='L'），值范围 [0, 255]
        - 输出：Tensor，形状 (1, H, W)，值范围 [0, 1]
        - 符合 MedSAM 的输入要求（MedSAM encoder 会处理通道转换和上采样）
        
        Args:
            slice_pil: PIL Image (灰度图像，mode='L')
        
        Returns:
            torch.Tensor: 变换后的切片，形状为 (1, H, W)，值范围 [0, 1]
        """
        # 直接使用预配置的 transform
        # transform 会处理：空间变换 -> ToTensor -> float 转换
        slice_tensor = self.mri_transform(slice_pil)
        
        # 确保输出形状正确：对于灰度图像，ToTensor 会输出 (1, H, W)
        # 如果已经是 (1, H, W)，则直接返回；如果是 (H, W)，则添加通道维度
        if slice_tensor.dim() == 2:
            slice_tensor = slice_tensor.unsqueeze(0)
        
        return slice_tensor


def create_dataloaders(
    train_csv: str,
    val_csv: Optional[str] = None,
    test_csv: Optional[str] = None,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle_train: bool = True,
    **dataset_kwargs
):
    """
    创建训练、验证和测试数据加载器
    
    Args:
        train_csv: 训练集CSV路径
        val_csv: 验证集CSV路径（可选）
        test_csv: 测试集CSV路径（可选）
        batch_size: 批次大小
        num_workers: 数据加载器工作进程数
        pin_memory: 是否固定内存
        shuffle_train: 是否打乱训练集
        **dataset_kwargs: 传递给RetinaCardiacDataset的其他参数
    
    Returns:
        Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]: 
        训练、验证、测试数据加载器
    """
    from torch.utils.data import DataLoader
    
    # 训练集
    train_dataset = RetinaCardiacDataset(
        data_csv=train_csv,
        train=True,
        **dataset_kwargs
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True  # 确保批次大小一致
    )
    
    # 验证集
    val_loader = None
    if val_csv:
        val_dataset = RetinaCardiacDataset(
            data_csv=val_csv,
            train=False,
            **dataset_kwargs
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    # 测试集
    test_loader = None
    if test_csv:
        test_dataset = RetinaCardiacDataset(
            data_csv=test_csv,
            train=False,
            **dataset_kwargs
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return train_loader, val_loader, test_loader

