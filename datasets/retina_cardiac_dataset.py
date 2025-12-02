"""
视网膜+心脏MRI数据集
用于DeepCAD Stage I训练
"""

import os
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
        max_mri_slices: int = 10,
        mri_slice_type: str = "representative",
        live_loading: bool = True,
        retinal_base_path: Optional[str] = None,
        mri_base_path: Optional[str] = None
    ):
        """
        初始化数据集
        
        Args:
            data_csv: CSV文件路径，包含以下列：
                     - subject_id: 受试者ID
                     - retinal_path: 视网膜图像路径（相对于retinal_base_path或绝对路径）
                     - mri_paths: MRI路径（可以是JSON字符串、逗号分隔的路径，或单个路径）
                     - label: CAD标签 (0或1)
            retinal_img_size: 视网膜图像输出尺寸
            mri_img_size: MRI图像输出尺寸
            train: 是否为训练模式（影响数据增强）
            retinal_augmentation: 视网膜图像增强强度 ("light", "medium", "strong")
            mri_augmentation: MRI图像增强强度
            max_mri_slices: 最大MRI切片数量
            mri_slice_type: MRI切片类型 ("mid", "representative", "all")
            live_loading: 是否实时加载（True: 从磁盘加载，False: 预加载到内存）
            retinal_base_path: 视网膜图像的基础路径（如果CSV中的路径是相对路径）
            mri_base_path: MRI数据的基础路径（如果CSV中的路径是相对路径）
        """
        super(RetinaCardiacDataset, self).__init__()
        
        # 加载数据索引
        if not os.path.exists(data_csv):
            raise FileNotFoundError(f"Data CSV not found: {data_csv}")
        
        self.df = pd.read_csv(data_csv)
        required_columns = ['subject_id', 'retinal_path', 'mri_paths', 'label']
        missing_columns = [col for col in required_columns if col not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
        
        self.train = train
        self.live_loading = live_loading
        self.retinal_base_path = retinal_base_path
        self.mri_base_path = mri_base_path
        self.max_mri_slices = max_mri_slices
        self.mri_slice_type = mri_slice_type
        self.retinal_img_size = retinal_img_size
        self.mri_img_size = mri_img_size
        self.mri_augmentation = mri_augmentation
        
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
            retinal_path = self._get_retinal_path(idx)
            mri_paths = self._get_mri_paths(idx)
            
            # 加载视网膜图像
            retinal_img = load_retinal_image(retinal_path)
            self.retinal_data.append(retinal_img)
            
            # 加载MRI切片
            mri_slices = load_mri_slices(
                mri_paths,
                max_slices=self.max_mri_slices
            )
            self.mri_data.append(mri_slices)
        
        print(f"Preloaded {len(self.retinal_data)} samples")
    
    def _get_retinal_path(self, idx: int) -> str:
        """获取视网膜图像完整路径"""
        path = self.df.iloc[idx]['retinal_path']
        if os.path.isabs(path):
            return path
        elif self.retinal_base_path:
            return os.path.join(self.retinal_base_path, path)
        else:
            return path
    
    def _get_mri_paths(self, idx: int) -> List[str]:
        """获取MRI路径列表"""
        mri_paths_str = str(self.df.iloc[idx]['mri_paths'])
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
        subject_id = str(self.df.iloc[idx]['subject_id'])
        label = int(self.df.iloc[idx]['label'])
        
        # 加载视网膜图像
        if self.live_loading:
            retinal_path = self._get_retinal_path(idx)
            retinal_img = load_retinal_image(retinal_path)
        else:
            retinal_img = self.retinal_data[idx]
        
        # 应用视网膜图像变换
        x_R = self.retinal_transform(retinal_img)
        
        # 加载MRI切片
        if self.live_loading:
            mri_paths = self._get_mri_paths(idx)
            mri_slices = load_mri_slices(
                mri_paths,
                max_slices=self.max_mri_slices
            )
        else:
            mri_slices = self.mri_data[idx]
        
        # 对每个MRI切片应用变换
        # mri_slices形状: (num_slices, H, W)
        transformed_slices = []
        for i in range(mri_slices.shape[0]):
            slice_img = mri_slices[i].numpy()
            # 转换为PIL Image（需要先归一化到0-255范围）
            slice_min, slice_max = slice_img.min(), slice_img.max()
            if slice_max > slice_min:
                slice_img = ((slice_img - slice_min) / (slice_max - slice_min) * 255).astype(np.uint8)
            else:
                slice_img = np.zeros_like(slice_img, dtype=np.uint8)
            
            slice_pil = Image.fromarray(slice_img, mode='L')
            
            # 应用变换（注意：mri_transform期望PIL Image或Tensor）
            # 我们需要创建一个自定义的变换来处理这种情况
            transformed_slice = self._apply_mri_transform(slice_pil)
            transformed_slices.append(transformed_slice)
        
        # 堆叠所有切片: (num_slices, 1, H, W) 或 (num_slices, H, W)
        x_C = torch.stack(transformed_slices)
        
        # 如果切片没有通道维度，添加一个
        if x_C.dim() == 3:
            x_C = x_C.unsqueeze(1)  # (num_slices, 1, H, W)
        
        return {
            'x_R': x_R,
            'x_C': x_C,
            'y': torch.tensor(label, dtype=torch.long),
            'subject_id': subject_id
        }
    
    def _apply_mri_transform(self, slice_pil: Image.Image) -> torch.Tensor:
        """
        对单个MRI切片应用变换
        
        Args:
            slice_pil: PIL Image (灰度)
        
        Returns:
            torch.Tensor: 变换后的切片
        """
        # 将PIL Image转换为numpy数组，然后转换为Tensor
        import numpy as np
        slice_array = np.array(slice_pil, dtype=np.float32)
        slice_tensor = torch.from_numpy(slice_array)
        
        # 应用变换（变换管道中的Lambda函数会处理Tensor）
        # 但我们需要手动处理Resize等操作
        from torchvision import transforms as T
        
        # 创建临时变换管道（不包含Lambda）
        if self.train:
            if self.mri_augmentation == "light":
                transform = T.Compose([
                    T.Resize(size=(self.mri_img_size + 16, self.mri_img_size + 16)),
                    T.RandomCrop(size=(self.mri_img_size, self.mri_img_size)),
                    T.RandomHorizontalFlip(p=0.5),
                ])
            elif self.mri_augmentation == "medium":
                transform = T.Compose([
                    T.Resize(size=(self.mri_img_size + 16, self.mri_img_size + 16)),
                    T.RandomCrop(size=(self.mri_img_size, self.mri_img_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=10),
                ])
            else:  # strong
                transform = T.Compose([
                    T.Resize(size=(self.mri_img_size + 32, self.mri_img_size + 32)),
                    T.RandomCrop(size=(self.mri_img_size, self.mri_img_size)),
                    T.RandomHorizontalFlip(p=0.5),
                    T.RandomRotation(degrees=15),
                ])
        else:
            transform = T.Compose([
                T.Resize(size=(self.mri_img_size, self.mri_img_size)),
            ])
        
        # 应用变换
        slice_pil_transformed = transform(slice_pil)
        slice_array = np.array(slice_pil_transformed, dtype=np.float32)
        slice_tensor = torch.from_numpy(slice_array)
        
        # 归一化
        slice_tensor = (slice_tensor - slice_tensor.mean()) / (slice_tensor.std() + 1e-8)
        
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

