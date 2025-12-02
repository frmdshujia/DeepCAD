"""
投影头模块
将编码器输出映射到共享潜在空间，并进行L2归一化
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """
    投影头 f_{φ}
    
    将编码器嵌入映射到共享潜在空间，输出L2归一化
    z = f_φ(h)，其中z在单位超球面上（L2归一化）
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: int = 128,  # 默认共享潜在空间维度
        num_layers: int = 2,
        use_bn: bool = True,
        dropout: float = 0.0
    ):
        """
        初始化投影头
        
        Args:
            input_dim: 输入维度（编码器输出的嵌入维度）
                      对于视网膜: 1024 (RETFound ViT-Large)
                      对于MRI: 768 (MedSAM ViT-Base 投影后)
            hidden_dim: 隐藏层维度（如果为None，则使用input_dim）
            output_dim: 输出维度（共享潜在空间的维度）
                       默认值: 128
                       设计参考: MMCL-Tabular-Imaging 项目中的常见设置
                       说明: 这是两个模态投影后的统一维度，用于计算跨模态相似度
                             较小的维度有助于减少计算量和提高泛化能力
                             常见可调整值: 64, 128, 256, 512
            num_layers: MLP层数（至少为2）
            use_bn: 是否使用批归一化
            dropout: Dropout比率
        """
        super(ProjectionHead, self).__init__()
        
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")
        
        if hidden_dim is None:
            hidden_dim = input_dim
        
        layers = []
        
        # 第一层：input_dim -> hidden_dim
        layers.append(nn.Linear(input_dim, hidden_dim))
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        
        # 中间层（如果有）
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
        
        # 最后一层：hidden_dim -> output_dim（不添加激活函数，因为后面会L2归一化）
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入嵌入，形状为 (batch_size, input_dim)
        
        Returns:
            投影后的嵌入，形状为 (batch_size, output_dim)，已L2归一化
        """
        # 通过MLP
        z = self.mlp(x)  # (batch_size, output_dim)
        
        # L2归一化到单位超球面
        z = F.normalize(z, p=2, dim=1)
        
        return z
    
    def get_output_dim(self) -> int:
        """返回输出维度"""
        return self.output_dim


class SimpleProjectionHead(nn.Module):
    """
    简单的2层投影头（用于快速实验或基线）
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 128,  # 默认共享潜在空间维度，与 ProjectionHead 相同
        use_bn: bool = True
    ):
        """
        Args:
            input_dim: 输入维度
            output_dim: 输出维度
            use_bn: 是否使用批归一化
        """
        super(SimpleProjectionHead, self).__init__()
        
        layers = [
            nn.Linear(input_dim, output_dim),
        ]
        
        if use_bn:
            layers.append(nn.BatchNorm1d(output_dim))
        
        layers.append(nn.ReLU())
        layers.append(nn.Linear(output_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        self.output_dim = output_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入嵌入，形状为 (batch_size, input_dim)
        
        Returns:
            投影后的嵌入，形状为 (batch_size, output_dim)，已L2归一化
        """
        z = self.mlp(x)
        z = F.normalize(z, p=2, dim=1)
        return z
    
    def get_output_dim(self) -> int:
        """返回输出维度"""
        return self.output_dim

