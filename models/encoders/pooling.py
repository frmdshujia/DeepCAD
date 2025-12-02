"""
可训练池化层
用于将切片级特征聚合为主体级嵌入
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionPooling(nn.Module):
    """
    注意力池化层
    使用可学习的注意力权重对切片级特征进行加权聚合
    """
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        """
        Args:
            embed_dim: 特征维度
                      对于 MRI 编码器: 768 (MedSAM ViT-Base 投影后的维度)
            num_heads: 注意力头数（默认8）
                      设计选择: 8 是常见的多头注意力头数
                      要求: embed_dim 必须能被 num_heads 整除
                      对于 embed_dim=768: 768 / 8 = 96 (每个头的维度)
            dropout: Dropout比率（默认0.1）
        """
        super(AttentionPooling, self).__init__()
        
        self.embed_dim = embed_dim  # 768 (对于 MRI 编码器)
        self.num_heads = num_heads  # 8 (默认)
        self.head_dim = embed_dim // num_heads  # 768 / 8 = 96
        
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        # 可学习的查询向量
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # 线性投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # 输出投影
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        对切片级特征进行注意力池化
        
        Args:
            slice_features: 切片级特征，形状为 (batch_size, num_slices, embed_dim)
        
        Returns:
            主体级嵌入，形状为 (batch_size, embed_dim)
        """
        batch_size, num_slices, embed_dim = slice_features.shape
        
        # 扩展查询向量到批次大小
        query = self.query.expand(batch_size, -1, -1)  # (B, 1, D)
        
        # 投影
        Q = self.q_proj(query)  # (B, 1, D)
        K = self.k_proj(slice_features)  # (B, num_slices, D)
        V = self.v_proj(slice_features)  # (B, num_slices, D)
        
        # 重塑为多头形式
        Q = Q.view(batch_size, 1, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, 1, D_h)
        K = K.view(batch_size, num_slices, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, num_slices, D_h)
        V = V.view(batch_size, num_slices, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, num_slices, D_h)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # (B, H, 1, num_slices)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力权重
        out = torch.matmul(attn_weights, V)  # (B, H, 1, D_h)
        out = out.transpose(1, 2).contiguous().view(batch_size, 1, embed_dim)  # (B, 1, D)
        out = self.out_proj(out)  # (B, 1, D)
        
        # 残差连接和层归一化
        out = self.norm(out.squeeze(1))  # (B, D)
        
        return out


class LearnableWeightedPooling(nn.Module):
    """
    可学习加权池化层
    使用可学习的权重对切片进行加权平均
    """
    
    def __init__(self, embed_dim: int):
        """
        Args:
            embed_dim: 特征维度
        """
        super(LearnableWeightedPooling, self).__init__()
        
        # 可学习的权重网络
        self.weight_net = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        对切片级特征进行加权池化
        
        Args:
            slice_features: 切片级特征，形状为 (batch_size, num_slices, embed_dim)
        
        Returns:
            主体级嵌入，形状为 (batch_size, embed_dim)
        """
        # 计算每个切片的权重
        weights = self.weight_net(slice_features)  # (B, num_slices, 1)
        weights = F.softmax(weights, dim=1)  # 归一化权重
        
        # 加权求和
        out = torch.sum(weights * slice_features, dim=1)  # (B, embed_dim)
        out = self.norm(out)
        
        return out


class MeanPooling(nn.Module):
    """
    简单的平均池化层（用于对比或基线）
    """
    
    def __init__(self, embed_dim: int):
        super(MeanPooling, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_features: 切片级特征，形状为 (batch_size, num_slices, embed_dim)
        
        Returns:
            主体级嵌入，形状为 (batch_size, embed_dim)
        """
        out = torch.mean(slice_features, dim=1)  # (B, embed_dim)
        out = self.norm(out)
        return out


class MaxPooling(nn.Module):
    """
    最大池化层（用于对比或基线）
    """
    
    def __init__(self, embed_dim: int):
        super(MaxPooling, self).__init__()
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, slice_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            slice_features: 切片级特征，形状为 (batch_size, num_slices, embed_dim)
        
        Returns:
            主体级嵌入，形状为 (batch_size, embed_dim)
        """
        out, _ = torch.max(slice_features, dim=1)  # (B, embed_dim)
        out = self.norm(out)
        return out

