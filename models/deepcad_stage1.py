"""
DeepCAD Stage I 完整模型
整合视网膜编码器、MRI编码器和投影头
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .encoders import RetinalEncoder, MRICardioEncoder
from .projection_heads import ProjectionHead


class DeepCADStageI(nn.Module):
    """
    DeepCAD Stage I 完整模型
    
    包含：
    - 视网膜编码器 f_{θ_R}
    - 心脏MRI编码器 f_{θ_C}
    - 视网膜投影头 f_{φ_R}
    - 心脏MRI投影头 f_{φ_C}
    """
    
    def __init__(
        self,
        # 视网膜编码器参数
        retinal_encoder: Optional[RetinalEncoder] = None,
        retinal_pretrained_path: Optional[str] = None,
        retinal_img_size: int = 224,  # 默认图像尺寸
        retinal_freeze_backbone: bool = False,
        # MRI编码器参数
        mri_encoder: Optional[MRICardioEncoder] = None,
        mri_pretrained_path: Optional[str] = None,
        mri_img_size: int = 224,  # 默认图像尺寸（MedSAM内部会调整到1024）
        mri_pooling_type: str = "attention",
        mri_freeze_backbone: bool = False,
        # 投影头参数
        latent_dim: int = 128,  # 共享潜在空间维度，默认128
        projection_hidden_dim: Optional[int] = None,
        projection_num_layers: int = 2,
        projection_use_bn: bool = True,
        projection_dropout: float = 0.0
    ):
        """
        初始化 DeepCAD Stage I 模型
        
        Args:
            retinal_encoder: 预初始化的视网膜编码器（如果为None，则创建新的）
            retinal_pretrained_path: RETFound 预训练权重路径
            retinal_img_size: 视网膜图像尺寸
            retinal_freeze_backbone: 是否冻结视网膜编码器骨干网络
            mri_encoder: 预初始化的MRI编码器（如果为None，则创建新的）
            mri_pretrained_path: MedSAM 预训练权重路径
            mri_img_size: MRI图像尺寸
            mri_pooling_type: MRI池化类型
            mri_freeze_backbone: 是否冻结MRI编码器骨干网络
            latent_dim: 共享潜在空间维度
            projection_hidden_dim: 投影头隐藏层维度
            projection_num_layers: 投影头层数
            projection_use_bn: 投影头是否使用批归一化
            projection_dropout: 投影头Dropout比率
        """
        super(DeepCADStageI, self).__init__()
        
        # 视网膜编码器
        if retinal_encoder is None:
            self.retinal_encoder = RetinalEncoder(
                img_size=retinal_img_size,
                pretrained_path=retinal_pretrained_path,
                global_pool=True,
                freeze_backbone=retinal_freeze_backbone
            )
        else:
            self.retinal_encoder = retinal_encoder
        
        # 视网膜编码器输出维度: 1024
        # 来源: RETFound ViT-Large (参考 retinal_encoder.py 中的注释)
        # 参考: RETFound-main/models_vit.py 的 RETFound_mae() 函数
        retinal_embed_dim = self.retinal_encoder.get_embed_dim()  # 1024
        self.retinal_embed_dim = retinal_embed_dim
        
        # 心脏MRI编码器
        if mri_encoder is None:
            self.mri_encoder = MRICardioEncoder(
                img_size=mri_img_size,
                pretrained_path=mri_pretrained_path,
                pooling_type=mri_pooling_type,
                freeze_backbone=mri_freeze_backbone
            )
        else:
            self.mri_encoder = mri_encoder
        
        # MRI编码器输出维度: 768
        # 来源: MedSAM ViT-Base 投影后的维度 (参考 mri_encoder.py 中的注释)
        # 参考: MedSAM-main/segment_anything/build_sam.py 的 build_sam_vit_b()
        #       虽然 MedSAM 原始输出是 256 维，但我们投影到 768 维以匹配 ViT-Base 标准维度
        mri_embed_dim = self.mri_encoder.get_embed_dim()  # 768
        
        # 投影头配置
        # 隐藏层维度: 默认使用两个编码器输出维度的最大值
        # 设计选择: max(1024, 768) = 1024，确保有足够的容量进行特征变换
        if projection_hidden_dim is None:
            projection_hidden_dim = max(retinal_embed_dim, mri_embed_dim)  # 1024
        
        # 视网膜投影头: 1024 -> 1024 (hidden) -> 128 (output)
        # 输入维度: 1024 (RETFound ViT-Large 输出)
        # 输出维度: latent_dim (默认 128，共享潜在空间维度)
        self.projection_R = ProjectionHead(
            input_dim=retinal_embed_dim,  # 1024
            hidden_dim=projection_hidden_dim,  # 1024 (默认)
            output_dim=latent_dim,  # 128 (默认，可配置)
            num_layers=projection_num_layers,
            use_bn=projection_use_bn,
            dropout=projection_dropout
        )
        
        # MRI投影头: 768 -> 1024 (hidden) -> 128 (output)
        # 输入维度: 768 (MedSAM ViT-Base 投影后输出)
        # 输出维度: latent_dim (默认 128，与视网膜投影头相同，形成共享空间)
        self.projection_C = ProjectionHead(
            input_dim=mri_embed_dim,  # 768
            hidden_dim=projection_hidden_dim,  # 1024 (默认，与视网膜投影头相同)
            output_dim=latent_dim,  # 128 (默认，与视网膜投影头相同)
            num_layers=projection_num_layers,
            use_bn=projection_use_bn,
            dropout=projection_dropout
        )
        
        # 共享潜在空间维度: 128 (默认)
        # 设计选择: 参考 MMCL-Tabular-Imaging 项目中的常见设置
        # 说明: 这是两个模态投影后的统一维度，用于计算跨模态相似度
        #       较小的维度 (128) 有助于:
        #       1. 减少计算量
        #       2. 提高泛化能力
        #       3. 便于后续下游任务
        # 可调整: 可以通过 --latent_dim 参数修改 (常见值: 64, 128, 256, 512)
        self.latent_dim = latent_dim
    
    def forward(
        self,
        x_R: torch.Tensor,
        x_C: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_R: 视网膜图像，形状为 (batch_size, 3, H, W)
            x_C: 心脏MRI切片，形状为 (batch_size, num_slices, 1, H, W)
        
        Returns:
            Dict包含:
                - 'h_R': 视网膜嵌入 (batch_size, 1024)
                      维度来源: RETFound ViT-Large 的标准输出维度
                - 'h_C': MRI嵌入 (batch_size, 768)
                      维度来源: MedSAM ViT-Base 投影后的维度
                - 'z_R': 视网膜投影 (batch_size, latent_dim)，已L2归一化
                      维度来源: 投影头输出，默认 128
                - 'z_C': MRI投影 (batch_size, latent_dim)，已L2归一化
                      维度来源: 投影头输出，默认 128，与 z_R 相同以形成共享空间
        """
        # 编码阶段 - 视网膜
        # 支持两种输入形式：
        # 1) 单视图: x_R 形状为 (B, 3, H, W)
        # 2) 多视图: x_R 形状为 (B, num_retinal, 3, H, W)，例如同一受试者的多张眼底
        #    我们会对每张眼底独立编码，然后对特征在视图维度做平均池化，得到主体级嵌入
        if x_R.dim() == 4:
            # 单视图 (B, 3, H, W)
            h_R = self.retinal_encoder(x_R)  # (B, 1024)
        elif x_R.dim() == 5:
            # 多视图 (B, num_retinal, 3, H, W)
            B, V, C, H, W = x_R.shape
            x_R_flat = x_R.view(B * V, C, H, W)  # (B*V, 3, H, W)
            h_R_views = self.retinal_encoder(x_R_flat)  # (B*V, 1024)
            h_R_views = h_R_views.view(B, V, self.retinal_embed_dim)  # (B, V, 1024)
            # 简单平均池化得到主体级视网膜嵌入 (B, 1024)
            # 如需更复杂的池化（注意力等），可在此处替换
            h_R = h_R_views.mean(dim=1)
        else:
            raise ValueError(f"Unexpected x_R shape: {x_R.shape}")
        
        # MRI编码器输出: (B, 768)
        # 维度 768 是 MedSAM ViT-Base 投影后的维度 (参考 mri_encoder.py)
        h_C = self.mri_encoder(x_C)  # (B, 768)
        
        # 投影
        z_R = self.projection_R(h_R)  # (B, latent_dim)，已归一化
        z_C = self.projection_C(h_C)  # (B, latent_dim)，已归一化
        
        return {
            'h_R': h_R,
            'h_C': h_C,
            'z_R': z_R,
            'z_C': z_C
        }
    
    def get_latent_dim(self) -> int:
        """返回潜在空间维度"""
        return self.latent_dim
    
    def freeze_encoders(self):
        """冻结编码器参数（只训练投影头）"""
        for param in self.retinal_encoder.parameters():
            param.requires_grad = False
        for param in self.mri_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """解冻编码器参数"""
        for param in self.retinal_encoder.parameters():
            param.requires_grad = True
        for param in self.mri_encoder.parameters():
            param.requires_grad = True

