"""
心脏MRI编码器
基于 MedSAM ViT-Base 骨干网络
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional

# 添加 MedSAM 路径以便导入
MEDSAM_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                          'MedSAM-main')
if MEDSAM_PATH not in sys.path:
    sys.path.insert(0, MEDSAM_PATH)

try:
    from segment_anything import build_sam_vit_b, sam_model_registry
    MEDSAM_AVAILABLE = True
except ImportError:
    MEDSAM_AVAILABLE = False
    print("Warning: MedSAM not found. Please ensure MedSAM-main is in the parent directory.")

from .pooling import AttentionPooling, LearnableWeightedPooling, MeanPooling


class MRICardioEncoder(nn.Module):
    """
    心脏MRI编码器 f_{θ_C}
    
    基于 MedSAM ViT-Base 骨干网络，处理多切片输入并聚合为主体级嵌入
    """
    
    def __init__(
        self,
        img_size: int = 224,
        pretrained_path: Optional[str] = None,
        pooling_type: str = "attention",
        freeze_backbone: bool = False,
        num_slices: Optional[int] = None
    ):
        """
        初始化MRI编码器
        
        Args:
            img_size: 输入图像尺寸（MedSAM默认使用1024，但我们可以适配）
            pretrained_path: MedSAM 预训练权重路径（可选）
            pooling_type: 池化类型 ("attention", "learnable_weighted", "mean", "max")
            freeze_backbone: 是否冻结骨干网络参数
            num_slices: 预期的切片数量（用于初始化池化层，如果为None则动态适应）
        """
        super(MRICardioEncoder, self).__init__()
        
        if not MEDSAM_AVAILABLE:
            raise ImportError(
                "MedSAM not available. Please ensure MedSAM-main is accessible and its "
                "Python dependencies are installed (see docs/ENVIRONMENT_SETUP.md)."
            )
        
        if pretrained_path is None:
            raise ValueError(
                "MRICardioEncoder requires a MedSAM ViT-B checkpoint. "
                "Please download the official `medsam_vit_b.pth` (e.g. from "
                "https://github.com/bowang-lab/MedSAM) and pass the local path via "
                "`--mri_pretrained /path/to/medsam_vit_b.pth`."
            )
        
        if not os.path.isfile(pretrained_path):
            raise FileNotFoundError(
                f"MedSAM checkpoint not found: {pretrained_path}. "
                "Verify the path or download the weights (medsam_vit_b.pth) and "
                "provide the absolute path."
            )
        
        try:
            sam_model = sam_model_registry["vit_b"](checkpoint=pretrained_path)
            self.backbone = sam_model.image_encoder
            print(f"Loaded MedSAM pretrained weights from: {pretrained_path}")
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load MedSAM checkpoint from {pretrained_path}. "
                "Ensure the file is a valid ViT-B MedSAM weight file (medsam_vit_b.pth). "
                f"Original error: {exc}"
            ) from exc
        

        # MedSAM ViT-Base 的输出通道数（out_chans）
        # 维度: 256
        # 来源: MedSAM-main/segment_anything/build_sam.py 中的 _build_sam() 函数
        #       prompt_embed_dim = 256 (这是 SAM 架构中的 prompt embedding 维度)
        #       ImageEncoderViT 的 out_chans=prompt_embed_dim=256
        # 说明: MedSAM image_encoder 输出形状为 (B, 256, 64, 64) for 1024x1024 input
        #       这是 SAM 架构的设计，用于与 prompt encoder 和 mask decoder 对接
        self.backbone_out_dim = 256
        
        # 投影层目标维度: 768
        # 来源: MedSAM ViT-Base 的 encoder_embed_dim
        # 参考: MedSAM-main/segment_anything/build_sam.py 中的 build_sam_vit_b()
        #       encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12
        # 说明: 这是 Vision Transformer Base (ViT-B) 的标准嵌入维度
        #       虽然 MedSAM 输出是 256 维（用于 SAM 任务），但我们将其投影回 768 维
        #       以匹配 ViT-Base 的原始特征维度，便于后续处理
        # 设计选择: 使用 768 而不是 256，因为：
        #           1. 768 是 ViT-Base 的标准维度，特征更丰富
        #           2. 与视网膜编码器的 1024 维在数量级上更接近
        #           3. 便于后续投影头设计（两个编码器输出维度差异不会太大）
        self.embed_dim = 768
        self.slice_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # (B, 256, 1, 1)
            nn.Flatten(),  # (B, 256)
            nn.Linear(self.backbone_out_dim, self.embed_dim),  # (B, 768)
            nn.LayerNorm(self.embed_dim)
        )
        
        # 池化层：将切片级特征聚合为主体级嵌入
        if pooling_type == "attention":
            # 注意力池化参数:
            # - embed_dim: 768 (MRI编码器输出维度)
            # - num_heads: 8 (多头注意力头数，常见设置)
            # - dropout: 0.1 (Dropout比率，防止过拟合)
            self.pooling = AttentionPooling(
                embed_dim=self.embed_dim,  # 768
                num_heads=8,  # 默认8头，768/8=96维每头
                dropout=0.1  # 默认0.1
            )
        elif pooling_type == "learnable_weighted":
            self.pooling = LearnableWeightedPooling(embed_dim=self.embed_dim)
        elif pooling_type == "mean":
            self.pooling = MeanPooling(embed_dim=self.embed_dim)
        elif pooling_type == "max":
            from .pooling import MaxPooling
            self.pooling = MaxPooling(embed_dim=self.embed_dim)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        self.pooling_type = pooling_type
        
        # 冻结骨干网络（如果需要）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入MRI切片，形状为 (batch_size, num_slices, 1, H, W) 或 (batch_size, num_slices, H, W)
               注意：MedSAM 期望 3 通道输入，但MRI通常是单通道
               我们会在内部处理通道转换
        
        Returns:
            MRI嵌入，形状为 (batch_size, embed_dim)
        """
        batch_size = x.shape[0]
        
        # 处理输入形状
        if x.dim() == 5:
            # (B, num_slices, 1, H, W)
            num_slices = x.shape[1]
            x = x.view(batch_size * num_slices, *x.shape[2:])  # (B*num_slices, 1, H, W)
        elif x.dim() == 4:
            # (B, num_slices, H, W) - 假设只有一个通道维度
            num_slices = x.shape[1]
            x = x.unsqueeze(2)  # (B, num_slices, 1, H, W)
            x = x.view(batch_size * num_slices, 1, *x.shape[3:])  # (B*num_slices, 1, H, W)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}")
        
        # MedSAM 期望 3 通道输入，我们需要将单通道复制为 3 通道
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # (B*num_slices, 3, H, W)
        
        # 调整图像尺寸到 MedSAM 期望的 1024x1024（如果需要）
        # 图像尺寸: 1024x1024
        # 来源: MedSAM-main/segment_anything/build_sam.py 中的 _build_sam() 函数
        #       image_size = 1024 (这是 SAM/MedSAM 架构的标准输入尺寸)
        # 说明: 虽然我们的数据可能是 224x224，但 MedSAM 编码器期望 1024x1024 输入
        #       因此需要上采样（使用双线性插值）
        # 注意: 如果输入已经是 224x224，我们需要上采样，不直接使用1024作为输入是为了兼容内存有限的GPU
        if x.shape[-1] != 1024:
            x = nn.functional.interpolate(
                x, size=(1024, 1024), mode='bilinear', align_corners=False
            )
        
        # 通过骨干网络提取特征
        # MedSAM image_encoder 输出: (B*num_slices, 256, 64, 64)
        slice_features = self.backbone(x)  # (B*num_slices, 256, 64, 64)
        
        # 投影到统一嵌入维度
        # slice_features: (B*num_slices, 256, 64, 64)
        slice_embeddings = self.slice_proj(slice_features)  # (B*num_slices, 768)
        
        # 重塑为 (batch_size, num_slices, embed_dim)
        slice_embeddings = slice_embeddings.view(batch_size, num_slices, self.embed_dim)
        
        # 池化：聚合切片级特征为主体级嵌入
        subject_embedding = self.pooling(slice_embeddings)  # (batch_size, embed_dim)
        
        return subject_embedding
    
    def get_embed_dim(self) -> int:
        """返回嵌入维度"""
        return self.embed_dim
    
    def load_pretrained_weights(self, checkpoint_path: str):
        """
        加载 MedSAM 预训练权重
        
        Args:
            checkpoint_path: 检查点路径
        """
        print(f"Loading MedSAM pretrained weights from: {checkpoint_path}")
        
        try:
            sam_model = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
            # 只加载 image_encoder 的权重
            self.backbone.load_state_dict(sam_model.image_encoder.state_dict())
            print("Successfully loaded MedSAM pretrained weights")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Continuing with randomly initialized weights...")

