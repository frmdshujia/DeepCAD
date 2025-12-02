"""
视网膜编码器
基于 RETFound ViT-Large MAE 编码器
"""

import os
import sys
import torch
import torch.nn as nn
from typing import Optional
from functools import partial

# 添加 RETFound 路径以便导入
RETFOUND_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), 
                             'RETFound-main')
if RETFOUND_PATH not in sys.path:
    sys.path.insert(0, RETFOUND_PATH)

try:
    import models_vit as retfound_models
    from util.pos_embed import interpolate_pos_embed
    RETFOUND_AVAILABLE = True
except ImportError:
    RETFOUND_AVAILABLE = False
    print("Warning: RETFound not found. Please ensure RETFound-main is in the parent directory.")


class RetinalEncoder(nn.Module):
    """
    视网膜编码器 f_{θ_R}
    
    基于 RETFound ViT-Large MAE 编码器，丢弃解码器，输出全局视网膜嵌入
    """
    
    def __init__(
        self,
        img_size: int = 224,
        pretrained_path: Optional[str] = None,
        global_pool: bool = True,
        drop_path_rate: float = 0.0,
        freeze_backbone: bool = False
    ):
        """
        初始化视网膜编码器
        
        Args:
            img_size: 输入图像尺寸
            pretrained_path: RETFound 预训练权重路径（可选）
                            - 如果是 HuggingFace Hub ID（如 "RETFound_mae"），会从 Hub 下载
                            - 如果是本地路径，会从本地加载
            global_pool: 是否使用全局平均池化（True）或 CLS token（False）
            drop_path_rate: Drop path rate for stochastic depth
            freeze_backbone: 是否冻结骨干网络参数
        """
        super(RetinalEncoder, self).__init__()
        
        if not RETFOUND_AVAILABLE:
            raise ImportError(
                "RETFound models not available. Please ensure RETFound-main is accessible. "
                "You may need to install it or add it to your path."
            )
        
        # 构建 RETFound ViT-Large 模型
        self.backbone = retfound_models.RETFound_mae(
            img_size=img_size,
            num_classes=0,  # 不需要分类头
            drop_path_rate=drop_path_rate,
            global_pool=global_pool
        )
        
        # 移除分类头（如果存在）
        if hasattr(self.backbone, 'head'):
            self.backbone.head = nn.Identity()
        
        # 嵌入维度: 1024
        # 来源: RETFound ViT-Large 架构定义
        # 参考: RETFound-main/models_vit.py 中的 RETFound_mae() 函数
        #       embed_dim=1024, depth=24, num_heads=16
        # 说明: 这是 Vision Transformer Large (ViT-L) 的标准嵌入维度
        #       与标准 ViT-Large 配置一致 (patch_size=16, embed_dim=1024)
        self.embed_dim = 1024
        self.global_pool = global_pool
        
        # 加载预训练权重
        if pretrained_path:
            self.load_pretrained_weights(pretrained_path)
        
        # 冻结骨干网络（如果需要）
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
    
    def load_pretrained_weights(self, pretrained_path: str):
        """
        加载 RETFound 预训练权重
        
        Args:
            pretrained_path: 权重路径或 HuggingFace Hub ID
        """
        print(f"Loading RETFound pretrained weights from: {pretrained_path}")
        
        # 检查是否是 HuggingFace Hub ID
        if not os.path.exists(pretrained_path) and not pretrained_path.endswith('.pth'):
            # 尝试从 HuggingFace Hub 下载
            try:
                from huggingface_hub import hf_hub_download
                print(f"Downloading from HuggingFace Hub: {pretrained_path}")
                checkpoint_path = hf_hub_download(
                    repo_id=f"YukunZhou/{pretrained_path}",
                    filename=f"{pretrained_path}.pth",
                )
            except Exception as e:
                raise ValueError(
                    f"Could not load pretrained weights from {pretrained_path}. "
                    f"Please provide a valid local path or HuggingFace Hub ID. Error: {e}"
                )
        else:
            checkpoint_path = pretrained_path
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # 提取模型权重
        if 'model' in checkpoint:
            checkpoint_model = checkpoint['model']
        elif 'teacher' in checkpoint:
            checkpoint_model = checkpoint['teacher']
        else:
            checkpoint_model = checkpoint
        
        # 键名清理（RETFound 特定的键名转换）
        checkpoint_model = {k.replace("backbone.", ""): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w12.", "mlp.fc1."): v for k, v in checkpoint_model.items()}
        checkpoint_model = {k.replace("mlp.w3.", "mlp.fc2."): v for k, v in checkpoint_model.items()}
        
        # 移除分类头权重（如果存在）
        state_dict = self.backbone.state_dict()
        for k in ["head.weight", "head.bias"]:
            if k in checkpoint_model and k in state_dict:
                if checkpoint_model[k].shape != state_dict[k].shape:
                    print(f"Removing key {k} from pretrained checkpoint (shape mismatch)")
                    del checkpoint_model[k]
        
        # 插值位置编码（如果需要）
        if 'pos_embed' in checkpoint_model:
            try:
                interpolate_pos_embed(self.backbone, checkpoint_model)
            except Exception as e:
                print(f"Warning: Could not interpolate pos_embed: {e}")
        
        # 加载权重（非严格模式，允许部分不匹配）
        missing_keys, unexpected_keys = self.backbone.load_state_dict(
            checkpoint_model, strict=False
        )
        
        if missing_keys:
            print(f"Missing keys (will use random init): {len(missing_keys)} keys")
        if unexpected_keys:
            print(f"Unexpected keys (ignored): {len(unexpected_keys)} keys")
        
        print("Successfully loaded RETFound pretrained weights")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入视网膜图像，形状为 (batch_size, 3, H, W)
        
        Returns:
            视网膜嵌入，形状为 (batch_size, embed_dim)
        """
        # 使用 forward_features 获取特征（跳过分类头）
        features = self.backbone.forward_features(x)
        
        # features 形状取决于 global_pool 设置
        # 如果 global_pool=True: (B, 1, D) -> (B, D)
        # 如果 global_pool=False: (B, D) 直接返回
        if features.dim() == 3 and features.shape[1] == 1:
            features = features.squeeze(1)
        
        return features
    
    def get_embed_dim(self) -> int:
        """返回嵌入维度"""
        return self.embed_dim

