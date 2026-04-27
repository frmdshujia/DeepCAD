"""
models_contrast.py
双塔对比学习模型：
  - FundusContrastModel : RETFound ViT-Large backbone + Projection Head
  - CMREncoder          : 14-dim PC score → embedding MLP
  - build_param_groups  : 分层学习率参数分组（复用 util/lr_decay.py）
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# timm 0.3.2 兼容补丁（依赖已移除的 torch._six）
import collections.abc
if 'torch._six' not in sys.modules:
    class _TorchSix:
        container_abcs = collections.abc
        inf = float('inf')
    sys.modules['torch._six'] = _TorchSix()

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

import models_vit
import util.lr_decay as lrd


# ─────────────────────────────────────────────
#  投影头：1024 → 512 → proj_dim（BN + ReLU）
# ─────────────────────────────────────────────
class ProjectionHead(nn.Module):
    """
    两层 MLP 投影头，将 ViT-Large CLS token（1024-dim）映射到对比空间。
    中间用 BN + ReLU；最后一层无激活（由调用方做 L2 norm）。
    """

    def __init__(self, in_dim: int = 1024, hidden_dim: int = 512, out_dim: int = 256,
                 dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────
#  CMR 编码器：14 → 128 → proj_dim
# ─────────────────────────────────────────────
class CMREncoder(nn.Module):
    """
    将 14 维 CMR PC score 向量编码到对比空间。
    参数量极小（~5 万），全程可训练，每 batch fresh forward 代价可忽略。
    """

    def __init__(self, in_dim: int = 14, hidden_dim: int = 128, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),   # 防止 CMR 嵌入全局坍塌（modality gap）
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (K, 14) PC score 向量
        返回 L2 归一化后的 (K, proj_dim) 嵌入
        """
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ─────────────────────────────────────────────
#  Fundus 编码器：RETFound ViT-Large + 投影头
# ─────────────────────────────────────────────
class FundusContrastModel(nn.Module):
    """
    眼底图像对比学习编码器。
    forward() 调用 backbone.forward_features() 直接拿 CLS token（1024-dim），
    再经投影头压缩为 proj_dim 维，最后 L2 归一化。
    """

    def __init__(self, proj_dim: int = 256, drop_path_rate: float = 0.1):
        super().__init__()
        # ViT-Large（embed_dim=1024，无分类头）
        self.backbone = models_vit.vit_large_patch16(
            img_size=224,
            num_classes=0,           # timm 0.3.2: num_classes=0 → Identity head
            drop_path_rate=drop_path_rate,
            global_pool=False,       # 使用 CLS token，非全局平均池化
        )
        self.proj_head = ProjectionHead(in_dim=1024, hidden_dim=512, out_dim=proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 224, 224)
        返回 L2 归一化后的 (B, proj_dim) 嵌入
        """
        # forward_features 在 global_pool=False 时返回 CLS token (B, 1024)
        features = self.backbone.forward_features(x)
        z = self.proj_head(features)
        return F.normalize(z, dim=-1)

    def load_pretrained(self, ckpt_path: str):
        """加载 RETFound 预训练权重到 backbone，忽略分类头。"""
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        checkpoint_model = checkpoint['model']

        # 移除分类头权重（形状不匹配或不需要）
        for k in list(checkpoint_model.keys()):
            if k.startswith('head'):
                del checkpoint_model[k]

        from util.pos_embed import interpolate_pos_embed
        interpolate_pos_embed(self.backbone, checkpoint_model)

        msg = self.backbone.load_state_dict(checkpoint_model, strict=False)
        print(f'[FundusContrastModel] Loaded RETFound weights from {ckpt_path}')
        print(f'  Missing keys : {msg.missing_keys}')
        print(f'  Unexpected   : {msg.unexpected_keys}')

    def save_encoder_ckpt(self, path: str):
        """
        保存与 main_finetune.py --finetune 格式兼容的 checkpoint。
        仅保存 backbone（ViT）权重，下游微调时直接 --finetune 指向此文件即可。
        """
        ckpt = {'model': self.backbone.state_dict()}
        torch.save(ckpt, path)
        print(f'[FundusContrastModel] Saved encoder checkpoint to {path}')


# ─────────────────────────────────────────────
#  参数分组：分层学习率 + 独立 lr_scale
# ─────────────────────────────────────────────
def build_param_groups(fundus_model: FundusContrastModel,
                       cmr_encoder: CMREncoder,
                       weight_decay: float = 0.05,
                       layer_decay: float = 0.75,
                       proj_lr_scale: float = 10.0,
                       cmr_lr_scale: float = 100.0):
    """
    构建优化器参数分组：
      - ViT backbone : 通过 param_groups_lrd 实现层间衰减（lr_scale ≤ 1.0）
      - Projection head : lr_scale = proj_lr_scale（默认 10x backbone top lr）
      - CMR encoder : lr_scale = cmr_lr_scale（默认 100x backbone top lr）

    调用方将 args.lr 作为基础 LR，lr_sched.adjust_learning_rate 统一处理。
    推荐：
      args.blr = 1e-5  →  backbone top layer LR ≈ 1e-5
      proj_lr_scale=10 →  proj head LR ≈ 1e-4
      cmr_lr_scale=100 →  CMR MLP LR  ≈ 1e-3
    """
    # ViT backbone 层间衰减（lr_scale 由 layer_decay 控制）
    no_wd_list = fundus_model.backbone.no_weight_decay() \
        if hasattr(fundus_model.backbone, 'no_weight_decay') else {}
    backbone_groups = lrd.param_groups_lrd(
        fundus_model.backbone,
        weight_decay=weight_decay,
        no_weight_decay_list=no_wd_list,
        layer_decay=layer_decay,
    )

    # Projection head（较高 lr，从头学）
    proj_group = {
        'params': list(fundus_model.proj_head.parameters()),
        'lr_scale': proj_lr_scale,
        'weight_decay': weight_decay,
    }

    # CMR MLP（最高 lr，参数极少）
    cmr_group = {
        'params': list(cmr_encoder.parameters()),
        'lr_scale': cmr_lr_scale,
        'weight_decay': weight_decay,
    }

    return backbone_groups + [proj_group, cmr_group]
