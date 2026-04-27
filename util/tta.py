"""
Test-Time Augmentation (TTA)：对同一张图做多种确定性几何变换，模型 logits 取平均后再 softmax。

用于验证/推理；训练路径请勿使用。
支持的视角名：orig, hflip, vflip, rot90, rot180, rot270

默认几何与 contrastive_pretrain/datasets_contrast 训练增强对齐思路一致：
  训练侧为 RandomHorizontalFlip + RandomVerticalFlip + RandomRotation(±180°)；
  TTA 侧用原图 + 两轴翻转 + 90° 步进旋转（D4 离散子群），共 7 次前向。
"""

from __future__ import annotations

import torch

_VALID_MODES = frozenset({"orig", "hflip", "vflip", "rot90", "rot180", "rot270"})

# 与对比学习「几何策略」默认对齐（见模块 docstring）；推理耗时约为单视角的 7 倍
DEFAULT_TTA_MODES = "orig,hflip,vflip,rot90,rot180,rot270"


def parse_tta_modes(s: str) -> list[str]:
    """逗号分隔；去空格并校验。默认见 DEFAULT_TTA_MODES。"""
    modes = [x.strip() for x in s.split(",") if x.strip()]
    if not modes:
        raise ValueError("tta_modes 不能为空")
    bad = [m for m in modes if m not in _VALID_MODES]
    if bad:
        raise ValueError(f"未知 TTA 模式: {bad}，可选: {sorted(_VALID_MODES)}")
    return modes


def apply_tta_view(x: torch.Tensor, mode: str) -> torch.Tensor:
    """对 batch 图像张量 (B,C,H,W) 做几何变换。"""
    if mode == "orig":
        return x
    if mode == "hflip":
        return torch.flip(x, (3,))
    if mode == "vflip":
        return torch.flip(x, (2,))
    if mode == "rot90":
        return torch.rot90(x, k=1, dims=(2, 3))
    if mode == "rot180":
        return torch.rot90(x, k=2, dims=(2, 3))
    if mode == "rot270":
        return torch.rot90(x, k=3, dims=(2, 3))
    raise ValueError(mode)


@torch.no_grad()
def forward_logits_tta(model: torch.nn.Module, images: torch.Tensor, modes: list[str]) -> torch.Tensor:
    """
    对每个视角分别前向，在 logits 上算术平均（与多数 TTA 实现一致）。

    Args:
        model: eval 模式下的分类模型
        images: (B, C, H, W)，已与训练一致的 normalize
        modes: 如 ['orig', 'hflip']

    Returns:
        (B, num_classes) logits
    """
    outs = []
    for m in modes:
        x = apply_tta_view(images, m)
        outs.append(model(x))
    return torch.stack(outs, dim=0).mean(dim=0)
