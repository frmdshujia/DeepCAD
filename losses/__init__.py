"""
损失函数模块
包含监督跨模态对比损失
"""

from .cross_modal_contrastive import (
    cross_modal_contrastive_loss,
    CrossModalContrastiveLoss,
    compute_positive_mask,
    cosine_similarity_matrix
)

__all__ = [
    'cross_modal_contrastive_loss',
    'CrossModalContrastiveLoss',
    'compute_positive_mask',
    'cosine_similarity_matrix'
]

