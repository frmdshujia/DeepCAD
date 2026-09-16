"""Loss functions for the DeepCAD development stages."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def symmetric_info_nce(
    fundus_projection: torch.Tensor,
    cmr_projection: torch.Tensor,
    temperature: torch.Tensor | float = 0.07,
) -> torch.Tensor:
    fundus_projection = F.normalize(fundus_projection, dim=1)
    cmr_projection = F.normalize(cmr_projection, dim=1)
    scale = 1.0 / temperature if not isinstance(temperature, torch.Tensor) \
        else temperature.reciprocal()
    logits = scale * fundus_projection @ cmr_projection.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def masked_multitask_loss(
    classification_logits: torch.Tensor,
    regression_predictions: torch.Tensor,
    classification_targets: torch.Tensor,
    regression_targets: torch.Tensor,
    classification_mask: torch.Tensor,
    regression_mask: torch.Tensor,
    regression_weight: float = 1.0,
) -> torch.Tensor:
    losses = []
    if classification_logits.numel() and classification_mask.any():
        losses.append(F.binary_cross_entropy_with_logits(
            classification_logits[classification_mask],
            classification_targets[classification_mask]))
    if regression_predictions.numel() and regression_mask.any():
        losses.append(regression_weight * F.mse_loss(
            regression_predictions[regression_mask],
            regression_targets[regression_mask]))
    if not losses:
        return classification_logits.sum() * 0.0 + regression_predictions.sum() * 0.0
    return torch.stack(losses).sum()
