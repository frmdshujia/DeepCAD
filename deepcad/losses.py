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
    classification_pos_weight: torch.Tensor | None = None,
    classification_weight: float = 0.5,
    regression_weight: float = 0.5,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Balanced loss over task groups and then over tasks within each group.

    For the primary 4-classification + 22-regression teacher this implements
    ``0.5 * mean(task BCE) + 0.5 * mean(task MSE)``. If one task group is absent
    (for example, the legacy eight-regression ablation), the available group is
    assigned unit weight so its effective learning-rate scale is not halved.
    """
    zero = classification_logits.sum() * 0.0 + regression_predictions.sum() * 0.0
    classification_losses = []
    for task in range(classification_logits.shape[1]):
        valid = classification_mask[:, task]
        if valid.any():
            pos_weight = (
                classification_pos_weight[task]
                if classification_pos_weight is not None else None)
            classification_losses.append(F.binary_cross_entropy_with_logits(
                classification_logits[valid, task],
                classification_targets[valid, task],
                pos_weight=pos_weight))
    regression_losses = []
    for task in range(regression_predictions.shape[1]):
        valid = regression_mask[:, task]
        if valid.any():
            regression_losses.append(F.mse_loss(
                regression_predictions[valid, task],
                regression_targets[valid, task]))

    classification_loss = (
        torch.stack(classification_losses).mean()
        if classification_losses else zero)
    regression_loss = (
        torch.stack(regression_losses).mean()
        if regression_losses else zero)
    has_classification = bool(classification_losses)
    has_regression = bool(regression_losses)
    if has_classification and has_regression:
        total = (classification_weight * classification_loss
                 + regression_weight * regression_loss)
    elif has_classification:
        total = classification_loss
    elif has_regression:
        total = regression_loss
    else:
        total = zero
    if return_components:
        return total, classification_loss, regression_loss
    return total
