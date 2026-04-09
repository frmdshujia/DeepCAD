import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.25):
        """Initializer for FocalLoss class."""
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, pred, label):
        """Calculates Focal Loss."""
        label = F.one_hot(label, num_classes=2).float() 
        loss = F.binary_cross_entropy_with_logits(pred, label, reduction='none')
        pred_prob = pred.sigmoid()  # prob from logits
        p_t = label * pred_prob + (1 - label) * (1 - pred_prob)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= modulating_factor
        if self.alpha > 0:
            alpha_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
            loss *= alpha_factor
        return loss.mean()  # 或者 loss.sum()，根据需求选择