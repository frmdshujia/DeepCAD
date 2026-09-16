"""Prediction heads used for CMR supervision and clinical fusion."""
from __future__ import annotations

import torch
import torch.nn as nn


class MultiTaskHead(nn.Module):
    def __init__(self, input_dim: int, n_classification: int, n_regression: int):
        super().__init__()
        self.classification_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 1))
            for _ in range(n_classification)
        ])
        self.regression_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(input_dim, 256), nn.GELU(), nn.Linear(256, 1))
            for _ in range(n_regression)
        ])

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        classification = (
            torch.cat([head(features) for head in self.classification_heads], dim=1)
            if self.classification_heads else features.new_zeros((features.shape[0], 0))
        )
        regression = (
            torch.cat([head(features) for head in self.regression_heads], dim=1)
            if self.regression_heads else features.new_zeros((features.shape[0], 0))
        )
        return classification, regression


class ClinicalRiskMLP(nn.Module):
    """Stage III fusion of retinal probability and clinical risk factors."""

    def __init__(self, input_dim: int = 8, hidden_dims: tuple[int, ...] = (32, 16),
                 dropout: float = 0.2):
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([
                nn.Linear(previous, hidden), nn.ReLU(), nn.Dropout(dropout)])
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)
