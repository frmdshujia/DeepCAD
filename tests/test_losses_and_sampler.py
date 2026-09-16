import numpy as np
import torch

from deepcad.data import UniqueParticipantSampler
from deepcad.losses import masked_multitask_loss, symmetric_info_nce


def test_infonce_prefers_matched_pairs():
    matched = torch.eye(4)
    mismatched = matched.flip(0)
    assert symmetric_info_nce(matched, matched) < symmetric_info_nce(
        matched, mismatched)


def test_unique_participant_sampler():
    dataset = type("Dataset", (), {"eids": [1, 1, 2, 3, 3, 3]})()
    sampler = UniqueParticipantSampler(dataset, seed=4)
    indices = list(sampler)
    selected = [dataset.eids[index] for index in indices]
    assert len(indices) == 3
    assert len(set(selected)) == 3


def test_balanced_multitask_loss_uses_per_task_pos_weight():
    cls_logits = torch.zeros(2, 2)
    cls_targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    reg_prediction = torch.zeros(2, 1)
    reg_targets = torch.ones(2, 1)
    total, cls, reg = masked_multitask_loss(
        cls_logits, reg_prediction, cls_targets, reg_targets,
        torch.ones_like(cls_targets, dtype=torch.bool),
        torch.ones_like(reg_targets, dtype=torch.bool),
        classification_pos_weight=torch.tensor([2.0, 3.0]),
        return_components=True)
    assert torch.allclose(total, 0.5 * cls + 0.5 * reg)
