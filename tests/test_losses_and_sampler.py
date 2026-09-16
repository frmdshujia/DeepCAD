import numpy as np
import torch

from deepcad.data import UniqueParticipantSampler
from deepcad.losses import symmetric_info_nce


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
