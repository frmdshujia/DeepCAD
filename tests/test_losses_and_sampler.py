import numpy as np
import pandas as pd
import torch

from deepcad.data import UniqueParticipantSampler, select_one_image_per_participant
from deepcad.losses import masked_multitask_loss, symmetric_info_nce
from deepcad.training.common import assert_manifest_schema


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


def test_deterministic_single_image_selection_is_label_independent():
    frame = pd.DataFrame({
        "eid": [1, 1, 2, 2],
        "fundus_path": ["b.png", "a.png", "d.png", "c.png"],
        "visit_interval_years": [2.0, -0.5, np.nan, np.nan],
        "fundus_instance": [1, 4, 2, 1],
        "label": [0, 0, 1, 1],
    })
    selected = select_one_image_per_participant(frame)
    assert selected["fundus_path"].tolist() == ["a.png", "c.png"]


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


def test_cmr_focal_loss_differs_from_weighted_bce():
    logits = torch.tensor([[0.0], [2.0]])
    targets = torch.tensor([[1.0], [0.0]])
    mask = torch.ones_like(targets, dtype=torch.bool)
    empty = torch.empty(2, 0)
    empty_mask = torch.empty(2, 0, dtype=torch.bool)
    bce = masked_multitask_loss(
        logits, empty, targets, empty, mask, empty_mask,
        classification_pos_weight=torch.tensor([3.0]), focal_gamma=0.0)
    focal = masked_multitask_loss(
        logits, empty, targets, empty, mask, empty_mask,
        classification_pos_weight=torch.tensor([3.0]), focal_gamma=2.0)
    assert focal > 0
    assert not torch.allclose(focal, bce)


def test_manifest_schema_rejects_cross_split_participant(tmp_path):
    path = tmp_path / "manifest.csv"
    pd.DataFrame({
        "eid": [1, 1], "split": ["train", "val"],
        "fundus_path": ["a.png", "b.png"], "label": [0, 0],
    }).to_csv(path, index=False)
    try:
        assert_manifest_schema(
            path, ["fundus_path", "label"],
            allow_repeated_within_split=True)
    except ValueError as error:
        assert "more than one data split" in str(error)
    else:
        raise AssertionError("Cross-split participant leakage was not rejected.")
