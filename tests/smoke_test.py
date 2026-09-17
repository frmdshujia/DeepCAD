"""Dependency-light smoke test; run directly when pytest is unavailable."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import deepcad.models.cmr_encoder as cmr_module
from deepcad.data import UniqueParticipantSampler
from deepcad.losses import symmetric_info_nce
from deepcad.training.alignment import (
    alignment_metrics, collect_participant_embeddings,
)


class DummyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.batch_sizes = []

    def forward(self, images):
        self.batch_sizes.append(images.shape[0])
        pooled = images.mean(dim=(1, 2, 3), keepdim=True)
        return pooled.expand(images.shape[0], 768, 7, 7)


def main() -> None:
    original_factory = cmr_module.build_backbone
    dummy_backbone = DummyBackbone()
    cmr_module.build_backbone = lambda *args, **kwargs: dummy_backbone
    try:
        model = cmr_module.CMREncoderV4(
            spatial_pool=2, transformer_depth=1,
            fusion_mode="hierarchical", backbone_ckpt=None)
        images = torch.randn(2, 16, 1, 32, 32)
        projection, representation = model(
            images, t1_available=torch.tensor([True, False]))
        assert projection.shape == (2, 256)
        assert representation.shape == (2, 768)
        # 2*15 cine frames plus only the one observed T1 map. The missing T1
        # sample must never enter the image backbone.
        assert dummy_backbone.batch_sizes == [30, 1]
        _, mask = model._tokenise(
            images, t1_available=torch.tensor([True, False]),
            return_key_padding_mask=True)
        assert mask[0].sum().item() == 0
        assert mask[1].sum().item() == 4
    finally:
        cmr_module.build_backbone = original_factory

    matched = torch.eye(4)
    assert symmetric_info_nce(matched, matched) < symmetric_info_nce(
        matched, matched.flip(0))
    dataset = type("Dataset", (), {"eids": [1, 1, 2, 3, 3]})()
    selected = list(UniqueParticipantSampler(dataset, seed=7))
    assert len(selected) == 3
    assert len({dataset.eids[index] for index in selected}) == 3

    class FakeFundusEncoder(nn.Module):
        def forward(self, images):
            projection = torch.nn.functional.normalize(images, dim=-1)
            return projection, images

    participant_batch = {
        "eid": torch.tensor([10, 20]),
        "image": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "cmr_embedding": torch.tensor(
            [[1.0, 1.0], [0.0, 1.0]]),
        "t1_available": torch.tensor([True, False]),
    }
    eids, fundus, cmr, t1 = collect_participant_embeddings(
        FakeFundusEncoder(), nn.Identity(), [participant_batch],
        torch.device("cpu"), "none")
    assert eids.tolist() == [10, 20]
    assert t1.tolist() == [True, False]
    metrics = alignment_metrics(fundus, cmr, torch.tensor(1 / 0.1).log())
    assert metrics["n"] == 2
    print("smoke_test: PASS")


if __name__ == "__main__":
    main()
