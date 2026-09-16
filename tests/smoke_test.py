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


class DummyBackbone(nn.Module):
    def forward(self, images):
        pooled = images.mean(dim=(1, 2, 3), keepdim=True)
        return pooled.expand(images.shape[0], 768, 7, 7)


def main() -> None:
    original_factory = cmr_module.build_backbone
    cmr_module.build_backbone = lambda *args, **kwargs: DummyBackbone()
    try:
        model = cmr_module.CMREncoderV4(
            spatial_pool=2, transformer_depth=1,
            fusion_mode="hierarchical", backbone_ckpt=None)
        images = torch.randn(2, 16, 1, 32, 32)
        projection, representation = model(
            images, t1_available=torch.tensor([True, False]))
        assert projection.shape == (2, 256)
        assert representation.shape == (2, 768)
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
    print("smoke_test: PASS")


if __name__ == "__main__":
    main()
