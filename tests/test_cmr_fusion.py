import torch
import torch.nn as nn

import deepcad.models.cmr_encoder as cmr_module


class DummyBackbone(nn.Module):
    def forward(self, images):
        pooled = images.mean(dim=(1, 2, 3), keepdim=True)
        return pooled.expand(images.shape[0], 768, 7, 7)


def build_model(monkeypatch, fusion_mode="hierarchical"):
    monkeypatch.setattr(
        cmr_module, "build_backbone", lambda *args, **kwargs: DummyBackbone())
    return cmr_module.CMREncoderV4(
        spatial_pool=2, transformer_depth=1, fusion_mode=fusion_mode,
        backbone="medsam2_heart", backbone_ckpt=None)


def test_hierarchical_forward_shapes(monkeypatch):
    model = build_model(monkeypatch)
    images = torch.randn(2, 16, 1, 32, 32)
    projection, representation = model(
        images, t1_available=torch.tensor([True, False]))
    assert projection.shape == (2, 256)
    assert representation.shape == (2, 768)


def test_missing_t1_masks_exact_t1_region(monkeypatch):
    model = build_model(monkeypatch)
    images = torch.randn(2, 16, 1, 32, 32)
    _, mask = model._tokenise(
        images, t1_available=torch.tensor([True, False]),
        return_key_padding_mask=True)
    assert mask[0].sum().item() == 0
    assert mask[1].sum().item() == 4


def test_bidirectional_attention_preserves_stream_lengths():
    module = cmr_module.BidirectionalCrossAttention(dim=32, num_heads=4)
    a, b = module(torch.randn(2, 5, 32), torch.randn(2, 7, 32))
    assert a.shape == (2, 5, 32)
    assert b.shape == (2, 7, 32)
