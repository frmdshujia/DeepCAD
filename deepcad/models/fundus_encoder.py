"""RETFound-compatible fundus encoder with explicit bottleneck adapters."""
from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer


class Adapter(nn.Module):
    def __init__(self, dim: int = 1024, bottleneck_dim: int = 64):
        super().__init__()
        self.down = nn.Linear(dim, bottleneck_dim)
        self.activation = nn.GELU()
        self.up = nn.Linear(bottleneck_dim, dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.up(self.activation(self.down(tokens)))


class RETFoundFundusEncoder(nn.Module):
    """ViT-Large retinal encoder used for Stage I and Stage II.

    The implementation avoids forward hooks: every transformer block is followed
    explicitly by its residual adapter, making the released forward path easy to
    inspect. Official RETFound checkpoints can be loaded with
    :meth:`load_retfound_weights`.
    """

    def __init__(
        self,
        adapter_dim: int = 64,
        global_pool: bool = True,
        projection_dim: int = 128,
    ):
        super().__init__()
        self.global_pool = global_pool
        self.backbone = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            num_classes=0,
            embed_dim=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
        # RETFound's global-pooling variant removes the CLS-token norm and
        # applies a separate fc_norm after mean-pooling patch tokens.
        self.backbone.norm = nn.Identity()
        self.adapters = nn.ModuleList(
            [Adapter(1024, adapter_dim) for _ in self.backbone.blocks])
        self.pool_norm = nn.LayerNorm(1024)
        self.projection = nn.Sequential(
            nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, projection_dim))

    def load_retfound_weights(self, checkpoint_path: str) -> None:
        path = Path(checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        checkpoint = torch.load(path, map_location="cpu")
        state = checkpoint.get("model", checkpoint)
        pool_state = {
            key.removeprefix("fc_norm."): value for key, value in state.items()
            if key.startswith("fc_norm.")}
        candidate_state = {k: v for k, v in state.items()
                           if not k.startswith("head.")
                           and not k.startswith("fc_norm.")}
        target_state = self.backbone.state_dict()
        state = {
            key: value for key, value in candidate_state.items()
            if key in target_state and value.shape == target_state[key].shape}
        missing, unexpected = self.backbone.load_state_dict(state, strict=False)
        if pool_state:
            self.pool_norm.load_state_dict(pool_state, strict=True)
        real_missing = [k for k in missing if not k.startswith("head.")]
        skipped = len(candidate_state) - len(state)
        if real_missing or unexpected:
            raise RuntimeError(
                f"RETFound encoder mismatch: missing={real_missing}, "
                f"unexpected={unexpected}")
        print(f"RETFound encoder loaded ({len(state)} tensors; "
              f"ignored {skipped} non-encoder tensors)")

    def configure_trainable(self, unfreeze_last_blocks: int = 12) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapters.parameters():
            parameter.requires_grad_(True)
        for parameter in self.pool_norm.parameters():
            parameter.requires_grad_(True)
        for parameter in self.projection.parameters():
            parameter.requires_grad_(True)
        if unfreeze_last_blocks > 0:
            for block in self.backbone.blocks[-unfreeze_last_blocks:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            for parameter in self.backbone.norm.parameters():
                parameter.requires_grad_(True)

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        backbone = self.backbone
        tokens = backbone.patch_embed(images)
        cls = backbone.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        tokens = backbone.pos_drop(tokens + backbone.pos_embed)
        if hasattr(backbone, "patch_drop"):
            tokens = backbone.patch_drop(tokens)
        if hasattr(backbone, "norm_pre"):
            tokens = backbone.norm_pre(tokens)
        for block, adapter in zip(backbone.blocks, self.adapters):
            tokens = block(tokens)
            tokens = tokens + adapter(tokens)
        if self.global_pool:
            return self.pool_norm(tokens[:, 1:].mean(dim=1))
        return backbone.norm(tokens)[:, 0]

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.forward_features(images)
        projection = F.normalize(self.projection(features), dim=-1)
        return projection, features


class FundusBinaryClassifier(nn.Module):
    def __init__(self, encoder: RETFoundFundusEncoder):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(1024, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        _, features = self.encoder(images)
        return self.classifier(features).squeeze(-1)
