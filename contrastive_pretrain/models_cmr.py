"""CMR encoder for Stage 2 image-level contrastive learning.

Architecture
------------
- Backbone: SAM ViT-B (initialized from MedSAM v1 weights, HuggingFace
  `flaviagiammarino/medsam-vit-base`). We re-implement a minimal SAM vision
  encoder to avoid depending on `transformers` (Py3.7 env constraint).
- Input: 4 grayscale keyframes per subject at 224x224. We replicate to
  3-channel.
- Positional embedding is bicubically interpolated from 64x64 (1024/16) to
  14x14 (224/16) once at init.
- Because the MedSAM window size is 14 and our patch grid is also 14x14, the
  windowed-attention layers collapse to global attention, so we treat all 12
  blocks uniformly while still honoring the decomposed relative positional
  bias (rel_pos_h / rel_pos_w, shape (27, head_dim)).
- Multi-frame aggregation: learned attention pool 4->1.
- Output: (B, proj_dim) contrastive embedding + (B, embed_dim) raw feature.

Weight key mapping (HF -> ours)
-------------------------------
- `vision_encoder.pos_embed`                          -> `pos_embed` (interpolate)
- `vision_encoder.patch_embed.projection.{weight,bias}` -> `patch_embed.proj.{weight,bias}`
- `vision_encoder.layers.{i}.layer_norm{1,2}.*`       -> `blocks.{i}.norm{1,2}.*`
- `vision_encoder.layers.{i}.attn.{qkv,proj}.*`       -> `blocks.{i}.attn.{qkv,proj}.*`
- `vision_encoder.layers.{i}.attn.rel_pos_{h,w}`      -> `blocks.{i}.attn.rel_pos_{h,w}`
- `vision_encoder.layers.{i}.mlp.lin{1,2}.*`          -> `blocks.{i}.mlp.fc{1,2}.*`
- neck / prompt_encoder / mask_decoder are dropped.
"""
from __future__ import annotations
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------
# Decomposed relative positional bias (per SAM image_encoder.py)
# ----------------------------------------------------------------------------

def _get_rel_pos(q_size: int, k_size: int, rel_pos: torch.Tensor) -> torch.Tensor:
    """Re-samples rel_pos from (2*max(q,k)-1, C) to (q_size, k_size, C)."""
    max_rel_dist = int(2 * max(q_size, k_size) - 1)
    if rel_pos.shape[0] != max_rel_dist:
        rel_pos_resized = F.interpolate(
            rel_pos.reshape(1, rel_pos.shape[0], -1).permute(0, 2, 1),
            size=max_rel_dist, mode="linear",
        )
        rel_pos_resized = rel_pos_resized.reshape(-1, max_rel_dist).permute(1, 0)
    else:
        rel_pos_resized = rel_pos
    q_coords = torch.arange(q_size, device=rel_pos.device)[:, None] * max(k_size / q_size, 1.0)
    k_coords = torch.arange(k_size, device=rel_pos.device)[None, :] * max(q_size / k_size, 1.0)
    relative_coords = (q_coords - k_coords) + (k_size - 1) * max(q_size / k_size, 1.0)
    return rel_pos_resized[relative_coords.long()]


def _add_decomposed_rel_pos(
    attn: torch.Tensor, q: torch.Tensor,
    rel_pos_h: torch.Tensor, rel_pos_w: torch.Tensor,
    q_size: Tuple[int, int], k_size: Tuple[int, int],
) -> torch.Tensor:
    """attn: (B*nH, H*W, H*W). q: (B*nH, H*W, C). Adds decomposed rel pos bias."""
    q_h, q_w = q_size
    k_h, k_w = k_size
    Rh = _get_rel_pos(q_h, k_h, rel_pos_h)
    Rw = _get_rel_pos(q_w, k_w, rel_pos_w)
    B, _, C = q.shape
    r_q = q.reshape(B, q_h, q_w, C)
    rel_h = torch.einsum("bhwc,hkc->bhwk", r_q, Rh)
    rel_w = torch.einsum("bhwc,wkc->bhwk", r_q, Rw)
    attn = attn.view(B, q_h, q_w, k_h, k_w)
    attn = attn + rel_h[:, :, :, :, None] + rel_w[:, :, :, None, :]
    return attn.view(B, q_h * q_w, k_h * k_w)


# ----------------------------------------------------------------------------
# Core blocks
# ----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                     # (B, C, H, W)
        return x.permute(0, 2, 3, 1)          # (B, H, W, C)


class MLPBlock(nn.Module):
    def __init__(self, embed_dim: int, mlp_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, embed_dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class SamAttention(nn.Module):
    """rel_pos is stored with size (2*rel_pos_size - 1, head_dim); if
    rel_pos_size differs from the runtime grid size, `_get_rel_pos` will
    bicubically resample it on the fly. This lets us keep HF's native rel_pos
    tensors verbatim (windowed layers keep 27x64; global layers keep 127x64)
    while running on any grid size."""

    def __init__(self, dim: int = 768, num_heads: int = 12,
                 rel_pos_size: int = 14, use_rel_pos: bool = True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)
        self.use_rel_pos = use_rel_pos
        if use_rel_pos:
            self.rel_pos_h = nn.Parameter(torch.zeros(2 * rel_pos_size - 1, head_dim))
            self.rel_pos_w = nn.Parameter(torch.zeros(2 * rel_pos_size - 1, head_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, H, W, _ = x.shape
        qkv = self.qkv(x).reshape(B, H * W, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q = q.reshape(B * self.num_heads, H * W, -1)
        k = k.reshape(B * self.num_heads, H * W, -1)
        v = v.reshape(B * self.num_heads, H * W, -1)

        attn = (q * self.scale) @ k.transpose(-2, -1)
        if self.use_rel_pos:
            attn = _add_decomposed_rel_pos(attn, q, self.rel_pos_h, self.rel_pos_w, (H, W), (H, W))
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, H, W, -1).permute(0, 2, 3, 1, 4).reshape(B, H, W, -1)
        return self.proj(x)


class SamBlock(nn.Module):
    def __init__(self, dim: int = 768, num_heads: int = 12, mlp_ratio: float = 4.0,
                 rel_pos_size: int = 14):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SamAttention(dim=dim, num_heads=num_heads, rel_pos_size=rel_pos_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLPBlock(dim, int(dim * mlp_ratio))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SamVisionEncoderViTB(nn.Module):
    """Minimal SAM ViT-B vision encoder reproducing the forward structure used by
    MedSAM, with 14x14 patch grid (224 input / 16 patch)."""

    # SAM ViT-B global-attention layer indices (others are windowed w/ window=14)
    GLOBAL_ATTN_IDX = (2, 5, 8, 11)
    ORIG_WINDOW_SIZE = 14
    ORIG_GRID_SIZE = 64  # SAM default = 1024 / 16

    def __init__(self, img_size: int = 224, patch_size: int = 16, in_chans: int = 3,
                 embed_dim: int = 768, depth: int = 12, num_heads: int = 12,
                 mlp_ratio: float = 4.0):
        super().__init__()
        assert img_size % patch_size == 0
        self.grid_size = img_size // patch_size  # 14 for 224
        self.patch_embed = PatchEmbed(patch_size, in_chans, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.grid_size, self.grid_size, embed_dim))
        blocks = []
        for i in range(depth):
            rps = self.ORIG_GRID_SIZE if i in self.GLOBAL_ATTN_IDX else self.ORIG_WINDOW_SIZE
            blocks.append(SamBlock(embed_dim, num_heads, mlp_ratio, rel_pos_size=rps))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x  # (B, 14, 14, 768)


# ----------------------------------------------------------------------------
# Weight loading: map HuggingFace flaviagiammarino/medsam-vit-base to our keys
# ----------------------------------------------------------------------------

def _interpolate_pos_embed_hf(pos: torch.Tensor, target_hw: int) -> torch.Tensor:
    """pos: (1, 64, 64, 768) -> (1, target_hw, target_hw, 768) via bicubic."""
    if pos.shape[1] == target_hw:
        return pos
    pos = pos.permute(0, 3, 1, 2)                       # (1, 768, 64, 64)
    pos = F.interpolate(pos, size=(target_hw, target_hw), mode='bicubic', align_corners=False)
    return pos.permute(0, 2, 3, 1).contiguous()


def load_medsam_vision_encoder_weights(
    model: SamVisionEncoderViTB, ckpt_path: str, strict: bool = True, verbose: bool = True
) -> dict:
    """Load flaviagiammarino/medsam-vit-base pytorch_model.bin into our model."""
    sd = torch.load(ckpt_path, map_location='cpu')
    ve = {k: v for k, v in sd.items() if k.startswith('vision_encoder.')}
    new_sd = {}

    # pos_embed (interpolate 64x64 -> grid_size x grid_size)
    pos = ve['vision_encoder.pos_embed']  # (1, 64, 64, 768)
    new_sd['pos_embed'] = _interpolate_pos_embed_hf(pos, model.grid_size)

    # patch embed
    new_sd['patch_embed.proj.weight'] = ve['vision_encoder.patch_embed.projection.weight']
    new_sd['patch_embed.proj.bias']   = ve['vision_encoder.patch_embed.projection.bias']

    # blocks
    depth = len(model.blocks)
    for i in range(depth):
        src = f'vision_encoder.layers.{i}'
        dst = f'blocks.{i}'
        mapping = {
            f'{dst}.norm1.weight':         f'{src}.layer_norm1.weight',
            f'{dst}.norm1.bias':           f'{src}.layer_norm1.bias',
            f'{dst}.norm2.weight':         f'{src}.layer_norm2.weight',
            f'{dst}.norm2.bias':           f'{src}.layer_norm2.bias',
            f'{dst}.attn.qkv.weight':      f'{src}.attn.qkv.weight',
            f'{dst}.attn.qkv.bias':        f'{src}.attn.qkv.bias',
            f'{dst}.attn.proj.weight':     f'{src}.attn.proj.weight',
            f'{dst}.attn.proj.bias':       f'{src}.attn.proj.bias',
            f'{dst}.attn.rel_pos_h':       f'{src}.attn.rel_pos_h',
            f'{dst}.attn.rel_pos_w':       f'{src}.attn.rel_pos_w',
            f'{dst}.mlp.fc1.weight':       f'{src}.mlp.lin1.weight',
            f'{dst}.mlp.fc1.bias':         f'{src}.mlp.lin1.bias',
            f'{dst}.mlp.fc2.weight':       f'{src}.mlp.lin2.weight',
            f'{dst}.mlp.fc2.bias':         f'{src}.mlp.lin2.bias',
        }
        for k_dst, k_src in mapping.items():
            new_sd[k_dst] = ve[k_src]

    missing, unexpected = model.load_state_dict(new_sd, strict=strict)
    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f'[medsam] loaded {len(new_sd)} tensors into model ({n_params/1e6:.1f}M params)')
        if missing:
            print(f'[medsam] MISSING: {missing[:5]}... ({len(missing)} total)')
        if unexpected:
            print(f'[medsam] UNEXPECTED: {unexpected[:5]}... ({len(unexpected)} total)')
    return {'missing': missing, 'unexpected': unexpected, 'loaded': len(new_sd)}


# ----------------------------------------------------------------------------
# Frame attention pool + projection head
# ----------------------------------------------------------------------------

class FrameAttnPool(nn.Module):
    """Learnable-query cross-attention to pool N frame tokens to 1 summary.
    (Implements `batch_first=True` semantics manually for PyTorch 1.8 compat.)"""
    def __init__(self, dim: int = 768, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        nn.init.trunc_normal_(self.query, std=0.02)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout)
        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, D) -> MHA wants (L, B, D)
        B = x.shape[0]
        q = self.query.expand(B, -1, -1).contiguous()        # (B, 1, D)
        kv = self.norm_kv(x)                                  # (B, N, D)
        q_t = q.transpose(0, 1)                               # (1, B, D)
        kv_t = kv.transpose(0, 1)                             # (N, B, D)
        out, _ = self.attn(q_t, kv_t, kv_t, need_weights=False)
        out = out.transpose(0, 1)                             # (B, 1, D)
        return self.norm_out(out.squeeze(1))                  # (B, D)


class CMREncoder(nn.Module):
    """MedSAM ViT-B backbone + spatial mean pool + frame attention pool + proj head."""

    def __init__(self, proj_dim: int = 256, num_frames: int = 4,
                 embed_dim: int = 768, img_size: int = 224,
                 medsam_ckpt: Optional[str] = None,
                 freeze_backbone: bool = False):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.backbone = SamVisionEncoderViTB(
            img_size=img_size, patch_size=16, in_chans=3,
            embed_dim=embed_dim, depth=12, num_heads=12, mlp_ratio=4.0,
        )
        if medsam_ckpt is not None:
            load_medsam_vision_encoder_weights(self.backbone, medsam_ckpt, strict=True, verbose=True)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.frame_pool = FrameAttnPool(dim=embed_dim, num_heads=4)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, F, 1, 224, 224) grayscale (or (B, F, 224, 224)).
        Returns (proj_emb (B, proj_dim), raw_emb (B, embed_dim))."""
        if x.dim() == 4:           # (B, F, H, W)
            x = x.unsqueeze(2)
        B, F_, C, H, W = x.shape
        assert F_ == self.num_frames, f'expected {self.num_frames} frames, got {F_}'
        if C == 1:
            x = x.expand(-1, -1, 3, -1, -1)
        x = x.reshape(B * F_, 3, H, W)

        feat = self.backbone(x)               # (B*F, 14, 14, 768)
        feat = feat.mean(dim=(1, 2))          # (B*F, 768)  spatial GAP
        feat = feat.reshape(B, F_, -1)        # (B, F, 768)

        pooled = self.frame_pool(feat)        # (B, 768)
        proj = self.proj(pooled)              # (B, proj_dim)
        proj = F.normalize(proj, dim=-1)
        return proj, pooled


# ----------------------------------------------------------------------------
# Sanity-check CLI
# ----------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse, os
    ap = argparse.ArgumentParser()
    ap.add_argument('--medsam_ckpt', type=str, default=None,
                    help='path to flaviagiammarino medsam pytorch_model.bin')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    model = CMREncoder(proj_dim=256, num_frames=4, medsam_ckpt=args.medsam_ckpt).to(args.device)
    model.eval()

    x = torch.randn(2, 4, 1, 224, 224, device=args.device)
    with torch.no_grad():
        proj, feat = model(x)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'[check] model={n_params/1e6:.1f}M params')
    print(f'[check] input shape:  {tuple(x.shape)}')
    print(f'[check] proj shape:   {tuple(proj.shape)}')
    print(f'[check] feat shape:   {tuple(feat.shape)}')
    print(f'[check] proj norm:    {proj.norm(dim=-1).mean().item():.4f}  (should be 1.0)')
    print('[check] OK')
