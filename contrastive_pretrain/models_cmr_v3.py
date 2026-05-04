"""
models_cmr_v3.py  —  CMR spatial-temporal encoder (v3, 16-frame)

Architecture
------------
Shared MedSAM ViT-B backbone processes all 16 frames independently.
Each frame: 224×224 → ViT-B → (14,14,768) → AdaptiveAvgPool2d(P) → P²×768 region tokens.
Each token carries three additive positional embeddings:
  region_emb  : P² learnable spatial positions (shared across all frames)
  view_emb    : 6-class embedding encoding the CMR view (LAX2Ch/4Ch, SAX-base/mid/apex, T1)
  time_emb    : 4-class embedding encoding the cardiac phase (ED/mid-sys/ES/static)
A learned [CLS] token is prepended; a 2-layer pre-norm Transformer attends over all
1 + 16×P² tokens. The CLS output drives task prediction and attention visualisation.

Why spatial tokens matter: without them (GAP → 1 token/frame), the model cannot compare
the SAME myocardial region across ED/mid-sys/ES, which is required to detect regional
wall-motion abnormalities (RWMA) — the primary imaging hallmark of ischaemia and MI.

Frame layout (16 total):
  [0]  LAX 2Ch   ED        view=0  time=0   anterior/inferior wall baseline
  [1]  LAX 2Ch   mid-sys   view=0  time=1   anterior/inferior systolic motion
  [2]  LAX 2Ch   ES        view=0  time=2   anterior/inferior end-systole
  [3]  LAX 4Ch   ED        view=1  time=0   biventricular + septal baseline
  [4]  LAX 4Ch   mid-sys   view=1  time=1   biventricular systolic motion
  [5]  LAX 4Ch   ES        view=1  time=2   biventricular end-systole
  [6]  SAX base  ED        view=2  time=0
  [7]  SAX base  mid-sys   view=2  time=1   basal wall motion
  [8]  SAX base  ES        view=2  time=2
  [9]  SAX mid   ED        view=3  time=0
  [10] SAX mid   mid-sys   view=3  time=1   mid-ventricular wall motion (most important)
  [11] SAX mid   ES        view=3  time=2
  [12] SAX apex  ED        view=4  time=0
  [13] SAX apex  mid-sys   view=4  time=1   apical wall motion
  [14] SAX apex  ES        view=4  time=2
  [15] T1 map    static    view=5  time=3   myocardial tissue characterisation

N_VIEWS=6, N_TIMES=4, default spatial_pool P=4 → 16 tokens/frame → 257 total tokens.
"""
from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from contrastive_pretrain.models_cmr import (
    SamVisionEncoderViTB,
    load_medsam_vision_encoder_weights,
)

try:
    from sam2.modeling.backbones.hieradet import Hiera as _Hiera
    _HIERA_AVAILABLE = True
except ImportError:
    _Hiera = None
    _HIERA_AVAILABLE = False

# Hiera backbone configs (from sam2.1 yaml files)
# out_channels = last-stage output channels (backbone_channel_list[0])
_HIERA_CONFIGS = {
    'sam2_tiny': dict(
        embed_dim=96, num_heads=1,
        stages=(1, 2, 7, 2), global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
        out_channels=768,
    ),
    'sam2_small': dict(
        embed_dim=96, num_heads=1,
        stages=(1, 2, 11, 2), global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
        out_channels=768,
    ),
    # MedSAM2 variants are Hiera Tiny fine-tuned on medical images
    'medsam2_latest': dict(
        embed_dim=96, num_heads=1,
        stages=(1, 2, 7, 2), global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
        out_channels=768,
    ),
    'medsam2_heart': dict(
        embed_dim=96, num_heads=1,
        stages=(1, 2, 7, 2), global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
        out_channels=768,
    ),
}


class _MedSAMBackbone(nn.Module):
    """MedSAM ViT-B wrapper — normalises output to (B, C, H, W)."""
    out_channels = 768

    def __init__(self, img_size: int, ckpt: Optional[str]):
        super().__init__()
        self.vit = SamVisionEncoderViTB(
            img_size=img_size, patch_size=16, in_chans=3,
            embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0,
        )
        if ckpt is not None:
            load_medsam_vision_encoder_weights(
                self.vit, ckpt, strict=True, verbose=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.vit(x)                  # (B, H, W, C)
        return feat.permute(0, 3, 1, 2)     # (B, C, H, W)


class _HieraBackbone(nn.Module):
    """SAM2 Hiera trunk wrapper — loads weights and returns (B, C, H, W) last stage."""

    def __init__(self, cfg_name: str, ckpt: Optional[str]):
        super().__init__()
        assert _HIERA_AVAILABLE, 'sam2 package not found; run setup_sam2.sh first'
        cfg = _HIERA_CONFIGS[cfg_name]
        self.out_channels = cfg['out_channels']
        self.trunk = _Hiera(
            embed_dim=cfg['embed_dim'],
            num_heads=cfg['num_heads'],
            stages=cfg['stages'],
            global_att_blocks=cfg['global_att_blocks'],
            window_pos_embed_bkg_spatial_size=cfg['window_pos_embed_bkg_spatial_size'],
            return_interm_layers=True,
        )
        if ckpt is not None:
            state = torch.load(ckpt, map_location='cpu')
            state = state.get('model', state)
            trunk_state = {
                k.replace('image_encoder.trunk.', ''): v
                for k, v in state.items()
                if k.startswith('image_encoder.trunk.')
            }
            missing, unexpected = self.trunk.load_state_dict(trunk_state, strict=False)
            if missing:
                print(f'  [backbone] {cfg_name}: {len(missing)} missing keys (ok if minor)')
            print(f'  [backbone] {cfg_name} loaded from {ckpt}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.trunk(x)   # list of (B, C, H, W) per stage
        return feats[-1]        # last stage — highest semantic, (B, C, H, W)


def build_backbone(name: str, ckpt: Optional[str], img_size: int = 224) -> nn.Module:
    """
    Factory: return a backbone module that takes (B, 3, H, W) and returns (B, C, H, W).

    Supported names: 'medsam', 'sam2_tiny', 'sam2_small',
                     'medsam2_latest', 'medsam2_heart'
    """
    if name == 'medsam':
        return _MedSAMBackbone(img_size=img_size, ckpt=ckpt)
    if name in _HIERA_CONFIGS:
        return _HieraBackbone(cfg_name=name, ckpt=ckpt)
    raise ValueError(f'Unknown backbone: {name!r}. '
                     f'Choose from: medsam, {", ".join(_HIERA_CONFIGS)}')


# ── Frame metadata ─────────────────────────────────────────────────────────────
FRAME_META: list[tuple[int, int]] = [
    (0, 0),  # [0]  LAX 2Ch   ED
    (0, 1),  # [1]  LAX 2Ch   mid-sys
    (0, 2),  # [2]  LAX 2Ch   ES
    (1, 0),  # [3]  LAX 4Ch   ED
    (1, 1),  # [4]  LAX 4Ch   mid-sys
    (1, 2),  # [5]  LAX 4Ch   ES
    (2, 0),  # [6]  SAX base  ED
    (2, 1),  # [7]  SAX base  mid-sys
    (2, 2),  # [8]  SAX base  ES
    (3, 0),  # [9]  SAX mid   ED
    (3, 1),  # [10] SAX mid   mid-sys
    (3, 2),  # [11] SAX mid   ES
    (4, 0),  # [12] SAX apex  ED
    (4, 1),  # [13] SAX apex  mid-sys
    (4, 2),  # [14] SAX apex  ES
    (5, 3),  # [15] T1 map    static
]
NUM_FRAMES  = 16
N_VIEWS     = 6   # LAX2Ch, LAX4Ch, SAX_base, SAX_mid, SAX_apex, T1
N_TIMES     = 4   # ED, mid-sys, ES, static

# Frame index ranges for cross-modal fusion
SAX_FRAME_RANGE = (6, 15)   # frames 6..14 inclusive (9 SAX frames)
T1_FRAME_IDX    = 15        # frame 15 = T1 map

# ED→ES delta pairs: (ed_frame_idx, es_frame_idx, view_id)
# One delta per cine view that has both ED and ES timepoints
DELTA_PAIRS: list[tuple[int, int, int]] = [
    (0,  2,  0),  # LAX 2Ch:   ED(0)  → ES(2)
    (3,  5,  1),  # LAX 4Ch:   ED(3)  → ES(5)
    (6,  8,  2),  # SAX base:  ED(6)  → ES(8)
    (9,  11, 3),  # SAX mid:   ED(9)  → ES(11)
    (12, 14, 4),  # SAX apex:  ED(12) → ES(14)
]
N_DELTA = len(DELTA_PAIRS)   # 5

# Human-readable labels (used in attention visualisation)
FRAME_LABELS = [
    'LAX-2Ch ED',   'LAX-2Ch mid',  'LAX-2Ch ES',
    'LAX-4Ch ED',   'LAX-4Ch mid',  'LAX-4Ch ES',
    'SAX-base ED',  'SAX-base mid', 'SAX-base ES',
    'SAX-mid ED',   'SAX-mid mid',  'SAX-mid ES',
    'SAX-apex ED',  'SAX-apex mid', 'SAX-apex ES',
    'T1 map',
]


# ── T1-Cine Bidirectional Cross-Attention Fusion ───────────────────────────────
class T1CineFusion(nn.Module):
    """
    Bidirectional cross-attention between T1 tissue tokens and SAX motion tokens.

    SAX tokens query T1  → each SAX region learns: "is this tissue fibrotic?"
    T1  tokens query SAX → each T1 region learns:  "does this tissue region move?"

    This fusion is spatial-alignment-free: attention handles approximate
    correspondence between T1 and cine SAX without hard-coded coordinate mapping.

    Args:
        dim       : token dimension (768)
        num_heads : attention heads (default 4, lightweight)
    """
    def __init__(self, dim: int = 768, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()
        # pre-norm for each side
        self.norm_sax = nn.LayerNorm(dim)
        self.norm_t1  = nn.LayerNorm(dim)
        # SAX → T1 cross-attention (SAX is query, T1 is key/value)
        self.sax_to_t1 = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        # T1 → SAX cross-attention (T1 is query, SAX is key/value)
        self.t1_to_sax = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.norm_out_sax = nn.LayerNorm(dim)
        self.norm_out_t1  = nn.LayerNorm(dim)

    def forward(
        self,
        sax_tokens: torch.Tensor,   # (B, N_sax, D)  — 9 frames × P² regions
        t1_tokens:  torch.Tensor,   # (B, P²,   D)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sax_n = self.norm_sax(sax_tokens)
        t1_n  = self.norm_t1(t1_tokens)

        # SAX gains tissue context from T1
        sax_cross, _ = self.sax_to_t1(sax_n, t1_n, t1_n, need_weights=False)
        sax_tokens = self.norm_out_sax(sax_tokens + sax_cross)

        # T1 gains motion context from SAX
        t1_cross, _  = self.t1_to_sax(t1_n, sax_n, sax_n, need_weights=False)
        t1_tokens = self.norm_out_t1(t1_tokens + t1_cross)

        return sax_tokens, t1_tokens


# ── Temporal Transformer ───────────────────────────────────────────────────────
class TemporalTransformer(nn.Module):
    """Pre-norm Transformer encoder. Input/output: (B, N, D)."""
    def __init__(self, dim: int = 768, num_heads: int = 8, depth: int = 2,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation='gelu',
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.transformer(x)


# ── CMREncoderV3 ───────────────────────────────────────────────────────────────
class CMREncoderV3(nn.Module):
    """
    16-frame spatial-temporal CMR encoder.

    forward(x)            → (proj_emb, raw_emb)
    forward_with_attn(x)  → (proj_emb, raw_emb, attn_map)
      attn_map: (B, NUM_FRAMES, P²) — CLS attention to each frame-region token,
                suitable for overlaying on original CMR images.
    """

    def __init__(
        self,
        proj_dim: int = 256,
        embed_dim: int = 768,
        img_size: int = 224,
        spatial_pool: int = 4,
        transformer_heads: int = 8,
        transformer_depth: int = 2,
        transformer_dropout: float = 0.1,
        attn_dropout: float = 0.0,
        backbone: str = 'medsam',
        backbone_ckpt: Optional[str] = None,
        freeze_backbone: bool = False,
        grad_checkpoint: bool = False,
        # legacy arg — kept for backward compat, ignored if backbone != 'medsam'
        medsam_ckpt: Optional[str] = None,
    ):
        super().__init__()
        self.num_frames      = NUM_FRAMES
        self.embed_dim       = embed_dim
        self.spatial_pool    = spatial_pool
        self.grad_checkpoint = grad_checkpoint
        P2 = spatial_pool * spatial_pool

        # resolve ckpt: backbone_ckpt takes priority; fall back to medsam_ckpt for medsam
        ckpt = backbone_ckpt if backbone_ckpt is not None else (
            medsam_ckpt if backbone == 'medsam' else None)

        # ── shared backbone (unified BCHW output) ──
        self.backbone = build_backbone(backbone, ckpt, img_size=img_size)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)

        # ── spatial adaptive pool (14×14 → P×P) ──
        self.spatial_adapt = nn.AdaptiveAvgPool2d(spatial_pool)

        # ── positional embeddings ──
        self.region_emb = nn.Parameter(torch.zeros(1, P2, embed_dim))
        self.view_emb   = nn.Embedding(N_VIEWS, embed_dim)
        self.time_emb   = nn.Embedding(N_TIMES, embed_dim)
        nn.init.trunc_normal_(self.region_emb, std=0.02)

        view_ids = torch.tensor([m[0] for m in FRAME_META], dtype=torch.long)
        time_ids = torch.tensor([m[1] for m in FRAME_META], dtype=torch.long)
        self.register_buffer('view_ids', view_ids)
        self.register_buffer('time_ids', time_ids)

        # ── CLS token ──
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── T1-Cine fusion (before main transformer) ──
        self.t1_cine_fusion = T1CineFusion(dim=embed_dim, num_heads=4,
                                           dropout=attn_dropout)

        # ── ED→ES difference tokens ──
        # One learnable marker shared by all delta tokens (identifies "I am a Δ token")
        self.delta_marker = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.delta_marker, std=0.02)
        # register delta pair info as buffers
        delta_ed  = torch.tensor([p[0] for p in DELTA_PAIRS], dtype=torch.long)
        delta_es  = torch.tensor([p[1] for p in DELTA_PAIRS], dtype=torch.long)
        delta_vid = torch.tensor([p[2] for p in DELTA_PAIRS], dtype=torch.long)
        self.register_buffer('delta_ed',  delta_ed)
        self.register_buffer('delta_es',  delta_es)
        self.register_buffer('delta_vid', delta_vid)

        # ── spatial-temporal transformer ──
        # total tokens: 1(CLS) + 16×P²(frames) + 5×P²(deltas) = 1+256+80=337 (default)
        self.st_norm = nn.LayerNorm(embed_dim)
        self.st_tf   = TemporalTransformer(
            dim=embed_dim, num_heads=transformer_heads,
            depth=transformer_depth, dropout=transformer_dropout,
        )

        # ── projection head ──
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, proj_dim),
        )
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.view_emb.weight, std=0.02)
        nn.init.trunc_normal_(self.time_emb.weight, std=0.02)
        for m in self.proj.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── shared tokenisation logic ──────────────────────────────────────────────
    def _tokenise(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, 1or3, H, W)
        Returns tokens: (B, 1 + F×P², D)  — CLS prepended, pre-transformer

        Pipeline:
          1. Backbone → spatial pool → region tokens per frame
          2. Add 3-way positional embeddings (region + view + time)
          3. T1-Cine bidirectional cross-attention fusion
             (SAX tokens ↔ T1 tokens before main Transformer)
          4. Flatten all frames, prepend CLS, apply pre-norm
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)
        B, F_, C, H, W = x.shape
        assert F_ == NUM_FRAMES, f'expected {NUM_FRAMES} frames, got {F_}'
        if C == 1:
            x = x.expand(-1, -1, 3, -1, -1)

        x_flat = x.reshape(B * F_, 3, H, W)
        if self.grad_checkpoint and x_flat.requires_grad:
            feat = checkpoint(self.backbone, x_flat, use_reentrant=False)
        else:
            feat = self.backbone(x_flat)        # (B*F, C, H, W) — all backbones normalised

        # spatial pool → region tokens
        P = self.spatial_pool
        feat = self.spatial_adapt(feat)                              # (B*F, C, P, P)
        feat = feat.permute(0, 2, 3, 1).reshape(B * F_, P * P, -1) # (B*F, P², C)

        # positional embeddings: region + view + time
        feat = feat + self.region_emb
        feat = feat.reshape(B, F_, P * P, -1)
        feat = (feat
                + self.view_emb(self.view_ids)[None, :, None, :]
                + self.time_emb(self.time_ids)[None, :, None, :])
        # feat: (B, 16, P², 768)

        # ── T1-Cine bidirectional cross-attention fusion ──────────────────────
        # SAX frames: indices 6..14 (9 frames × P² = 144 tokens)
        s0, s1 = SAX_FRAME_RANGE            # 6, 15
        sax = feat[:, s0:s1, :, :].reshape(B, (s1 - s0) * P * P, -1)  # (B, 144, 768)
        t1  = feat[:, T1_FRAME_IDX, :, :]                               # (B, 16,  768)

        sax_fused, t1_fused = self.t1_cine_fusion(sax, t1)

        # reassemble without in-place ops to preserve autograd graph
        lax_tokens = feat[:, :s0, :, :]                             # (B, 6, P², D) LAX
        sax_tokens = sax_fused.reshape(B, s1 - s0, P * P, -1)      # (B, 9, P², D) SAX fused
        t1_tokens  = t1_fused.unsqueeze(1)                          # (B, 1, P², D) T1  fused
        feat = torch.cat([lax_tokens, sax_tokens, t1_tokens], dim=1)  # (B, 16, P², D)
        # ─────────────────────────────────────────────────────────────────────

        feat_flat = feat.reshape(B, F_ * P * P, -1)            # (B, F×P², 768)

        # ── build ED→ES difference tokens ────────────────────────────────────
        # feat_raw (before positional encoding) is more informative for deltas,
        # but here we use feat (after pos enc) for simplicity; the region+view
        # embeddings cancel in the subtraction, leaving clean feature diff +
        # (time_emb[ES] - time_emb[ED]) which the model can use as extra signal.
        delta_parts = []
        for i in range(N_DELTA):
            ed_idx = self.delta_ed[i].item()
            es_idx = self.delta_es[i].item()
            v_id   = self.delta_vid[i]
            # ES features minus ED features: (B, P², D)
            delta = feat[:, es_idx, :, :] - feat[:, ed_idx, :, :]
            # add spatial + view identity + delta marker
            delta = (delta
                     + self.region_emb                              # (1, P², D) spatial position
                     + self.view_emb(v_id)                          # (D,) view identity
                     + self.delta_marker)                           # (1, 1, D) Δ-token marker
            delta_parts.append(delta)   # (B, P², D)
        # stack → (B, N_DELTA×P², D)
        delta_tokens = torch.cat(delta_parts, dim=1)

        # concatenate: [frame tokens | delta tokens]
        all_tokens = torch.cat([feat_flat, delta_tokens], dim=1)   # (B, (F+N_DELTA)×P², D)

        cls = self.cls_token.expand(B, -1, -1)
        return self.st_norm(torch.cat([cls, all_tokens], dim=1))    # (B, 1+(F+N_DELTA)×P², D)

    # ── standard forward ───────────────────────────────────────────────────────
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens  = self._tokenise(x)
        tokens  = self.st_tf(tokens)
        cls_out = tokens[:, 0]
        proj    = F.normalize(self.proj(cls_out), dim=-1)
        return proj, cls_out

    # ── forward with attention extraction ─────────────────────────────────────
    @torch.no_grad()
    def forward_with_attn(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as forward but also returns the CLS attention map from the last
        Transformer layer, reshaped as (B, NUM_FRAMES, P²).

        Usage (inference / visualisation only):
            model.eval()
            proj, feat, attn = model.forward_with_attn(cmr_batch)
            # attn[b, f, :] → attention weights for frame f, P² spatial regions
            # reshape attn[b, f, :] to (P, P) and upsample to 224×224 for overlay
        """
        tokens = self._tokenise(x)
        layers = self.st_tf.transformer.layers

        # run all layers except the last
        for layer in layers[:-1]:
            tokens = layer(tokens)

        # last layer: extract CLS attention weights explicitly
        last   = layers[-1]
        normed = last.norm1(tokens)                        # pre-norm
        attn_out, attn_w = last.self_attn(
            normed, normed, normed,
            need_weights=True, average_attn_weights=True,  # (B, N, N)
        )
        tokens = tokens + last.dropout1(attn_out)
        tokens = tokens + last._ff_block(last.norm2(tokens))

        if self.st_tf.transformer.norm is not None:
            tokens = self.st_tf.transformer.norm(tokens)

        cls_out = tokens[:, 0]
        proj    = F.normalize(self.proj(cls_out), dim=-1)

        # CLS attention over all tokens: frame tokens + delta tokens
        # Return separately so caller can visualise both
        P2 = self.spatial_pool ** 2
        B  = x.shape[0]
        all_attn    = attn_w[:, 0, 1:]                           # (B, (F+N_DELTA)×P²)
        frame_attn  = all_attn[:, :NUM_FRAMES * P2].reshape(B, NUM_FRAMES, P2)
        delta_attn  = all_attn[:, NUM_FRAMES * P2:].reshape(B, N_DELTA, P2)
        return proj, cls_out, frame_attn, delta_attn

    # ── attention → spatial heatmap helper ────────────────────────────────────
    @staticmethod
    def attn_to_heatmap(
        attn: torch.Tensor,
        spatial_pool: int = 4,
        out_size: int = 224,
    ) -> torch.Tensor:
        """
        Upsample CLS attention weights to full image resolution.

        Args:
            attn        : (B, K, P²) — K = NUM_FRAMES or N_DELTA
            spatial_pool: P (same as model.spatial_pool)
            out_size    : target resolution (default 224)
        Returns:
            heatmap     : (B, K, out_size, out_size) float32 in [0, 1]
        """
        B, K, P2 = attn.shape
        P  = spatial_pool
        hm = attn.reshape(B * K, 1, P, P).float()
        hm = F.interpolate(hm, size=(out_size, out_size), mode='bilinear',
                           align_corners=False)
        hm = hm.reshape(B, K, out_size, out_size)
        mn = hm.flatten(2).min(dim=-1).values[:, :, None, None]
        mx = hm.flatten(2).max(dim=-1).values[:, :, None, None]
        return (hm - mn) / (mx - mn + 1e-8)


# ── sanity check ──────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--medsam_ckpt', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    m = CMREncoderV3(spatial_pool=4, medsam_ckpt=args.medsam_ckpt).to(args.device)
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f'params: {n/1e6:.1f}M  |  tokens: 1 + {NUM_FRAMES}×{4*4} = {1+NUM_FRAMES*16}')

    x = torch.randn(2, NUM_FRAMES, 1, 224, 224, device=args.device)
    with torch.cuda.amp.autocast():
        proj, feat = m(x)
    print(f'forward  proj:{proj.shape} feat:{feat.shape}  norm={proj.norm(dim=-1).mean():.4f}')

    proj2, feat2, attn = m.forward_with_attn(x)
    hm = CMREncoderV3.attn_to_heatmap(attn, spatial_pool=4)
    print(f'attn     map:{attn.shape}  heatmap:{hm.shape}  range=[{hm.min():.3f},{hm.max():.3f}]')
    print('OK')
