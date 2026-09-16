"""
models_cmr_v4_hierarchical.py — hierarchical tri-modal CMR encoder (16-frame)

Architecture
------------
Shared Hiera-tiny backbone processes all 16 frames independently.
Each frame: 224×224 → Hiera → spatial feature map → AdaptiveAvgPool2d(P)
→ P²×768 region tokens.
Each token carries three additive positional embeddings:
  region_emb  : P² learnable spatial positions (shared across all frames)
  view_emb    : 6-class embedding encoding the CMR view (LAX2Ch/4Ch, SAX-base/mid/apex, T1)
  time_emb    : 4-class embedding encoding the cardiac phase (ED/mid-sys/ES/static)
Before global aggregation, explicit bidirectional cross-attention first exchanges
information between LAX and SAX cine tokens, then exchanges information between the
fused cine tokens and native-T1 tokens. A learned [CLS] token is prepended and a
pre-norm Transformer aggregates frame, T1 and temporal-difference tokens. The CLS
output is the 768-dimensional patient representation.

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

N_VIEWS=6, N_TIMES=4, default spatial_pool P=4 → 16 tokens/frame.
The global sequence has 337 tokens: CLS + 256 frame tokens + 80 delta tokens.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

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
            if not Path(ckpt).is_file():
                raise FileNotFoundError(f'Backbone checkpoint not found: {ckpt}')
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

    Supported names: ``sam2_tiny``, ``sam2_small``, ``medsam2_latest`` and
    ``medsam2_heart``. The names select compatible Hiera configurations; the
    actual initialization is determined by ``ckpt``.
    """
    if name in _HIERA_CONFIGS:
        return _HieraBackbone(cfg_name=name, ckpt=ckpt)
    raise ValueError(f'Unknown backbone: {name!r}. '
                     f'Choose from: {", ".join(_HIERA_CONFIGS)}')


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

# Frame index ranges for hierarchical tri-modal fusion
LAX_FRAME_RANGE = (0, 6)    # frames 0..5 inclusive (6 LAX frames)
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


class BidirectionalCrossAttention(nn.Module):
    """Update two token streams with two explicit cross-attention calls.

    The calls are evaluated from the same pre-attention representations:

      A' = A + Attention(Q=A, K=B, V=B)
      B' = B + Attention(Q=B, K=A, V=A)

    Therefore this module is genuinely bidirectional; it is not self-attention over
    concatenated tokens and it does not infer the reverse direction implicitly.
    """

    def __init__(self, dim: int = 768, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        self.a_queries_b = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.b_queries_a = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout_a = nn.Dropout(dropout)
        self.dropout_b = nn.Dropout(dropout)
        self.out_norm_a = nn.LayerNorm(dim)
        self.out_norm_b = nn.LayerNorm(dim)

    def forward(
        self,
        stream_a: torch.Tensor,
        stream_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        a_norm = self.norm_a(stream_a)
        b_norm = self.norm_b(stream_b)

        # stream A <- stream B: Q=A, K=B, V=B
        a_context, _ = self.a_queries_b(
            a_norm, b_norm, b_norm, need_weights=False)
        # stream B <- stream A: Q=B, K=A, V=A
        b_context, _ = self.b_queries_a(
            b_norm, a_norm, a_norm, need_weights=False)

        stream_a = self.out_norm_a(stream_a + self.dropout_a(a_context))
        stream_b = self.out_norm_b(stream_b + self.dropout_b(b_context))
        return stream_a, stream_b


class HierarchicalCineT1Fusion(nn.Module):
    """LAX<->SAX followed by fused-cine<->T1 bidirectional fusion.

    Shapes for the default P=4 configuration:
      LAX: 6 * P² = 96 tokens
      SAX: 9 * P² = 144 tokens
      T1 : 1 * P² = 16 tokens
      fused cine: 240 tokens

    The token dimension remains ``dim`` because LAX and SAX are concatenated along
    the sequence axis, not the channel axis.
    """

    def __init__(self, dim: int = 768, num_heads: int = 4,
                 dropout: float = 0.0):
        super().__init__()
        self.lax_sax = BidirectionalCrossAttention(
            dim=dim, num_heads=num_heads, dropout=dropout)
        self.cine_merge_norm = nn.LayerNorm(dim)
        self.cine_t1 = BidirectionalCrossAttention(
            dim=dim, num_heads=num_heads, dropout=dropout)

    def forward(
        self,
        lax_tokens: torch.Tensor,
        sax_tokens: torch.Tensor,
        t1_tokens: torch.Tensor,
        t1_available: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # LAX <- SAX and SAX <- LAX.
        lax_updated, sax_updated = self.lax_sax(lax_tokens, sax_tokens)

        n_lax = lax_updated.shape[1]
        cine = self.cine_merge_norm(
            torch.cat([lax_updated, sax_updated], dim=1))

        # Cine <- T1 and T1 <- Cine.
        if t1_available is None:
            cine_updated, t1_updated = self.cine_t1(cine, t1_tokens)
        else:
            observed = t1_available.to(dtype=torch.bool, device=cine.device)
            observed_indices = observed.nonzero(as_tuple=False).flatten()
            # Missing-T1 subjects never enter cine/T1 attention. This is a true
            # routed subset, not an attention result discarded after the call.
            cine_updated = cine
            t1_updated = torch.zeros_like(t1_tokens)
            if observed_indices.numel():
                observed_cine, observed_t1 = self.cine_t1(
                    cine.index_select(0, observed_indices),
                    t1_tokens.index_select(0, observed_indices))
                cine_updated = cine_updated.index_copy(
                    0, observed_indices, observed_cine)
                t1_updated = t1_updated.index_copy(
                    0, observed_indices, observed_t1)
        lax_final = cine_updated[:, :n_lax, :]
        sax_final = cine_updated[:, n_lax:, :]
        return lax_final, sax_final, t1_updated


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

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return self.transformer(
            x, src_key_padding_mask=src_key_padding_mask)


# ── CMREncoderV3 ───────────────────────────────────────────────────────────────
class CMREncoderV4(nn.Module):
    """
    16-frame hierarchical tri-modal CMR encoder.

    ``fusion_mode`` defines controlled architecture ablations:
      - ``hierarchical``: LAX<->SAX, then fused-cine<->T1 (primary model)
      - ``sax_t1``: legacy SAX<->T1 local fusion
      - ``global_only``: no explicit local cross-attention

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
        cross_attn_heads: int = 4,
        fusion_mode: str = 'hierarchical',
        backbone: str = 'medsam2_heart',
        backbone_ckpt: Optional[str] = None,
        freeze_backbone: bool = False,
        grad_checkpoint: bool = False,
    ):
        super().__init__()
        self.num_frames      = NUM_FRAMES
        self.embed_dim       = embed_dim
        self.spatial_pool    = spatial_pool
        self.grad_checkpoint = grad_checkpoint
        self.fusion_mode     = fusion_mode
        P2 = spatial_pool * spatial_pool

        valid_fusion_modes = {'hierarchical', 'sax_t1', 'global_only'}
        if fusion_mode not in valid_fusion_modes:
            raise ValueError(
                f'Unknown fusion_mode={fusion_mode!r}; choose from '
                f'{sorted(valid_fusion_modes)}')

        # ── shared backbone (unified BCHW output) ──
        self.backbone = build_backbone(
            backbone, backbone_ckpt, img_size=img_size)
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

        # ── explicit local cross-attention (before global transformer) ──
        if fusion_mode == 'hierarchical':
            self.hierarchical_fusion = HierarchicalCineT1Fusion(
                dim=embed_dim, num_heads=cross_attn_heads,
                dropout=attn_dropout)
        elif fusion_mode == 'sax_t1':
            self.t1_cine_fusion = T1CineFusion(
                dim=embed_dim, num_heads=cross_attn_heads,
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
    def _tokenise(
        self,
        x: torch.Tensor,
        t1_available: Optional[torch.Tensor] = None,
        return_key_padding_mask: bool = False,
    ):
        """
        x: (B, F, 1or3, H, W)
        Returns tokens: (B, 1 + F×P², D)  — CLS prepended, pre-transformer

        Pipeline:
          1. Backbone → spatial pool → region tokens per frame
          2. Add 3-way positional embeddings (region + view + time)
          3. Apply the selected local fusion mode. The primary mode performs
             LAX<->SAX followed by fused-cine<->T1 cross-attention.
          4. Flatten all frames, prepend CLS, apply pre-norm
        """
        if x.dim() == 4:
            x = x.unsqueeze(2)
        B, F_, C, H, W = x.shape
        assert F_ == NUM_FRAMES, f'expected {NUM_FRAMES} frames, got {F_}'
        if t1_available is None:
            # Backward-compatible fallback for legacy arrays: detect whether the
            # T1 slot carries observed image signal before positional embeddings
            # are added. Published training code should pass the manifest mask.
            t1_available = x[:, T1_FRAME_IDX].detach().abs().flatten(1).amax(dim=1) > 1e-8
        else:
            t1_available = torch.as_tensor(
                t1_available, dtype=torch.bool, device=x.device).reshape(B)
        if C == 1:
            x = x.expand(-1, -1, 3, -1, -1)

        P = self.spatial_pool
        def encode_images(images: torch.Tensor) -> torch.Tensor:
            if self.grad_checkpoint and self.training:
                encoded = checkpoint(
                    self.backbone, images, use_reentrant=False)
            else:
                encoded = self.backbone(images)
            encoded = self.spatial_adapt(encoded)
            return encoded.permute(0, 2, 3, 1).reshape(
                images.shape[0], P * P, -1)

        # All cine frames are always observed and encoded. T1 is routed through
        # the backbone only for participants with an observed T1 map.
        cine_flat = x[:, :T1_FRAME_IDX].reshape(
            B * T1_FRAME_IDX, 3, H, W)
        cine_tokens = encode_images(cine_flat).reshape(
            B, T1_FRAME_IDX, P * P, -1)
        observed_indices = t1_available.nonzero(as_tuple=False).flatten()
        t1_tokens_raw = cine_tokens.new_zeros(
            B, P * P, cine_tokens.shape[-1])
        if observed_indices.numel():
            observed_t1_images = x.index_select(
                0, observed_indices)[:, T1_FRAME_IDX]
            observed_t1_tokens = encode_images(observed_t1_images)
            t1_tokens_raw = t1_tokens_raw.index_copy(
                0, observed_indices, observed_t1_tokens)
        feat = torch.cat([cine_tokens, t1_tokens_raw[:, None]], dim=1)

        # positional embeddings: region + view + time
        feat = (feat + self.region_emb
                + self.view_emb(self.view_ids)[None, :, None, :]
                + self.time_emb(self.time_ids)[None, :, None, :])
        # feat: (B, 16, P², 768)

        # ── local tri-modal fusion ────────────────────────────────────────────
        l0, l1 = LAX_FRAME_RANGE
        s0, s1 = SAX_FRAME_RANGE
        lax = feat[:, l0:l1, :, :].reshape(B, (l1 - l0) * P * P, -1)
        sax = feat[:, s0:s1, :, :].reshape(B, (s1 - s0) * P * P, -1)
        t1  = feat[:, T1_FRAME_IDX, :, :]

        if self.fusion_mode == 'hierarchical':
            lax, sax, t1 = self.hierarchical_fusion(
                lax, sax, t1, t1_available=t1_available)
        elif self.fusion_mode == 'sax_t1':
            observed_indices = t1_available.nonzero(as_tuple=False).flatten()
            t1_output = torch.zeros_like(t1)
            if observed_indices.numel():
                sax_fused, t1_fused = self.t1_cine_fusion(
                    sax.index_select(0, observed_indices),
                    t1.index_select(0, observed_indices))
                sax = sax.index_copy(0, observed_indices, sax_fused)
                t1_output = t1_output.index_copy(
                    0, observed_indices, t1_fused)
            t1 = t1_output
        else:
            # global_only has no local fusion, but missing T1 must still be
            # removed rather than converted into positional-embedding tokens.
            t1 = torch.where(
                t1_available[:, None, None], t1, torch.zeros_like(t1))

        # Reassemble without in-place operations to preserve autograd.
        lax_frames = lax.reshape(B, l1 - l0, P * P, -1)
        sax_frames = sax.reshape(B, s1 - s0, P * P, -1)
        t1_frame   = t1.unsqueeze(1)
        feat = torch.cat([lax_frames, sax_frames, t1_frame], dim=1)

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
        tokens = self.st_norm(torch.cat([cls, all_tokens], dim=1))

        # True entries are ignored as keys/values by the global Transformer.
        # Only the P² native-T1 positions are masked; all cine and delta tokens
        # remain available. Shape: (B, 1+(F+N_DELTA)*P²).
        key_padding_mask = torch.zeros(
            B, tokens.shape[1], dtype=torch.bool, device=tokens.device)
        t1_start = 1 + T1_FRAME_IDX * P * P
        t1_end = t1_start + P * P
        key_padding_mask[:, t1_start:t1_end] = ~t1_available[:, None]

        if return_key_padding_mask:
            return tokens, key_padding_mask
        return tokens

    # ── standard forward ───────────────────────────────────────────────────────
    def forward(
        self,
        x: torch.Tensor,
        t1_available: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        tokens, key_padding_mask = self._tokenise(
            x, t1_available=t1_available, return_key_padding_mask=True)
        tokens  = self.st_tf(tokens, src_key_padding_mask=key_padding_mask)
        cls_out = tokens[:, 0]
        proj    = F.normalize(self.proj(cls_out), dim=-1)
        return proj, cls_out

    # ── forward with attention extraction ─────────────────────────────────────
    @torch.no_grad()
    def forward_with_attn(
        self,
        x: torch.Tensor,
        t1_available: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Same as forward but also returns the CLS attention map from the last
        Transformer layer, reshaped as (B, NUM_FRAMES, P²).

        Usage (inference / visualisation only):
            model.eval()
            proj, feat, frame_attn, delta_attn = model.forward_with_attn(cmr_batch)
            # attn[b, f, :] → attention weights for frame f, P² spatial regions
            # reshape attn[b, f, :] to (P, P) and upsample to 224×224 for overlay
        """
        tokens, key_padding_mask = self._tokenise(
            x, t1_available=t1_available, return_key_padding_mask=True)
        layers = self.st_tf.transformer.layers

        # run all layers except the last
        for layer in layers[:-1]:
            tokens = layer(tokens, src_key_padding_mask=key_padding_mask)

        # last layer: extract CLS attention weights explicitly
        last   = layers[-1]
        normed = last.norm1(tokens)                        # pre-norm
        attn_out, attn_w = last.self_attn(
            normed, normed, normed,
            key_padding_mask=key_padding_mask,
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
    ap.add_argument('--backbone_ckpt', default=None)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    m = CMREncoderV4(
        spatial_pool=4,
        fusion_mode='hierarchical',
        backbone_ckpt=args.backbone_ckpt,
    ).to(args.device)
    m.eval()
    n = sum(p.numel() for p in m.parameters())
    print(f'params: {n/1e6:.1f}M  |  tokens: 1 + {NUM_FRAMES}×{4*4} = {1+NUM_FRAMES*16}')

    x = torch.randn(2, NUM_FRAMES, 1, 224, 224, device=args.device)
    with torch.cuda.amp.autocast():
        proj, feat = m(x)
    print(f'forward  proj:{proj.shape} feat:{feat.shape}  norm={proj.norm(dim=-1).mean():.4f}')

    proj2, feat2, frame_attn, delta_attn = m.forward_with_attn(x)
    hm = CMREncoderV4.attn_to_heatmap(frame_attn, spatial_pool=4)
    print(f'attn     frame:{frame_attn.shape} delta:{delta_attn.shape} '
          f'heatmap:{hm.shape} range=[{hm.min():.3f},{hm.max():.3f}]')
    print('OK')


# Compatibility alias for training utilities that still import CMREncoderV3.
CMREncoderV3 = CMREncoderV4
