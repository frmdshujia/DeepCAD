"""
models_crossattn.py

Stage 1 architecture: Enhanced CMR encoder via CrossAttention from frozen fundus.
Stage 2 architecture: Frozen anchor + trainable fundus encoder with RKD loss.

Stage 1 forward:
  frozen RETFound → fundus_tokens [B, N, 1024]
  CMREncoder (trainable) → z_c [B, 768]
  CrossAttention(q=z_c, kv=fundus_tokens) → attn_out [B, 768]
  z_c_enhanced = z_c + attn_out

Stage 2 forward:
  frozen Stage1 anchor (CMREncoder + CrossAttn) → z_c_enhanced [B, 768]
  trainable FundusEncoder_NEW → z_f [B, 1024]
  proj_head_f (trainable) → p_f [B, 256]  L2-normalised
  proj_head_c (trainable) → p_c [B, 256]  L2-normalised  (maps anchor to align space)
  RKD loss: KL(softmax(sim_f/T) || softmax(sim_c/T))
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.models_dualtower import FundusEncoder, TaskHead
from contrastive_pretrain.models_cmr import CMREncoder

HERE = pathlib.Path(__file__).parent

RETFOUND_CKPT = str(ROOT / 'RETFound_cfp_weights.pth')
MEDSAM_CKPT   = str(ROOT / 'pretrained_weights/hf_cache/'
                    'models--flaviagiammarino--medsam-vit-base/blobs/'
                    'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c')

CLS_COLS = ['composite_ischemic_hd', 'prevalent_I21']
REG_COLS = ['LV ejection fraction', 'LV end diastolic volume', 'LV end systolic volume']


# ── FundusEncoder with patch-token output ─────────────────────────────────────
def get_fundus_patch_tokens(fundus_encoder: FundusEncoder,
                            x: torch.Tensor) -> torch.Tensor:
    """
    Run RETFound ViT-L up to (but not including) global pooling.
    Returns patch tokens [B, 196, 1024] for 224x224 input with 16x16 patches.
    The fundus_encoder must be in eval() + no_grad() context for frozen use.
    """
    vit = fundus_encoder.vit
    B   = x.shape[0]

    x = vit.patch_embed(x)
    cls_tokens = vit.cls_token.expand(B, -1, -1)
    x = torch.cat((cls_tokens, x), dim=1)
    x = x + vit.pos_embed
    x = vit.pos_drop(x)

    for blk in vit.blocks:
        x = blk(x)

    # return patch tokens only, skip cls token; shape [B, 196, 1024]
    return x[:, 1:, :]


# ── CrossAttention ─────────────────────────────────────────────────────────────
class CrossAttention(nn.Module):
    """
    Single-head or multi-head cross-attention.
    query: CMR pooled token [B, query_dim]
    kv:    fundus patch tokens [B, N, kv_dim]
    output: [B, query_dim]

    out_proj zero-initialised so cross-attn starts as identity (z_c + 0).
    """

    def __init__(self, query_dim: int = 768, kv_dim: int = 1024,
                 n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert query_dim % n_heads == 0
        self.n_heads = n_heads
        self.scale   = (query_dim // n_heads) ** -0.5

        self.q_proj   = nn.Linear(query_dim, query_dim, bias=False)
        self.k_proj   = nn.Linear(kv_dim,    query_dim, bias=False)
        self.v_proj   = nn.Linear(kv_dim,    query_dim, bias=False)
        self.out_proj = nn.Linear(query_dim,  query_dim)
        self.dropout  = nn.Dropout(dropout)

        # zero-init: cross-attn outputs zero at training start → pure CMR signal
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, query: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """
        query: [B, query_dim]  (CMR pooled)
        kv:    [B, N, kv_dim]  (fundus patch tokens)
        returns: [B, query_dim]
        """
        B, D = query.shape
        H    = self.n_heads
        dh   = D // H

        q = self.q_proj(query).view(B, 1, H, dh).transpose(1, 2)   # [B, H, 1, dh]
        k = self.k_proj(kv).view(B, -1, H, dh).transpose(1, 2)     # [B, H, N, dh]
        v = self.v_proj(kv).view(B, -1, H, dh).transpose(1, 2)     # [B, H, N, dh]

        attn = (q @ k.transpose(-2, -1)) * self.scale               # [B, H, 1, N]
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, 1, D)           # [B, 1, D]
        return self.out_proj(out).squeeze(1)                         # [B, D]


# ── Stage 1 Model ─────────────────────────────────────────────────────────────
class Stage1EnhancedCMR(nn.Module):
    """
    Frozen fundus encoder provides patch tokens as K,V for CMR cross-attention.
    Only CMREncoder, CrossAttention, and TaskHead_c are trainable.

    z_c_enhanced = z_c + CrossAttn(q=z_c, kv=fundus_tokens)
    """

    def __init__(
        self,
        retfound_ckpt: str = RETFOUND_CKPT,
        medsam_ckpt:   str = MEDSAM_CKPT,
        n_cls: int = 2,
        n_reg: int = 3,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        # frozen fundus encoder (only used to provide patch tokens)
        self.fundus_enc = FundusEncoder(ckpt=retfound_ckpt, freeze=True,
                                        grad_ckpt=False)
        # trainable CMR encoder
        self.cmr_enc    = CMREncoder(proj_dim=256, num_frames=4, embed_dim=768,
                                     medsam_ckpt=medsam_ckpt, freeze_backbone=False)
        # trainable cross-attention
        self.cross_attn = CrossAttention(query_dim=768, kv_dim=1024, n_heads=8)
        # trainable task head
        self.task_head  = TaskHead(in_dim=768, n_cls=n_cls, n_reg=n_reg)

    def forward(self, fundus: torch.Tensor, cmr: torch.Tensor) -> dict:
        """
        fundus: [B, 3, 224, 224]
        cmr:    [B, 4, 1, 224, 224]
        returns dict: z_c, z_c_enhanced, cls_c, reg_c
        """
        with torch.no_grad():
            fundus_tokens = get_fundus_patch_tokens(self.fundus_enc, fundus)
            # cast to model dtype (fundus enc is frozen in fp32, need fp16 under autocast)
            fundus_tokens = fundus_tokens.to(dtype=next(self.cmr_enc.parameters()).dtype)

        _, z_c = self.cmr_enc(cmr)                                   # [B, 768]
        attn_out = self.cross_attn(z_c, fundus_tokens)               # [B, 768]
        z_c_enhanced = z_c + attn_out                                # [B, 768]

        cls_c, reg_c = self.task_head(z_c_enhanced)
        return {'z_c': z_c, 'z_c_enhanced': z_c_enhanced,
                'cls_c': cls_c, 'reg_c': reg_c}

    def trainable_parameters(self):
        """Only CMREncoder + CrossAttention + TaskHead."""
        return (list(self.cmr_enc.parameters()) +
                list(self.cross_attn.parameters()) +
                list(self.task_head.parameters()))


# ── Stage 2 Model ─────────────────────────────────────────────────────────────
class ProjHead(nn.Module):
    """Linear projection + L2 normalisation."""
    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.fc(z.float()), dim=-1, eps=1e-8)


class Stage2RKDModel(nn.Module):
    """
    Frozen: fundus_enc_frozen (from Stage1), cmr_enc + cross_attn (from Stage1)
    Trainable: fundus_enc_new (RETFound init), proj_head_f, proj_head_c,
               task_head_f (used only if alpha > 0)

    forward returns:
        z_f:   new fundus embedding [B, 1024]
        p_f:   L2-norm projection of z_f [B, 256]
        p_c:   L2-norm projection of z_c_enhanced [B, 256]  (detached in loss)
        cls_f, reg_f: task head outputs (empty lists if n_reg/n_cls=0)
    """

    def __init__(
        self,
        retfound_ckpt:   str = RETFOUND_CKPT,
        stage1_ckpt:     str | None = None,
        medsam_ckpt:     str = MEDSAM_CKPT,
        n_cls: int = 2,
        n_reg: int = 0,     # fundus CSV has no regression targets
        proj_dim: int = 256,
        grad_ckpt: bool = True,
    ):
        super().__init__()

        # ── Frozen anchor ─────────────────────────────────────────────────────
        self.fundus_enc_frozen = FundusEncoder(ckpt=retfound_ckpt, freeze=True,
                                               grad_ckpt=False)
        self.cmr_enc_frozen    = CMREncoder(proj_dim=proj_dim, num_frames=4,
                                            embed_dim=768, medsam_ckpt=medsam_ckpt,
                                            freeze_backbone=False)
        self.cross_attn_frozen = CrossAttention(query_dim=768, kv_dim=1024, n_heads=8)

        if stage1_ckpt is not None:
            ckpt = torch.load(stage1_ckpt, map_location='cpu')
            state = ckpt.get('model', ckpt)

            # Stage1 format: model dict with 'cmr_enc.*' keys
            cmr_state = {k.replace('cmr_enc.', ''): v
                         for k, v in state.items() if k.startswith('cmr_enc.')}
            if not cmr_state:
                # P0.1 standalone format: 'encoder' key, raw CMREncoder state dict
                cmr_state = ckpt.get('encoder', None)

            if cmr_state:
                msg = self.cmr_enc_frozen.load_state_dict(cmr_state, strict=False)
                print(f'[Stage2] Loaded CMR anchor: missing={len(msg.missing_keys)} '
                      f'unexpected={len(msg.unexpected_keys)}')

            # CrossAttn: only present in Stage1 checkpoints, not P0.1
            ca_state = {k.replace('cross_attn.', ''): v
                        for k, v in state.items() if k.startswith('cross_attn.')}
            if ca_state:
                self.cross_attn_frozen.load_state_dict(ca_state, strict=False)
                print('[Stage2] Loaded CrossAttn from Stage1 checkpoint.')
            else:
                print('[Stage2] No CrossAttn in checkpoint — using zero-init '
                      '(z_c_enhanced = z_c, pure CMR anchor).')

            print(f'[Stage2] Anchor loaded from {stage1_ckpt}')

        for p in self.fundus_enc_frozen.parameters(): p.requires_grad_(False)
        for p in self.cmr_enc_frozen.parameters():    p.requires_grad_(False)
        for p in self.cross_attn_frozen.parameters(): p.requires_grad_(False)

        # ── Trainable fundus encoder (fresh copy of RETFound) ─────────────────
        self.fundus_enc_new = FundusEncoder(ckpt=retfound_ckpt, freeze=False,
                                            grad_ckpt=grad_ckpt)

        # ── Projection heads (both trainable) ─────────────────────────────────
        self.proj_head_f = ProjHead(1024, proj_dim)
        self.proj_head_c = ProjHead(768,  proj_dim)

        # ── Optional task head for alpha > 0 ──────────────────────────────────
        self.task_head_f = TaskHead(1024, n_cls=n_cls, n_reg=n_reg) if n_cls > 0 else None

    def forward(self, fundus: torch.Tensor, cmr: torch.Tensor) -> dict:
        # frozen anchor: get z_c_enhanced
        with torch.no_grad():
            fundus_tokens = get_fundus_patch_tokens(self.fundus_enc_frozen, fundus)
            fundus_tokens = fundus_tokens.to(
                dtype=next(self.cmr_enc_frozen.parameters()).dtype)
            _, z_c = self.cmr_enc_frozen(cmr)
            attn_out = self.cross_attn_frozen(z_c, fundus_tokens)
            z_c_enhanced = z_c + attn_out                            # [B, 768]

        # trainable: new fundus encoder
        z_f  = self.fundus_enc_new(fundus)                           # [B, 1024]
        p_f  = self.proj_head_f(z_f)                                 # [B, 256] fp32
        p_c  = self.proj_head_c(z_c_enhanced)                       # [B, 256] fp32

        out = {'z_f': z_f, 'p_f': p_f, 'p_c': p_c}
        if self.task_head_f is not None:
            cls_f, reg_f = self.task_head_f(z_f)
            out['cls_f'] = cls_f
            out['reg_f'] = reg_f
        return out

    def trainable_parameters(self):
        params = (list(self.fundus_enc_new.parameters()) +
                  list(self.proj_head_f.parameters()) +
                  list(self.proj_head_c.parameters()))
        if self.task_head_f is not None:
            params += list(self.task_head_f.parameters())
        return params


# ── RKD Loss ──────────────────────────────────────────────────────────────────
def rkd_loss(p_f: torch.Tensor, p_c: torch.Tensor,
             temperature: float = 4.0) -> torch.Tensor:
    """
    Relational Knowledge Distillation: match pairwise similarity distributions.
    p_c.detach() so anchor gradients don't flow back.

    p_f, p_c: [B, D] L2-normalised
    Always computed in fp32 for numerical stability.
    """
    # force fp32 — safe to call inside autocast context
    p_f = p_f.float().detach().clone().requires_grad_(True)
    p_c = p_c.float().detach()
    # rebuild p_f with grad but in fp32 outside of any autocast influence
    with torch.cuda.amp.autocast(enabled=False):
        B = p_f.size(0)
        # recompute p_f in fp32 from the original (passed in as fp32 already from ProjHead)
        sim_c = (p_c @ p_c.T) / temperature            # [B, B]
        sim_f = (p_f @ p_f.T) / temperature            # [B, B]

        # mask diagonal (self-similarity is uninformative)
        # use -1e4 (safe in fp16, and >>  cosine/T range of ±1/temperature)
        mask  = ~torch.eye(B, dtype=torch.bool, device=p_f.device)
        target   = F.softmax(sim_c.masked_fill(~mask, -1e4), dim=-1)
        log_pred = F.log_softmax(sim_f.masked_fill(~mask, -1e4), dim=-1)

        return F.kl_div(log_pred, target.detach(), reduction='batchmean')
