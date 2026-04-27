"""
models_dualtower.py

Dual-tower cross-modal model for cardiovascular prediction.
  - FundusEncoder : RETFound ViT-L (1024-dim)
  - CMREncoder    : MedSAM ViT-B (768-dim pooled)
  - CrossModalGating : bidirectional cross-attention + sigmoid gating + LoRA backflow
  - TaskHead      : 4 classification + 3 regression heads
  - MultiTaskLoss : BCEWithLogitsLoss (pos-weighted) + MSE (reg) + cosine align loss
  - DualTowerModel: ties everything together, supports cmr_only / fundus_only / paired modes
"""
from __future__ import annotations

import sys
import pathlib
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

# ── path setup ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import models_vit
from contrastive_pretrain.models_cmr import CMREncoder, load_medsam_vision_encoder_weights

HERE = pathlib.Path(__file__).parent

RETFOUND_CKPT  = str(ROOT / 'RETFound_cfp_weights.pth')
MEDSAM_CKPT    = str(ROOT / 'pretrained_weights/hf_cache/models--flaviagiammarino--medsam-vit-base/blobs/b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c')

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]


# ── FundusEncoder ────────────────────────────────────────────────────────────
class FundusEncoder(nn.Module):
    """RETFound ViT-L; forward() returns (B, 1024) global-pool features."""

    def __init__(self, ckpt: Optional[str] = RETFOUND_CKPT, freeze: bool = False,
                 grad_ckpt: bool = False, freeze_blocks: int = 0):
        """freeze_blocks: freeze first N transformer blocks (0=train all)."""
        super().__init__()
        self.vit = models_vit.vit_large_patch16(
            global_pool=True,
            num_classes=0,
            drop_path_rate=0.1,
        )
        if ckpt is not None:
            self._load_retfound(ckpt)
        if freeze:
            for p in self.vit.parameters():
                p.requires_grad = False
        elif freeze_blocks > 0:
            # Freeze patch embed, positional embedding, and first freeze_blocks blocks
            for p in self.vit.patch_embed.parameters():
                p.requires_grad = False
            for blk in self.vit.blocks[:freeze_blocks]:
                for p in blk.parameters():
                    p.requires_grad = False
            n_frozen = sum(p.numel() for p in self.vit.parameters() if not p.requires_grad)
            n_total  = sum(p.numel() for p in self.vit.parameters())
            print(f'[FundusEncoder] frozen {n_frozen/1e6:.1f}M / {n_total/1e6:.1f}M params '
                  f'(first {freeze_blocks} blocks)')
        self._grad_ckpt = grad_ckpt

    def _load_retfound(self, path: str):
        sd = torch.load(path, map_location='cpu')
        # RETFound checkpoint may be wrapped in 'model' key
        if 'model' in sd:
            sd = sd['model']
        msg = self.vit.load_state_dict(sd, strict=False)
        print(f'[FundusEncoder] loaded {path}  missing={len(msg.missing_keys)}  '
              f'unexpected={len(msg.unexpected_keys)}')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._grad_ckpt and self.training:
            return self._forward_grad_ckpt(x)
        return self.vit.forward_features(x)  # (B, 1024)

    def _forward_grad_ckpt(self, x: torch.Tensor) -> torch.Tensor:
        from torch.utils.checkpoint import checkpoint
        B = x.shape[0]
        x = self.vit.patch_embed(x)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.vit.pos_embed
        x = self.vit.pos_drop(x)
        for blk in self.vit.blocks:
            x = checkpoint(blk, x, use_reentrant=False)
        if self.vit.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            x = self.vit.fc_norm(x)
        else:
            x = self.vit.norm(x)
            x = x[:, 0]
        return x


# ── LoRA layer ───────────────────────────────────────────────────────────────
class LoRALayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, r: int = 16, alpha: int = 32):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        self.scale  = alpha / r
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(x)) * self.scale


# ── CrossModalGating ─────────────────────────────────────────────────────────
class CrossModalGating(nn.Module):
    """Bidirectional cross-attention + sigmoid gating + LoRA backflow (§5.2)."""

    def __init__(self, dim_f: int = 1024, dim_c: int = 768,
                 hidden: int = 512, num_heads: int = 8, r: int = 16):
        super().__init__()
        self.proj_f = nn.Linear(dim_f, hidden)
        self.proj_c = nn.Linear(dim_c, hidden)
        # Use PyTorch 1.8-compatible MHA (no batch_first)
        self.attn_f2c = nn.MultiheadAttention(hidden, num_heads)
        self.attn_c2f = nn.MultiheadAttention(hidden, num_heads)
        self.norm_f = nn.LayerNorm(hidden)
        self.norm_c = nn.LayerNorm(hidden)
        self.gate_f = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.gate_c = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.back_f = nn.Linear(hidden, dim_f)
        self.back_c = nn.Linear(hidden, dim_c)
        self.lora_f = LoRALayer(dim_f, dim_f, r=r)
        self.lora_c = LoRALayer(dim_c, dim_c, r=r)

    def forward(self, z_f: torch.Tensor, z_c: torch.Tensor):
        # z_f (B, dim_f), z_c (B, dim_c)
        zf = self.proj_f(z_f).unsqueeze(0)   # (1, B, hidden)
        zc = self.proj_c(z_c).unsqueeze(0)   # (1, B, hidden)
        f_from_c, _ = self.attn_f2c(zf, zc, zc)   # (1, B, hidden)
        f_from_c = self.norm_f(f_from_c.squeeze(0))
        delta_f = self.gate_f(f_from_c) * f_from_c  # (B, hidden)
        c_from_f, _ = self.attn_c2f(zc, zf, zf)
        c_from_f = self.norm_c(c_from_f.squeeze(0))
        delta_c = self.gate_c(c_from_f) * c_from_f
        delta_f = self.back_f(delta_f)
        delta_c = self.back_c(delta_c)
        return z_f + self.lora_f(delta_f), z_c + self.lora_c(delta_c)


# ── TaskHead ─────────────────────────────────────────────────────────────────
class TaskHead(nn.Module):
    def __init__(self, in_dim: int, n_cls: int = 4, n_reg: int = 3, dropout: float = 0.3):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.cls_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n_cls)
        ])
        self.reg_heads = nn.ModuleList([
            nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 1))
            for _ in range(n_reg)
        ])

    def forward(self, z: torch.Tensor):
        z = self.drop(z)
        cls_out = [h(z).squeeze(-1) for h in self.cls_heads]
        reg_out = [h(z).squeeze(-1) for h in self.reg_heads]
        return cls_out, reg_out


# ── MultiTaskLoss ─────────────────────────────────────────────────────────────
class MultiTaskLoss(nn.Module):
    def __init__(self, pos_weights_cmr: list, pos_weights_fundus: list,
                 reg_norm: list, lambda_align: float = 0.1):
        super().__init__()
        self.lambda_align = lambda_align
        # CMR pos weights
        self.cls_crit_cmr = nn.ModuleList([
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]))
            for w in pos_weights_cmr
        ])
        # Fundus pos weights
        self.cls_crit_fundus = nn.ModuleList([
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]))
            for w in pos_weights_fundus
        ])
        self.reg_norm = reg_norm  # list of {'mean', 'std'}

    def _cls(self, preds, labels, criteria):
        loss = torch.tensor(0.0, device=preds[0].device)
        for pred, crit in zip(preds, criteria):
            valid = labels[:, len(loss.shape)] >= 0 if False else None
            # rebuild criterion on the right device each time
            pw = crit.pos_weight.to(preds[0].device)
            c = nn.BCEWithLogitsLoss(pos_weight=pw)
            valid = labels[:, criteria.index(crit)] >= 0
            if valid.sum() == 0:
                continue
            loss = loss + c(pred[valid], labels[valid, criteria.index(crit)])
        return loss

    def cls_loss(self, preds, labels, modality: str = 'cmr'):
        criteria = self.cls_crit_cmr if modality == 'cmr' else self.cls_crit_fundus
        loss = torch.tensor(0.0, device=preds[0].device)
        for i, (pred, crit) in enumerate(zip(preds, criteria)):
            valid = labels[:, i] >= 0
            if valid.sum() == 0:
                continue
            pw = crit.pos_weight.to(preds[0].device)
            c  = nn.BCEWithLogitsLoss(pos_weight=pw)
            loss = loss + c(pred[valid], labels[valid, i])
        return loss

    def reg_loss(self, preds, targets, reg_mask=None):
        loss = torch.tensor(0.0, device=preds[0].device)
        for i, pred in enumerate(preds):
            mu, sigma = self.reg_norm[i]['mean'], self.reg_norm[i]['std']
            t_norm = (targets[:, i] - mu) / sigma
            valid = ~torch.isnan(targets[:, i])
            if reg_mask is not None:
                valid = valid & reg_mask[:, i]
            if valid.sum() == 0:
                continue
            loss = loss + F.mse_loss(pred[valid], t_norm[valid])
        return loss

    def align_loss(self, z_f, z_c):
        return 1.0 - (z_f * z_c).sum(dim=-1).mean()

    def compute(self, results: dict, batch: dict) -> torch.Tensor:
        mode = batch['mode']
        cls  = batch['cls_labels']
        reg  = batch['reg_labels']
        mask = batch.get('reg_mask')

        if mode == 'cmr_only':
            loss = self.cls_loss(results['cls'], cls, 'cmr')
            loss = loss + self.reg_loss(results['reg'], reg)
        elif mode == 'fundus_only':
            loss = self.cls_loss(results['cls'], cls, 'fundus')
            loss = loss + self.reg_loss(results['reg'], reg, mask)
        elif mode == 'paired':
            loss = self.cls_loss(results['cmr_cls_base'], cls, 'cmr')
            loss = loss + self.cls_loss(results['fundus_cls_base'], cls, 'fundus')
            loss = loss + self.reg_loss(results['cmr_reg_base'], reg)
            loss = loss + self.reg_loss(results['fundus_reg_base'], reg, mask)
            w = 0.5
            loss = loss + w * self.cls_loss(results['cmr_cls_en'], cls, 'cmr')
            loss = loss + w * self.cls_loss(results['fundus_cls_en'], cls, 'fundus')
            loss = loss + w * self.reg_loss(results['cmr_reg_en'], reg)
            loss = loss + w * self.reg_loss(results['fundus_reg_en'], reg, mask)
            loss = loss + self.lambda_align * self.align_loss(
                results['align_f'], results['align_c'])
        else:
            raise ValueError(f'Unknown mode: {mode}')
        return loss


# ── DualTowerModel ─────────────────────────────────────────────────────────
class DualTowerModel(nn.Module):
    """
    Full dual-tower model.

    forward(batch) where batch is a dict with:
      mode: 'cmr_only' | 'fundus_only' | 'paired'
      cmr:  (B, 4, 1, 224, 224) for cmr_only / paired
      fundus: (B, 3, 224, 224) for fundus_only / paired
      cls_labels, reg_labels, reg_mask (unused in forward, passed to loss)
    """

    def __init__(
        self,
        retfound_ckpt: str = RETFOUND_CKPT,
        medsam_ckpt:   str = MEDSAM_CKPT,
        freeze_fundus: bool = False,
        freeze_cmr:    bool = False,
        dim_f: int = 1024,
        dim_c: int = 768,
        proj_dim: int = 256,
        mode: str = 'paired',          # 'cmr_only' | 'fundus_only' | 'paired'
        grad_ckpt: bool = False,
        freeze_fundus_blocks: int = 0,  # freeze first N ViT-L blocks
        freeze_cmr_blocks: bool = False,
    ):
        super().__init__()
        self._mode = mode
        need_fundus = mode in ('fundus_only', 'paired')
        need_cmr    = mode in ('cmr_only', 'paired')

        self.fundus_enc = FundusEncoder(ckpt=retfound_ckpt, freeze=freeze_fundus,
                                        grad_ckpt=grad_ckpt,
                                        freeze_blocks=freeze_fundus_blocks) if need_fundus else None
        self.cmr_enc    = CMREncoder(
            proj_dim=proj_dim, num_frames=4, embed_dim=dim_c,
            medsam_ckpt=medsam_ckpt, freeze_backbone=freeze_cmr or freeze_cmr_blocks,
        ) if need_cmr else None
        self.cross_modal  = CrossModalGating(dim_f=dim_f, dim_c=dim_c) if mode == 'paired' else None
        self.head_fundus  = TaskHead(dim_f) if need_fundus else None
        self.head_cmr     = TaskHead(dim_c) if need_cmr else None
        self.align_proj_f = nn.Linear(dim_f, proj_dim) if mode == 'paired' else None
        self.align_proj_c = nn.Linear(dim_c, proj_dim) if mode == 'paired' else None

    def forward(self, batch: dict) -> dict:
        mode = batch['mode']
        results = {}

        if mode == 'cmr_only':
            _, z_c = self.cmr_enc(batch['cmr'])   # (B, 768)
            cls, reg = self.head_cmr(z_c)
            results.update({'cls': cls, 'reg': reg})

        elif mode == 'fundus_only':
            z_f = self.fundus_enc(batch['fundus'])  # (B, 1024)
            cls, reg = self.head_fundus(z_f)
            results.update({'cls': cls, 'reg': reg})

        elif mode == 'paired':
            z_f = self.fundus_enc(batch['fundus'])       # (B, 1024)
            _, z_c = self.cmr_enc(batch['cmr'])          # (B, 768)
            cls_f0, reg_f0 = self.head_fundus(z_f)
            cls_c0, reg_c0 = self.head_cmr(z_c)
            z_f_en, z_c_en = self.cross_modal(z_f, z_c)
            cls_f1, reg_f1 = self.head_fundus(z_f_en)
            cls_c1, reg_c1 = self.head_cmr(z_c_en)
            results.update({
                'fundus_cls_base': cls_f0, 'fundus_reg_base': reg_f0,
                'cmr_cls_base':    cls_c0, 'cmr_reg_base':    reg_c0,
                'fundus_cls_en':   cls_f1, 'fundus_reg_en':   reg_f1,
                'cmr_cls_en':      cls_c1, 'cmr_reg_en':      reg_c1,
                'align_f': F.normalize(self.align_proj_f(z_f), dim=-1),
                'align_c': F.normalize(self.align_proj_c(z_c), dim=-1),
            })
        else:
            raise ValueError(f'Unknown mode: {mode}')

        return results


# ── Factory ────────────────────────────────────────────────────────────────
def build_loss(pos_weights_json_cmr: str | None = None,
               pos_weights_json_fundus: str | None = None,
               norm_params_json: str | None = None) -> MultiTaskLoss:
    if pos_weights_json_cmr is None:
        pos_weights_json_cmr = str(HERE / 'pos_weights_cmr.json')
    if pos_weights_json_fundus is None:
        pos_weights_json_fundus = str(HERE / 'pos_weights_fundus.json')
    if norm_params_json is None:
        norm_params_json = str(HERE / 'norm_params.json')

    with open(pos_weights_json_cmr) as f:
        pw_cmr = json.load(f)
    with open(pos_weights_json_fundus) as f:
        pw_f = json.load(f)
    with open(norm_params_json) as f:
        norm = json.load(f)

    CLS_KEYS = ['composite_ischemic_hd', 'prevalent_I21',
                'composite_cardiomyopathy_hf', 'composite_af_arrhythmia']
    REG_KEYS = ['lvef', 'lvedv', 'lvesv']

    return MultiTaskLoss(
        pos_weights_cmr=[pw_cmr[k] for k in CLS_KEYS],
        pos_weights_fundus=[pw_f[k] for k in CLS_KEYS],
        reg_norm=[
            {'mean': norm[k]['mean'], 'std': norm[k]['std']}
            for k in REG_KEYS
        ],
    )
