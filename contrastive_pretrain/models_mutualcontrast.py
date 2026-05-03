"""
models_mutualcontrast.py

Mutual Contrastive + Downstream Task joint training model.

Architecture:
  fundus → FundusEncoder (full FT) → z_f [B, 1024]
                ├→ TaskHead_f → cls_f [B, 4 logits], reg_f [B, 3]
                └→ ProjHead_f → p_f [B, 256]  (L2 normalised)

  CMR    → CMREncoder    (full FT) → z_c [B, 768]
                ├→ TaskHead_c → cls_c [B, 4 logits], reg_c [B, 3]
                └→ ProjHead_c → p_c [B, 256]  (L2 normalised)

The two towers never communicate during forward (test-time fundus tower is
self-contained). CrossModalGating is NOT used in this design.

Loss:
  L_total = L_task_f + gamma * L_task_c + lam * L_align
  L_task  = mean(BCE cls losses) + L_reg
  L_align = symmetric InfoNCE (temperature=0.07)
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.models_dualtower import FundusEncoder, TaskHead
from contrastive_pretrain.models_cmr import CMREncoder, load_medsam_vision_encoder_weights

HERE = pathlib.Path(__file__).parent

RETFOUND_CKPT = str(ROOT / 'RETFound_cfp_weights.pth')
MEDSAM_CKPT   = str(ROOT / 'pretrained_weights/hf_cache/'
                    'models--flaviagiammarino--medsam-vit-base/blobs/'
                    'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c')

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]
REG_COLS = [
    'LV ejection fraction',
    'LV end diastolic volume',
    'LV end systolic volume',
]


# ── ProjHead ──────────────────────────────────────────────────────────────────
class ProjHead(nn.Module):
    """Two-layer MLP projection head with L2-normalised output."""

    def __init__(self, in_dim: int, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=-1, eps=1e-8)


# ── MutualContrastModel ───────────────────────────────────────────────────────
class MutualContrastModel(nn.Module):
    """
    Two independent towers (fundus + CMR). No cross-modal communication during
    forward, so the fundus tower is fully self-contained at test time.
    """

    def __init__(
        self,
        retfound_ckpt: str = RETFOUND_CKPT,
        medsam_ckpt:   str = MEDSAM_CKPT,
        dim_f: int = 1024,
        dim_c: int = 768,
        proj_dim: int = 256,
        grad_ckpt: bool = False,
    ):
        super().__init__()
        self.fundus_enc = FundusEncoder(
            ckpt=retfound_ckpt, freeze=False,
            grad_ckpt=grad_ckpt, freeze_blocks=0,
        )
        self.cmr_enc = CMREncoder(
            proj_dim=proj_dim, num_frames=4, embed_dim=dim_c,
            medsam_ckpt=medsam_ckpt, freeze_backbone=False,
        )
        self.task_head_f = TaskHead(dim_f)
        self.task_head_c = TaskHead(dim_c)
        self.proj_head_f = ProjHead(dim_f, proj_dim)
        self.proj_head_c = ProjHead(dim_c, proj_dim)

    def forward(self, fundus: torch.Tensor, cmr: torch.Tensor,
                fundus_only: bool = False) -> dict:
        """
        Args:
            fundus: (B, 3, 224, 224)
            cmr:    (B, 4, 1, 224, 224)
            fundus_only: skip CMR encoder (saves memory when gamma=0, lam=0)
        Returns dict with keys:
            cls_f, reg_f, cls_c, reg_c : task head outputs
            p_f, p_c                   : L2-normalised projection embeddings
        """
        z_f = self.fundus_enc(fundus)          # (B, 1024)
        cls_f, reg_f = self.task_head_f(z_f)
        p_f = self.proj_head_f(z_f.float())    # cast fp32 for cosine stability

        if fundus_only:
            # dummy CMR outputs (all zeros, not used in loss when gamma=lam=0)
            p_c   = torch.zeros_like(p_f)
            cls_c = [torch.zeros_like(c) for c in cls_f]
            reg_c = [torch.zeros_like(r) for r in reg_f]
        else:
            _, z_c = self.cmr_enc(cmr)         # (B, 768)
            cls_c, reg_c = self.task_head_c(z_c)
            p_c = self.proj_head_c(z_c.float())

        return {
            'cls_f': cls_f, 'reg_f': reg_f,
            'cls_c': cls_c, 'reg_c': reg_c,
            'p_f': p_f, 'p_c': p_c,
        }

    def fundus_only_state_dict(self) -> dict:
        """Returns state dict containing only fundus encoder + task_head_f."""
        keys = (
            {f'fundus_enc.{k}': v for k, v in self.fundus_enc.state_dict().items()} |
            {f'task_head_f.{k}': v for k, v in self.task_head_f.state_dict().items()}
        )
        return keys


# ── Loss ──────────────────────────────────────────────────────────────────────
def _contrastive_loss(p_f: torch.Tensor, p_c: torch.Tensor,
                      temperature: float = 0.07) -> torch.Tensor:
    """Symmetric InfoNCE. p_f, p_c must be L2-normalised, shape (B, D)."""
    sim = (p_f @ p_c.T) / temperature            # (B, B)
    targets = torch.arange(p_f.size(0), device=p_f.device)
    loss_f2c = F.cross_entropy(sim, targets)
    loss_c2f = F.cross_entropy(sim.T, targets)
    return (loss_f2c + loss_c2f) / 2.0


def _task_loss(
    cls_preds: list,
    reg_preds: list,
    cls_labels: torch.Tensor,
    reg_labels: torch.Tensor,
    reg_mask: torch.Tensor,
    pos_weights: list | None,
    reg_norm: list,
) -> torch.Tensor:
    """
    L_task = mean(BCE cls losses across 4 diseases) + L_reg

    reg_norm: list of {'mean': float, 'std': float} per reg target.
    Normalisation applied before MSE so all three reg targets are on similar scale.
    """
    device = cls_preds[0].device

    # classification
    cls_losses = []
    for i, pred in enumerate(cls_preds):
        valid = cls_labels[:, i] >= 0
        if valid.sum() == 0:
            continue
        if pos_weights is not None:
            pw = torch.tensor([pos_weights[i]], dtype=torch.float32, device=device)
            crit = nn.BCEWithLogitsLoss(pos_weight=pw)
        else:
            crit = nn.BCEWithLogitsLoss()
        cls_losses.append(crit(pred[valid], cls_labels[valid, i]))
    l_cls = torch.stack(cls_losses).mean() if cls_losses else torch.tensor(0.0, device=device)

    # regression (normalised MSE, masked)
    reg_losses = []
    for i, pred in enumerate(reg_preds):
        mu, sigma = reg_norm[i]['mean'], reg_norm[i]['std']
        t_norm = (reg_labels[:, i] - mu) / sigma
        valid  = reg_mask[:, i]
        if valid.sum() == 0:
            continue
        reg_losses.append(F.mse_loss(pred[valid], t_norm[valid]))
    n_reg  = len(reg_losses)
    denom  = reg_mask.float().sum() * n_reg + 1e-6
    l_reg  = (torch.stack(reg_losses).sum() * reg_mask.shape[0] / denom
              if reg_losses else torch.tensor(0.0, device=device))

    return l_cls + l_reg


class MutualContrastLoss(nn.Module):
    """
    L_total = L_task_f + gamma * L_task_c + lam * L_align

    pos_weights_f / pos_weights_c: list of 4 floats for BCE pos_weight per disease.
    reg_norm: list of 3 dicts {'mean', 'std'} for LVEF / LVEDV / LVESV.
    """

    def __init__(
        self,
        pos_weights_f: list | None = None,
        pos_weights_c: list | None = None,
        reg_norm: list | None = None,
        gamma: float = 1.0,
        lam: float = 0.3,
        temperature: float = 0.07,
    ):
        super().__init__()
        self.pos_weights_f = pos_weights_f
        self.pos_weights_c = pos_weights_c
        self.reg_norm = reg_norm or [
            {'mean': 58.72, 'std': 7.37},
            {'mean': 149.56, 'std': 36.11},
            {'mean': 62.62, 'std': 22.85},
        ]
        self.gamma = gamma
        self.lam   = lam
        self.temperature = temperature

    def forward(self, results: dict, batch: dict) -> dict:
        cls_labels = batch['cls_labels']
        reg_labels = batch['reg_labels']
        reg_mask   = batch['reg_mask']

        l_task_f = _task_loss(
            results['cls_f'], results['reg_f'],
            cls_labels, reg_labels, reg_mask,
            self.pos_weights_f, self.reg_norm,
        )
        l_task_c = _task_loss(
            results['cls_c'], results['reg_c'],
            cls_labels, reg_labels, reg_mask,
            self.pos_weights_c, self.reg_norm,
        )
        # cast to fp32 before InfoNCE (projections already fp32)
        l_align = _contrastive_loss(
            results['p_f'], results['p_c'], self.temperature
        )
        total = l_task_f + self.gamma * l_task_c + self.lam * l_align
        return {
            'total':    total,
            'task_f':   l_task_f,
            'task_c':   l_task_c,
            'align':    l_align,
        }


def build_loss(
    gamma: float = 1.0,
    lam: float = 0.3,
    temperature: float = 0.07,
    pos_weights_json_cmr: str | None = None,
    pos_weights_json_fundus: str | None = None,
    norm_params_json: str | None = None,
) -> MutualContrastLoss:
    """Load pos_weights and norm_params from JSON files and build loss."""
    pos_weights_json_cmr    = pos_weights_json_cmr    or str(HERE / 'pos_weights_cmr.json')
    pos_weights_json_fundus = pos_weights_json_fundus or str(HERE / 'pos_weights_fundus.json')
    norm_params_json        = norm_params_json        or str(HERE / 'norm_params.json')

    CLS_KEYS = ['composite_ischemic_hd', 'prevalent_I21',
                'composite_cardiomyopathy_hf', 'composite_af_arrhythmia']
    REG_KEYS = ['lvef', 'lvedv', 'lvesv']

    with open(pos_weights_json_cmr) as f:
        pw_c = json.load(f)
    with open(pos_weights_json_fundus) as f:
        pw_f = json.load(f)
    with open(norm_params_json) as f:
        norm = json.load(f)

    return MutualContrastLoss(
        pos_weights_f=[pw_f[k] for k in CLS_KEYS],
        pos_weights_c=[pw_c[k] for k in CLS_KEYS],
        reg_norm=[{'mean': norm[k]['mean'], 'std': norm[k]['std']} for k in REG_KEYS],
        gamma=gamma,
        lam=lam,
        temperature=temperature,
    )
