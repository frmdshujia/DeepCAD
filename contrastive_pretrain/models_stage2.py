"""models_stage2.py -- dual-tower model for Stage 2 image-level fundus↔CMR
contrastive learning.

Left tower  : RETFound ViT-Large (reuses FundusContrastModel from models_contrast)
Right tower : MedSAM ViT-B + frame attention pool + projection head (CMREncoder
              from models_cmr, re-aliased as CMRImageEncoder)

Symmetric CLIP-style hard-label InfoNCE is computed in the engine; this file
only exposes the joint module and its optimizer parameter groups.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional, Tuple, List, Dict
import torch
import torch.nn as nn

from models_contrast import FundusContrastModel
from models_cmr import CMREncoder as CMRImageEncoder
import util.lr_decay as lrd


class Stage2DualTower(nn.Module):
    """Composite module. Exposes:
        .fundus : FundusContrastModel (RETFound ViT-L + ProjectionHead)
        .cmr    : CMRImageEncoder     (MedSAM ViT-B + FrameAttnPool + proj)
    forward(fundus_img, cmr_frames) -> (z_fundus, z_cmr) both L2-normalized.
    """

    def __init__(self,
                 proj_dim: int = 256,
                 num_frames: int = 4,
                 drop_path_rate: float = 0.1,
                 medsam_ckpt: Optional[str] = None,
                 retfound_ckpt: Optional[str] = None,
                 freeze_fundus_backbone: bool = False,
                 freeze_cmr_backbone: bool = False):
        super().__init__()
        self.fundus = FundusContrastModel(proj_dim=proj_dim, drop_path_rate=drop_path_rate)
        if retfound_ckpt is not None:
            self.fundus.load_pretrained(retfound_ckpt)
        if freeze_fundus_backbone:
            for p in self.fundus.backbone.parameters():
                p.requires_grad = False

        self.cmr = CMRImageEncoder(
            proj_dim=proj_dim, num_frames=num_frames,
            embed_dim=768, img_size=224,
            medsam_ckpt=medsam_ckpt,
            freeze_backbone=freeze_cmr_backbone,
        )

    def forward(self, fundus_img: torch.Tensor, cmr_frames: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """fundus_img: (B, 3, 224, 224); cmr_frames: (B, F, 1, 224, 224) or (B, F, 224, 224).
        Returns (z_fundus (B, D), z_cmr (B, D)), both L2-normalized."""
        z_f = self.fundus(fundus_img)
        z_c, _ = self.cmr(cmr_frames)
        return z_f, z_c


def build_stage2_param_groups(model: Stage2DualTower,
                              weight_decay: float = 0.05,
                              fundus_layer_decay: float = 0.75,
                              cmr_layer_decay: float = 0.80,
                              fundus_proj_lr_scale: float = 10.0,
                              cmr_pool_lr_scale: float = 20.0,
                              cmr_proj_lr_scale: float = 20.0) -> List[Dict]:
    """Optimizer parameter groups with layer-wise LR decay.

    We treat the two backbones as pretrained (small base LR) and the newly
    initialized heads (projection head, frame attention pool) with larger
    lr_scale. Optimizer base LR = args.lr; each group's effective LR =
    lr_scale * base LR (compatible with util.lr_sched.adjust_learning_rate).

    Suggested: args.blr = 1e-5 => backbone top-layer LR ≈ 1e-5;
      fundus proj  (lr_scale=10) => 1e-4
      cmr   pool   (lr_scale=20) => 2e-4
      cmr   proj   (lr_scale=20) => 2e-4
    """
    no_wd_fundus = model.fundus.backbone.no_weight_decay() \
        if hasattr(model.fundus.backbone, 'no_weight_decay') else set()

    fundus_groups = lrd.param_groups_lrd(
        model.fundus.backbone,
        weight_decay=weight_decay,
        no_weight_decay_list=no_wd_fundus,
        layer_decay=fundus_layer_decay,
    )

    fundus_proj_group = {
        'params': list(model.fundus.proj_head.parameters()),
        'lr_scale': fundus_proj_lr_scale,
        'weight_decay': weight_decay,
    }

    # MedSAM ViT-B layer-wise decay (12 blocks) — manual grouping because
    # util.lr_decay.param_groups_lrd assumes timm-style vit with cls_token.
    cmr_backbone_groups = _mediasm_layer_groups(
        model.cmr.backbone, weight_decay=weight_decay, layer_decay=cmr_layer_decay,
    )

    cmr_pool_group = {
        'params': list(model.cmr.frame_pool.parameters()),
        'lr_scale': cmr_pool_lr_scale,
        'weight_decay': weight_decay,
    }
    cmr_proj_group = {
        'params': list(model.cmr.proj.parameters()),
        'lr_scale': cmr_proj_lr_scale,
        'weight_decay': weight_decay,
    }

    return list(fundus_groups) + [fundus_proj_group] + cmr_backbone_groups + [cmr_pool_group, cmr_proj_group]


def _mediasm_layer_groups(backbone, weight_decay: float, layer_decay: float) -> List[Dict]:
    """Layer-wise LR decay for our SamVisionEncoderViTB.
    Layer 0 = patch_embed + pos_embed, layers 1..12 = each transformer block."""
    num_layers = len(backbone.blocks) + 1  # +1 for patch_embed/pos_embed
    layer_scales = [layer_decay ** (num_layers - 1 - i) for i in range(num_layers + 1)]

    groups: Dict[str, Dict] = {}

    def _add(pname: str, p: nn.Parameter, layer_id: int):
        if not p.requires_grad:
            return
        is_no_decay = p.ndim <= 1 or pname.endswith('.bias') or 'rel_pos' in pname or 'pos_embed' in pname
        key = f'cmr_layer_{layer_id}_{"no_decay" if is_no_decay else "decay"}'
        if key not in groups:
            groups[key] = {
                'params': [],
                'lr_scale': layer_scales[layer_id],
                'weight_decay': 0.0 if is_no_decay else weight_decay,
            }
        groups[key]['params'].append(p)

    # patch_embed + pos_embed -> layer 0
    for n, p in backbone.patch_embed.named_parameters():
        _add(f'patch_embed.{n}', p, 0)
    _add('pos_embed', backbone.pos_embed, 0)

    # blocks -> layers 1..N
    for i, blk in enumerate(backbone.blocks):
        for n, p in blk.named_parameters():
            _add(f'blocks.{i}.{n}', p, i + 1)

    return list(groups.values())


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--medsam_ckpt', type=str, default=None)
    ap.add_argument('--retfound_ckpt', type=str, default=None)
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    model = Stage2DualTower(
        proj_dim=256, num_frames=4,
        medsam_ckpt=args.medsam_ckpt,
        retfound_ckpt=args.retfound_ckpt,
    ).to(args.device)
    model.eval()

    fundus = torch.randn(2, 3, 224, 224, device=args.device)
    cmr = torch.randn(2, 4, 1, 224, 224, device=args.device)
    with torch.no_grad():
        zf, zc = model(fundus, cmr)
    n = sum(p.numel() for p in model.parameters())
    print(f'[stage2] dual-tower total params: {n/1e6:.1f}M')
    print(f'[stage2] z_fundus: {tuple(zf.shape)} norm={zf.norm(dim=-1).mean():.3f}')
    print(f'[stage2] z_cmr   : {tuple(zc.shape)} norm={zc.norm(dim=-1).mean():.3f}')

    # check param groups
    groups = build_stage2_param_groups(model)
    tot = sum(sum(p.numel() for p in g['params']) for g in groups)
    print(f'[stage2] param groups: {len(groups)}, total params in groups: {tot/1e6:.1f}M')
    lrs = sorted({g['lr_scale'] for g in groups})
    print(f'[stage2] distinct lr_scales: {lrs}')
    print('[stage2] OK')
