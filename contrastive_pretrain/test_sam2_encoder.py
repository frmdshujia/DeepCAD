"""
test_sam2_encoder.py  —  Verify SAM2.1 Hiera backbones without Hydra.

Directly instantiates Hiera from known config params and loads trunk weights
from SAM2.1 checkpoints. No build_sam2, no Hydra config resolution.

Usage:
    conda run -n modeltrain python contrastive_pretrain/test_sam2_encoder.py
"""
from __future__ import annotations
import sys, pathlib, torch, torch.nn as nn

HERE        = pathlib.Path(__file__).parent
WEIGHTS_DIR = HERE.parent / "pretrained_weights" / "sam2"
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"torch  : {torch.__version__}")
print(f"device : {device}\n")

# ── Hiera configs per variant (extracted from sam2.1 yaml files) ──────────────
# backbone_channel_list[-1] = last-stage output channels (most semantic)
HIERA_CONFIGS = {
    "tiny": dict(
        embed_dim=96, num_heads=1,
        stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
        out_channels=768,   # backbone_channel_list[0]
        ckpt="sam2.1_hiera_tiny.pt",
    ),
    "small": dict(
        embed_dim=96, num_heads=1,
        stages=(1, 2, 11, 2),
        global_att_blocks=(7, 10, 13),
        window_pos_embed_bkg_spatial_size=(7, 7),
        out_channels=768,
        ckpt="sam2.1_hiera_small.pt",
    ),
    "base_plus": dict(
        embed_dim=112, num_heads=2,
        stages=(2, 3, 16, 3),          # Hiera defaults
        global_att_blocks=(12, 16, 20), # Hiera defaults
        window_pos_embed_bkg_spatial_size=(14, 14),
        out_channels=896,   # backbone_channel_list[0]
        ckpt="sam2.1_hiera_base_plus.pt",
    ),
}


def load_hiera_trunk(variant: str) -> tuple[nn.Module | None, str]:
    cfg = HIERA_CONFIGS[variant]
    ckpt_path = WEIGHTS_DIR / cfg["ckpt"]
    if not ckpt_path.exists():
        return None, f"checkpoint not found: {ckpt_path}"

    from sam2.modeling.backbones.hieradet import Hiera
    trunk = Hiera(
        embed_dim=cfg["embed_dim"],
        num_heads=cfg["num_heads"],
        stages=cfg["stages"],
        global_att_blocks=cfg["global_att_blocks"],
        window_pos_embed_bkg_spatial_size=cfg["window_pos_embed_bkg_spatial_size"],
        return_interm_layers=True,
    )

    # SAM2 checkpoint structure: {"model": {"image_encoder.trunk.*": ...}}
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    trunk_state = {
        k.replace("image_encoder.trunk.", ""): v
        for k, v in state.items()
        if k.startswith("image_encoder.trunk.")
    }
    if not trunk_state:
        return None, "no image_encoder.trunk keys found in checkpoint"

    missing, unexpected = trunk.load_state_dict(trunk_state, strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}")
    trunk = trunk.to(device).eval()
    return trunk, "ok"


def probe(trunk: nn.Module, variant: str) -> int:
    cfg = HIERA_CONFIGS[variant]
    x = torch.randn(2, 3, 224, 224, device=device)

    with torch.no_grad(), torch.cuda.amp.autocast():
        # Hiera returns list of (B, C, H, W) tensors, one per stage
        out = trunk(x)

    # Use last stage — highest semantic, smallest spatial (7×7 for 224 input)
    feat = out[-1]   # (B, C, H, W)

    print(f"  raw last-stage     : {tuple(feat.shape)}  (B, C, H, W)")

    pool   = nn.AdaptiveAvgPool2d(4)
    pooled = pool(feat)                         # (B, C, 4, 4)
    B, C, P, _ = pooled.shape
    tokens = pooled.permute(0, 2, 3, 1).reshape(B, P * P, C)
    print(f"  after pool(4×4)    : {tuple(tokens.shape)}  tokens={P*P}  dim={C}")

    n_params = sum(p.numel() for p in trunk.parameters()) / 1e6
    print(f"  trunk params       : {n_params:.1f}M")

    proj_note = "(no proj needed)" if C == 768 else f"→ need Linear({C}, 768)"
    print(f"  projection         : {proj_note}")
    return C


# ── Run ───────────────────────────────────────────────────────────────────────
print("=" * 60)
results: dict[str, int | None] = {}

for variant in ["tiny", "small", "base_plus"]:
    print(f"\n[SAM2.1 {variant}]")
    trunk, status = load_hiera_trunk(variant)
    if trunk is None:
        print(f"  SKIP — {status}")
        results[variant] = None
        continue
    try:
        out_dim = probe(trunk, variant)
        results[variant] = out_dim
        print(f"  STATUS: OK")
    except Exception as e:
        import traceback
        print(f"  STATUS: FAILED")
        traceback.print_exc()
        results[variant] = None
    finally:
        del trunk
        torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("Summary:")
for v, dim in results.items():
    if dim is None:
        print(f"  {v:12s}: FAILED / SKIP")
    else:
        proj = f"Linear({dim}, 768)" if dim != 768 else "768 — no projection"
        print(f"  {v:12s}: out_dim={dim}  →  {proj}")
