"""
Recompute validation metrics for checkpoints_cmr_v3/best.pth using the same
dataset + evaluate() path as train_cmr_v3.py.

Canonical recomputed numbers live in checkpoints_cmr_v3/metrics_verified.json
(mean_AUC on task1_cmr_val ~0.685 under conda env modeltrain / PyTorch 2.1).

Run:

  conda activate modeltrain
  python contrastive_pretrain/verify_cmr_v3_val_auc.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import torch
from torch.utils.data import DataLoader

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.train_cmr_v3 import (  # noqa: E402
    VAL_CSV,
    CLS_COLS,
    CMRDatasetV3,
    evaluate,
)
from contrastive_pretrain.models_dualtower import TaskHead  # noqa: E402
from contrastive_pretrain.models_cmr_v3 import CMREncoderV3  # noqa: E402

DATA_DIR = "/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3"
CKPT = ROOT / "contrastive_pretrain/checkpoints_cmr_v3/best.pth"
VERIFIED_JSON = ROOT / "contrastive_pretrain/checkpoints_cmr_v3/metrics_verified.json"
MEDSAM_CKPT = str(
    ROOT
    / "pretrained_weights/hf_cache/"
    / "models--flaviagiammarino--medsam-vit-base/blobs/"
    / "b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c"
)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    val_ds = CMRDatasetV3(VAL_CSV, DATA_DIR, is_train=False)
    val_loader = DataLoader(
        val_ds, batch_size=24, shuffle=False, num_workers=2, pin_memory=False
    )

    encoder = CMREncoderV3(
        proj_dim=256,
        embed_dim=768,
        spatial_pool=4,
        transformer_heads=8,
        transformer_depth=2,
        medsam_ckpt=MEDSAM_CKPT,
        freeze_backbone=False,
    ).to(device)
    head = TaskHead(in_dim=768, n_cls=len(CLS_COLS), n_reg=3).to(device)

    ckpt = torch.load(str(CKPT), map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["head"])
    ep = ckpt.get("epoch", "?")
    print(f"[loaded] {CKPT}  epoch_saved={ep}")

    metrics = evaluate(encoder, head, val_loader, device)
    print("\n=== Recomputed validation metrics (same code path as train_cmr_v3) ===")
    for k in sorted(metrics):
        print(f"  {k}: {metrics[k]:.6g}")

    emb = ckpt.get("metrics")
    if emb:
        print("\n=== Embedded ckpt['metrics'] (should match after 2026-05-03 patch) ===")
        for k in sorted(emb):
            print(f"  {k}: {emb[k]:.6g}")
        print("\n=== Absolute deltas (recomputed - embedded) ===")
        for k in metrics:
            if k not in emb:
                continue
            print(f"  {k}: {metrics[k] - emb[k]:+.6g}")

    if VERIFIED_JSON.is_file():
        doc = json.loads(VERIFIED_JSON.read_text())
        ref = doc.get("task1_cmr_val_n1801", {})
        if ref:
            print("\n=== metrics_verified.json task1_cmr_val_n1801 (reference snapshot) ===")
            for k in sorted(ref):
                print(f"  {k}: {ref[k]}")


if __name__ == "__main__":
    main()
