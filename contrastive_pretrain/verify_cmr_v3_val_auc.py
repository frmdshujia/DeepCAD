"""
Recompute validation AUC for checkpoints_cmr_v3/best.pth using the same
dataset + evaluate() path as train_cmr_v3.py. Run:

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
METRICS_JSON = ROOT / "contrastive_pretrain/checkpoints_cmr_v3/metrics_history.json"
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

    if METRICS_JSON.is_file():
        hist = json.loads(METRICS_JSON.read_text())
        row = next((r for r in hist if r.get("epoch") == ep), None)
        if row:
            print(f"\n=== metrics_history.json epoch {ep} (same epoch as best.pth) ===")
            for k in sorted(row):
                if k == "epoch":
                    continue
                print(f"  {k}: {row[k]}")
            print("\n=== Absolute deltas (recomputed - logged) ===")
            for k in metrics:
                if k not in row:
                    continue
                dv = metrics[k] - row[k]
                print(f"  {k}: {dv:+.6g}")
        else:
            print(f"\n[warn] no history row for epoch {ep}")


if __name__ == "__main__":
    main()
