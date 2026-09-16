"""Optional end-to-end CMR forward test with a real Hiera checkpoint and array."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--backbone-checkpoint", required=True)
parser.add_argument("--cmr-array", required=True)
parser.add_argument("--gpu", default="0")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepcad.models import CMREncoderV4


def main() -> None:
    cmr = np.load(args.cmr_array).astype(np.float32)
    if cmr.shape != (16, 224, 224):
        raise ValueError(f"Expected (16, 224, 224), got {cmr.shape}")
    t1_available = bool(np.max(np.abs(cmr[15])) > 1e-8)
    model = CMREncoderV4(
        backbone="medsam2_heart",
        backbone_ckpt=args.backbone_checkpoint,
        fusion_mode="hierarchical").cuda().eval()
    tensor = torch.from_numpy(cmr)[None, :, None].cuda()
    with torch.no_grad():
        projection, representation = model(
            tensor, t1_available=torch.tensor([t1_available], device="cuda"))
    assert projection.shape == (1, 256)
    assert representation.shape == (1, 768)
    print(
        f"real_backbone_smoke: PASS projection={tuple(projection.shape)} "
        f"representation={tuple(representation.shape)} "
        f"t1_available={t1_available} "
        f"peak_gpu_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")


if __name__ == "__main__":
    main()
