"""Optional retinal-encoder forward/backward smoke test."""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default="0")
parser.add_argument("--checkpoint")
args = parser.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deepcad.models import RETFoundFundusEncoder


def main() -> None:
    model = RETFoundFundusEncoder(adapter_dim=64, projection_dim=128).cuda()
    if args.checkpoint:
        model.load_retfound_weights(args.checkpoint)
    model.configure_trainable(unfreeze_last_blocks=1)
    images = torch.randn(1, 3, 224, 224, device="cuda")
    projection, representation = model(images)
    assert projection.shape == (1, 128)
    assert representation.shape == (1, 1024)
    projection.sum().backward()
    print(
        f"fundus_encoder_smoke: PASS projection={tuple(projection.shape)} "
        f"representation={tuple(representation.shape)} "
        f"peak_gpu_gib={torch.cuda.max_memory_allocated() / 2**30:.3f}")


if __name__ == "__main__":
    main()
