#!/usr/bin/env python3
"""Evaluate Stage I using one prespecified image per participant."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from deepcad.data import Stage1ContrastiveDataset
from deepcad.manifest import read_manifest
from deepcad.models import RETFoundFundusEncoder
from deepcad.training.alignment import (
    alignment_metrics, collect_participant_embeddings,
)
from deepcad.training.common import choose_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--cmr-embeddings", required=True)
    parser.add_argument("--cmr-eids", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--teacher-manifest", nargs="+")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--acknowledge-final-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "none"],
                        default="bfloat16")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.split == "test" and not args.acknowledge_final_test:
        raise RuntimeError(
            "Final test evaluation requires --acknowledge-final-test.")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    fundus_encoder = RETFoundFundusEncoder(**checkpoint["encoder_config"])
    fundus_encoder.load_state_dict(checkpoint["fundus_encoder"], strict=True)
    cmr_projector = nn.Sequential(
        nn.Linear(768, 512), nn.GELU(),
        nn.Linear(512, checkpoint["encoder_config"]["projection_dim"]))
    cmr_projector.load_state_dict(checkpoint["cmr_projector"], strict=True)
    fundus_encoder.to(device).eval()
    cmr_projector.to(device).eval()

    dataset = Stage1ContrastiveDataset(
        args.manifest, args.split, args.cmr_embeddings, args.cmr_eids, False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    eids, fundus, cmr, t1_available = collect_participant_embeddings(
        fundus_encoder, cmr_projector, loader, device, args.amp_dtype)
    result = {
        "split": args.split,
        "image_policy": "one deterministic retinal photograph per participant",
        "overall": alignment_metrics(
            fundus, cmr, checkpoint["logit_scale"]),
        "t1_present": alignment_metrics(
            fundus[t1_available], cmr[t1_available],
            checkpoint["logit_scale"]),
        "t1_missing": alignment_metrics(
            fundus[~t1_available], cmr[~t1_available],
            checkpoint["logit_scale"]),
    }
    if args.teacher_manifest:
        teacher_eids = set(
            read_manifest(args.teacher_manifest)["eid"].astype(int).tolist())
        unseen = np.asarray([int(eid) not in teacher_eids for eid in eids])
        result["teacher_cohort_unseen"] = alignment_metrics(
            fundus[unseen], cmr[unseen], checkpoint["logit_scale"])
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite evaluation result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
