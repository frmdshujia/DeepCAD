#!/usr/bin/env python3
"""Evaluate a frozen Stage II checkpoint on one named manifest split."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepcad.data import FundusBinaryDataset
from deepcad.models import FundusBinaryClassifier, RETFoundFundusEncoder
from deepcad.training.common import choose_device
from deepcad.utils import safe_auroc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoder = RETFoundFundusEncoder(**checkpoint["encoder_config"])
    model = FundusBinaryClassifier(encoder)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    dataset = FundusBinaryDataset(args.manifest, args.split, False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    labels, probabilities = [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            labels.append(batch["label"].numpy())
            probabilities.append(logits.sigmoid().cpu().numpy())
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    result = {"split": args.split, "n": int(len(labels)),
              "auroc": safe_auroc(labels, probabilities)}
    with open(args.output_json, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
