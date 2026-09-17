#!/usr/bin/env python3
"""Evaluate a frozen Stage II checkpoint on one named manifest split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

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
    parser.add_argument("--acknowledge-final-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.split == "test" and not args.acknowledge_final_test:
        raise RuntimeError(
            "Final test evaluation requires --acknowledge-final-test.")

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    encoder = RETFoundFundusEncoder(**checkpoint["encoder_config"])
    model = FundusBinaryClassifier(encoder)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    calibration_temperature = float(
        checkpoint.get("calibration_temperature", 1.0))
    if calibration_temperature <= 0:
        raise ValueError("Calibration temperature must be positive.")
    dataset = FundusBinaryDataset(args.manifest, args.split, False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    eids, labels, probabilities = [], [], []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            batch_labels = batch["label"].numpy()
            batch_probabilities = (
                logits / calibration_temperature).sigmoid().cpu().numpy()
            eids.extend(int(eid) for eid in batch["eid"].tolist())
            labels.extend(int(label) for label in batch_labels)
            probabilities.extend(float(value) for value in batch_probabilities)
    if len(set(eids)) != len(eids):
        raise RuntimeError(
            "Evaluation requires exactly one retinal photograph per participant.")
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    result = {"split": args.split, "n_participants": int(len(labels)),
              "image_policy": "one deterministic retinal photograph per participant",
              "calibration_temperature": calibration_temperature,
              "auroc": safe_auroc(labels, probabilities)}
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
