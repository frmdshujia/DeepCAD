#!/usr/bin/env python3
"""Evaluate a frozen CMR teacher checkpoint on one named split."""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from deepcad.data import CMRDataset
from deepcad.models import CMREncoderV4, MultiTaskHead
from deepcad.training.common import choose_device
from deepcad.utils import safe_auroc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    config = dict(checkpoint["model_config"])
    config["backbone_ckpt"] = None
    model = CMREncoderV4(**config)
    model.load_state_dict(checkpoint["encoder"], strict=True)
    classification_columns = checkpoint["classification_columns"]
    regression_columns = checkpoint["regression_columns"]
    head = MultiTaskHead(
        768, len(classification_columns), len(regression_columns))
    head.load_state_dict(checkpoint["head"], strict=True)
    model.to(device).eval()
    head.to(device).eval()
    dataset = CMRDataset(
        args.manifest, args.split, classification_columns,
        regression_columns, False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")

    cls_logits, cls_targets, cls_masks = [], [], []
    reg_predictions, reg_targets, reg_masks = [], [], []
    with torch.no_grad():
        for batch in loader:
            _, features = model(
                batch["cmr"].to(device),
                t1_available=batch["t1_available"].to(device))
            cls, reg = head(features)
            cls_logits.append(cls.cpu().numpy())
            cls_targets.append(batch["classification_targets"].numpy())
            cls_masks.append(batch["classification_mask"].numpy())
            reg_predictions.append(reg.cpu().numpy())
            reg_targets.append(batch["regression_targets"].numpy())
            reg_masks.append(batch["regression_mask"].numpy())

    result = {"split": args.split, "n": len(dataset),
              "classification": {}, "regression": {}}
    if classification_columns:
        logits = np.concatenate(cls_logits)
        targets = np.concatenate(cls_targets)
        masks = np.concatenate(cls_masks)
        for index, name in enumerate(classification_columns):
            valid = masks[:, index]
            probabilities = 1.0 / (1.0 + np.exp(-logits[valid, index]))
            result["classification"][name] = {
                "n": int(valid.sum()),
                "auroc": safe_auroc(targets[valid, index], probabilities),
            }
    if regression_columns:
        prediction = np.concatenate(reg_predictions)
        target = np.concatenate(reg_targets)
        masks = np.concatenate(reg_masks)
        mean = np.asarray(checkpoint["regression_mean"])
        std = np.asarray(checkpoint["regression_std"])
        prediction = prediction * std[None] + mean[None]
        for index, name in enumerate(regression_columns):
            valid = masks[:, index]
            observed = target[valid, index]
            estimated = prediction[valid, index]
            correlation = (
                float(np.corrcoef(observed, estimated)[0, 1])
                if len(observed) > 1 else float("nan"))
            result["regression"][name] = {
                "n": int(valid.sum()),
                "mae": float(mean_absolute_error(observed, estimated)),
                "r2": float(r2_score(observed, estimated)),
                "pearson_r": correlation,
            }
    with open(args.output_json, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
