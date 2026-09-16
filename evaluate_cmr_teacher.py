#!/usr/bin/env python3
"""Evaluate a frozen CMR teacher checkpoint on one named split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score
from torch.utils.data import DataLoader

from deepcad.data import CMRDataset
from deepcad.models import CMREncoderV4, MultiTaskHead
from deepcad.training.common import choose_device
from deepcad.utils import safe_auroc


def calculate_metrics(
    classification_columns,
    regression_columns,
    logits,
    classification_targets,
    classification_masks,
    regression_predictions,
    regression_targets,
    regression_masks,
    sample_mask,
):
    result = {
        "n": int(sample_mask.sum()),
        "classification": {},
        "regression": {},
        "macro": {},
    }
    aucs = []
    for index, name in enumerate(classification_columns):
        valid = classification_masks[:, index] & sample_mask
        probabilities = 1.0 / (1.0 + np.exp(-logits[valid, index]))
        auc = safe_auroc(classification_targets[valid, index], probabilities)
        result["classification"][name] = {
            "n": int(valid.sum()), "auroc": auc}
        if np.isfinite(auc):
            aucs.append(auc)
    regression_summaries = {"mae": [], "r2": [], "pearson_r": []}
    for index, name in enumerate(regression_columns):
        valid = regression_masks[:, index] & sample_mask
        observed = regression_targets[valid, index]
        estimated = regression_predictions[valid, index]
        if len(observed) < 2:
            metrics = {"n": int(len(observed)), "mae": float("nan"),
                       "r2": float("nan"), "pearson_r": float("nan")}
        else:
            metrics = {
                "n": int(len(observed)),
                "mae": float(mean_absolute_error(observed, estimated)),
                "r2": float(r2_score(observed, estimated)),
                "pearson_r": float(np.corrcoef(observed, estimated)[0, 1]),
            }
        result["regression"][name] = metrics
        for metric in regression_summaries:
            if np.isfinite(metrics[metric]):
                regression_summaries[metric].append(metrics[metric])
    result["macro"] = {
        "auroc": float(np.mean(aucs)) if aucs else float("nan"),
        **{
            metric: float(np.mean(values)) if values else float("nan")
            for metric, values in regression_summaries.items()
        },
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--acknowledge-final-test", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if args.split == "test" and not args.acknowledge_final_test:
        raise RuntimeError(
            "Final test evaluation requires --acknowledge-final-test.")

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

    cls_logits, cls_targets, cls_masks, t1_flags = [], [], [], []
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
            t1_flags.append(batch["t1_available"].numpy())
            reg_predictions.append(reg.cpu().numpy())
            reg_targets.append(batch["regression_targets"].numpy())
            reg_masks.append(batch["regression_mask"].numpy())

    logits = np.concatenate(cls_logits)
    classification_targets = np.concatenate(cls_targets)
    classification_masks = np.concatenate(cls_masks)
    regression_predictions = np.concatenate(reg_predictions)
    regression_targets = np.concatenate(reg_targets)
    regression_masks = np.concatenate(reg_masks)
    t1_available = np.concatenate(t1_flags).astype(bool)
    mean = np.asarray(checkpoint["regression_mean"])
    std = np.asarray(checkpoint["regression_std"])
    regression_predictions = regression_predictions * std[None] + mean[None]
    all_samples = np.ones(len(dataset), dtype=bool)
    metric_args = (
        classification_columns, regression_columns, logits,
        classification_targets, classification_masks,
        regression_predictions, regression_targets, regression_masks)
    result = {
        "split": args.split,
        "overall": calculate_metrics(*metric_args, all_samples),
        "t1_present": calculate_metrics(*metric_args, t1_available),
        "t1_missing": calculate_metrics(*metric_args, ~t1_available),
    }
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
