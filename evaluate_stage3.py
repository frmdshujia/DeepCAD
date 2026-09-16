#!/usr/bin/env python3
"""Evaluate the frozen Stage III clinical-fusion model."""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from deepcad.manifest import read_manifest
from deepcad.models import ClinicalRiskMLP
from deepcad.training.common import choose_device
from deepcad.utils import safe_auroc


def operating_metrics(labels, probabilities, threshold):
    prediction = probabilities >= threshold
    tp = np.sum(prediction & (labels == 1))
    fn = np.sum(~prediction & (labels == 1))
    tn = np.sum(~prediction & (labels == 0))
    fp = np.sum(prediction & (labels == 0))
    return {
        "threshold": float(threshold),
        "sensitivity": float(tp / max(tp + fn, 1)),
        "specificity": float(tn / max(tn + fp, 1)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--validation-thresholds")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model = ClinicalRiskMLP(
        input_dim=len(checkpoint["columns"]),
        hidden_dims=tuple(checkpoint["hidden_dims"]),
        dropout=float(checkpoint["dropout"]))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    frame = read_manifest(args.manifest)
    frame = frame.loc[frame["split"] == args.split]
    values = frame[checkpoint["columns"]].apply(
        pd.to_numeric, errors="coerce").to_numpy(np.float32)
    values = np.where(
        np.isfinite(values), values, np.asarray(checkpoint["medians"]))
    values = ((values - np.asarray(checkpoint["means"]))
              / np.asarray(checkpoint["stds"]))
    with torch.no_grad():
        probabilities = model(
            torch.from_numpy(values.astype(np.float32)).to(device)
        ).sigmoid().cpu().numpy()
    labels = frame["label"].to_numpy(np.int64)
    result = {"split": args.split, "n": int(len(labels)),
              "auroc": safe_auroc(labels, probabilities)}
    if args.validation_thresholds:
        with open(args.validation_thresholds) as handle:
            thresholds = json.load(handle)
        if thresholds.get("source_split") != "val":
            raise ValueError("Operating thresholds must originate from val.")
        result["operating_points"] = {
            name: operating_metrics(labels, probabilities, value)
            for name, value in thresholds.items() if name != "source_split"}
    with open(args.output_json, "w") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
