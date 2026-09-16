#!/usr/bin/env python3
"""Evaluate a frozen Stage II checkpoint on one named manifest split."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
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
    dataset = FundusBinaryDataset(args.manifest, args.split, False)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    labels_by_eid = {}
    probabilities_by_eid = defaultdict(list)
    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            batch_labels = batch["label"].numpy()
            batch_probabilities = logits.sigmoid().cpu().numpy()
            for eid, label, probability in zip(
                    batch["eid"].tolist(), batch_labels, batch_probabilities):
                eid = int(eid)
                if eid in labels_by_eid and labels_by_eid[eid] != int(label):
                    raise ValueError(f"Conflicting labels for EID {eid}")
                labels_by_eid[eid] = int(label)
                probabilities_by_eid[eid].append(float(probability))
    eids = sorted(labels_by_eid)
    labels = np.asarray([labels_by_eid[eid] for eid in eids])
    probabilities = np.asarray([
        np.mean(probabilities_by_eid[eid]) for eid in eids])
    result = {"split": args.split, "n_participants": int(len(labels)),
              "aggregation": "mean eye-level probability within EID",
              "auroc": safe_auroc(labels, probabilities)}
    output = Path(args.output_json)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation result: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
