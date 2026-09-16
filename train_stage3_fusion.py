#!/usr/bin/env python3
"""Stage III: combine retinal probability with clinical risk factors."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from deepcad.models import ClinicalRiskMLP
from deepcad.manifest import read_manifest
from deepcad.training.common import (
    choose_device, dump_run_config, make_output_dir, parse_columns,
    save_checkpoint,
)
from deepcad.utils import safe_auroc, seed_everything


def prepare_features(frame, columns, medians, means, stds):
    values = frame[columns].apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    values = np.where(np.isfinite(values), values, medians)
    return ((values - means) / stds).astype(np.float32)


def make_loader(frame, columns, medians, means, stds, batch_size, shuffle):
    features = prepare_features(frame, columns, medians, means, stds)
    labels = pd.to_numeric(frame["label"], errors="raise").to_numpy(np.float32)
    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, device, optimizer):
    training = optimizer is not None
    model.train(training)
    losses, labels, probabilities = [], [], []
    for features, target in loader:
        features, target = features.to(device), target.to(device)
        with torch.set_grad_enabled(training):
            logits = model(features)
            loss = F.binary_cross_entropy_with_logits(logits, target)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach()) * len(target))
        labels.append(target.detach().cpu().numpy())
        probabilities.append(logits.detach().sigmoid().cpu().numpy())
    labels = np.concatenate(labels)
    probabilities = np.concatenate(probabilities)
    return (sum(losses) / len(labels), safe_auroc(labels, probabilities),
            labels, probabilities)


def operating_threshold(labels, probabilities, metric, target):
    candidates = np.unique(probabilities)
    best_threshold, best_gap = 0.5, math.inf
    for threshold in candidates:
        predicted = probabilities >= threshold
        tp = np.sum(predicted & (labels == 1))
        fn = np.sum(~predicted & (labels == 1))
        tn = np.sum(~predicted & (labels == 0))
        fp = np.sum(predicted & (labels == 0))
        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        value = specificity if metric == "specificity" else sensitivity
        gap = abs(value - target)
        if gap < best_gap:
            best_gap, best_threshold = gap, float(threshold)
    return best_threshold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--feature-columns",
        default="retinal_score,age,sex,hypertension,diabetes,smoking,"
                "dyslipidemia,family_history")
    parser.add_argument("--hidden-dims", default="32,16")
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    seed_everything(args.seed)
    output = make_output_dir(args.output_dir, args.allow_existing)
    device = choose_device(args.device)
    columns = parse_columns(args.feature_columns)
    hidden_dims = tuple(int(value) for value in parse_columns(args.hidden_dims))
    frame = read_manifest(args.manifest)
    train_frame = frame.loc[frame["split"] == "train"].copy()
    validation_frame = frame.loc[frame["split"] == "val"].copy()

    train_values = train_frame[columns].apply(
        pd.to_numeric, errors="coerce").to_numpy(np.float32)
    medians = np.nanmedian(train_values, axis=0)
    imputed = np.where(np.isfinite(train_values), train_values, medians)
    means = imputed.mean(axis=0)
    stds = imputed.std(axis=0)
    stds[stds == 0] = 1
    train_loader = make_loader(
        train_frame, columns, medians, means, stds, args.batch_size, True)
    validation_loader = make_loader(
        validation_frame, columns, medians, means, stds, args.batch_size, False)
    model = ClinicalRiskMLP(
        input_dim=len(columns), hidden_dims=hidden_dims,
        dropout=args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    dump_run_config(output, args, {
        "feature_columns_parsed": columns,
        "imputation_medians": medians.tolist(),
        "standardization_means": means.tolist(),
        "standardization_stds": stds.tolist(),
    })

    best_auc, stale = -math.inf, 0
    for epoch in range(args.epochs):
        train_loss, train_auc, _, _ = run_epoch(
            model, train_loader, device, optimizer)
        val_loss, val_auc, val_labels, val_probabilities = run_epoch(
            model, validation_loader, device, None)
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} "
              f"train_auc={train_auc:.4f} val_loss={val_loss:.6f} "
              f"val_auc={val_auc:.4f}", flush=True)
        payload = dict(
            epoch=epoch + 1, model=model.state_dict(), columns=columns,
            hidden_dims=hidden_dims, dropout=args.dropout,
            medians=medians, means=means, stds=stds,
            validation_auc=val_auc)
        save_checkpoint(output / "last.pt", **payload)
        if np.isfinite(val_auc) and val_auc > best_auc:
            best_auc, stale = val_auc, 0
            save_checkpoint(output / "best.pt", **payload)
            thresholds = {
                "source_split": "val",
                "high_specificity_95": operating_threshold(
                    val_labels, val_probabilities, "specificity", 0.95),
                "high_sensitivity_90": operating_threshold(
                    val_labels, val_probabilities, "sensitivity", 0.90),
            }
            (output / "validation_thresholds.json").write_text(
                json.dumps(thresholds, indent=2, sort_keys=True))
        else:
            stale += 1
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
