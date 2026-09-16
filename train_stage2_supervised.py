#!/usr/bin/env python3
"""Stage II: supervised CAD fine-tuning initialized from Stage I."""
from __future__ import annotations

import argparse
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepcad.data import FundusBinaryDataset
from deepcad.models import FundusBinaryClassifier, RETFoundFundusEncoder
from deepcad.training.common import (
    choose_device, dump_run_config, make_output_dir, save_checkpoint,
    trainable_parameters,
)
from deepcad.utils import safe_auroc, seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument(
        "--initial-checkpoint", required=True,
        help="A Stage I alignment checkpoint or a preceding Stage II checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--loss-type", choices=["paper_bce", "bce_pos_weight"],
                        default="paper_bce")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--positive-weight", type=float)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=12)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "none"],
                        default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def run_epoch(model, loader, device, optimizer, positive_weight, amp_dtype,
              loss_type, label_smoothing):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    labels, probabilities = [], []
    for batch in loader:
        images = batch["image"].to(device)
        target = batch["label"].to(device)
        enabled = device.type == "cuda" and amp_dtype != "none"
        dtype = torch.bfloat16 if amp_dtype == "bfloat16" else torch.float16
        with torch.set_grad_enabled(training), torch.autocast(
                device_type=device.type, dtype=dtype, enabled=enabled):
            logits = model(images)
            loss_target = target
            if loss_type == "paper_bce":
                loss_target = (target * (1.0 - label_smoothing)
                               + (1.0 - target) * label_smoothing)
            loss = F.binary_cross_entropy_with_logits(
                logits, loss_target,
                pos_weight=positive_weight if loss_type == "bce_pos_weight"
                else None)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * images.shape[0]
        total_items += images.shape[0]
        labels.append(target.detach().cpu().numpy())
        probabilities.append(logits.detach().sigmoid().cpu().numpy())
    labels_np = np.concatenate(labels)
    probabilities_np = np.concatenate(probabilities)
    return total_loss / max(total_items, 1), safe_auroc(labels_np, probabilities_np)


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    output = make_output_dir(args.output_dir, args.allow_existing)
    device = choose_device(args.device)
    dump_run_config(output, args)

    initial = torch.load(args.initial_checkpoint, map_location="cpu")
    encoder_config = initial.get(
        "encoder_config", {"adapter_dim": 64, "projection_dim": 128})
    encoder = RETFoundFundusEncoder(**encoder_config)
    if "fundus_encoder" in initial:
        encoder.load_state_dict(initial["fundus_encoder"], strict=True)
        model = FundusBinaryClassifier(encoder)
    elif "model" in initial:
        model = FundusBinaryClassifier(encoder)
        model.load_state_dict(initial["model"], strict=True)
    else:
        raise KeyError(
            "Initial checkpoint is neither Stage I nor Stage II format.")
    encoder.configure_trainable(args.unfreeze_last_blocks)
    model = model.to(device)

    train_dataset = FundusBinaryDataset(args.manifest, "train", True)
    validation_dataset = FundusBinaryDataset(args.manifest, "val", False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    positive_weight = None
    if args.loss_type == "bce_pos_weight":
        if args.positive_weight is not None:
            value = args.positive_weight
        else:
            labels = train_dataset.frame["label"].astype(int)
            value = float((labels == 0).sum()) / max(int((labels == 1).sum()), 1)
        positive_weight = torch.tensor(value, device=device)
    optimizer = torch.optim.AdamW(
        trainable_parameters([model]), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))

    best_auc = -math.inf
    stale = 0
    for epoch in range(args.epochs):
        train_loss, train_auc = run_epoch(
            model, train_loader, device, optimizer, positive_weight,
            args.amp_dtype, args.loss_type, args.label_smoothing)
        validation_loss, validation_auc = run_epoch(
            model, validation_loader, device, None, positive_weight,
            args.amp_dtype, args.loss_type, args.label_smoothing)
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} "
              f"train_auc={train_auc:.4f} val_loss={validation_loss:.6f} "
              f"val_auc={validation_auc:.4f}", flush=True)
        payload = dict(
            epoch=epoch + 1, model=model.state_dict(),
            encoder_config=encoder_config, validation_auc=validation_auc)
        save_checkpoint(output / "last.pt", **payload)
        if np.isfinite(validation_auc) and validation_auc > best_auc:
            best_auc = validation_auc
            stale = 0
            save_checkpoint(output / "best.pt", **payload)
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
