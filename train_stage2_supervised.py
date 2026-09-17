#!/usr/bin/env python3
"""Stage II: supervised CAD fine-tuning initialized from Stage I."""
from __future__ import annotations

import argparse
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepcad.config import parse_args_with_config
from deepcad.data import FundusBinaryDataset, UniqueParticipantSampler
from deepcad.models import FundusBinaryClassifier, RETFoundFundusEncoder
from deepcad.training.common import (
    assert_manifest_schema, autocast_context, choose_device, dump_run_config,
    make_output_dir, save_checkpoint, trainable_parameters,
)
from deepcad.utils import safe_auroc, seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument(
        "--initial-checkpoint", required=True,
        help="A Stage I alignment checkpoint or a preceding Stage II checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--development-step", choices=["sdpp", "shcc"],
                        default="sdpp")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--loss-type", choices=["focal", "bce"], default="focal")
    parser.add_argument("--focal-alpha", type=float, default=0.25)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=12)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--calibrate", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Fit temperature scaling on the validation split")
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "none"],
                        default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-existing", action="store_true")
    return parse_args_with_config(parser)


def supervised_loss(logits, target, loss_type, focal_alpha, focal_gamma):
    if loss_type == "bce":
        return F.binary_cross_entropy_with_logits(logits, target)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = torch.sigmoid(logits)
    probability_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = focal_alpha * target + (1.0 - focal_alpha) * (1.0 - target)
    return (alpha_t * (1.0 - probability_t).pow(focal_gamma) * bce).mean()


def run_epoch(model, loader, device, optimizer, amp_dtype, loss_type,
              focal_alpha, focal_gamma):
    training = optimizer is not None
    model.train(training)
    total_loss, total_items = 0.0, 0
    labels, probabilities, eids = [], [], []
    for batch in loader:
        images = batch["image"].to(device)
        target = batch["label"].to(device)
        with torch.set_grad_enabled(training), autocast_context(
                device, amp_dtype):
            logits = model(images)
            loss = supervised_loss(
                logits, target, loss_type, focal_alpha, focal_gamma)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach()) * images.shape[0]
        total_items += images.shape[0]
        labels.append(target.detach().cpu().numpy())
        probabilities.append(logits.detach().sigmoid().cpu().numpy())
        eids.extend(int(eid) for eid in batch["eid"].tolist())
    labels_np = np.concatenate(labels)
    probabilities_np = np.concatenate(probabilities)
    if len(set(eids)) != len(eids):
        raise RuntimeError("Each epoch must contain at most one image per participant.")
    return total_loss / max(total_items, 1), safe_auroc(labels_np, probabilities_np)


@torch.no_grad()
def collect_logits(model, loader, device, amp_dtype):
    model.eval()
    logits, labels = [], []
    for batch in loader:
        with autocast_context(device, amp_dtype):
            output = model(batch["image"].to(device))
        logits.append(output.float().cpu())
        labels.append(batch["label"].float().cpu())
    return torch.cat(logits), torch.cat(labels)


def fit_temperature(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Fit one positive scalar temperature on validation logits only."""
    log_temperature = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.LBFGS(
        [log_temperature], lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.exp().clamp(0.05, 20.0)
        loss = F.binary_cross_entropy_with_logits(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().exp().clamp(0.05, 20.0))


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    assert_manifest_schema(
        args.manifest, ["fundus_path", "label"],
        allow_repeated_within_split=True)
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
    train_sampler = UniqueParticipantSampler(train_dataset, seed=args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    optimizer = torch.optim.AdamW(
        trainable_parameters([model]), lr=args.learning_rate,
        weight_decay=args.weight_decay)

    best_auc = -math.inf
    stale = 0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_loss, train_auc = run_epoch(
            model, train_loader, device, optimizer, args.amp_dtype,
            args.loss_type, args.focal_alpha, args.focal_gamma)
        validation_loss, validation_auc = run_epoch(
            model, validation_loader, device, None, args.amp_dtype,
            args.loss_type, args.focal_alpha, args.focal_gamma)
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} "
              f"train_auc={train_auc:.4f} val_loss={validation_loss:.6f} "
              f"val_auc={validation_auc:.4f}", flush=True)
        payload = dict(
            epoch=epoch + 1, model=model.state_dict(),
            encoder_config=encoder_config, validation_auc=validation_auc,
            development_step=args.development_step,
            calibration_temperature=1.0)
        save_checkpoint(output / "last.pt", **payload)
        if np.isfinite(validation_auc) and validation_auc > best_auc:
            best_auc = validation_auc
            stale = 0
            save_checkpoint(output / "best.pt", **payload)
        else:
            stale += 1
        if stale >= args.patience:
            break

    if args.calibrate:
        best_path = output / "best.pt"
        if not best_path.exists():
            raise RuntimeError("No finite-AUROC checkpoint is available to calibrate.")
        best_checkpoint = torch.load(best_path, map_location="cpu")
        model.load_state_dict(best_checkpoint["model"], strict=True)
        validation_logits, validation_labels = collect_logits(
            model, validation_loader, device, args.amp_dtype)
        temperature = fit_temperature(validation_logits, validation_labels)
        best_checkpoint["calibration_temperature"] = temperature
        save_checkpoint(best_path, **best_checkpoint)
        (output / "calibration.json").write_text(json.dumps({
            "method": "temperature_scaling",
            "source_split": "val",
            "temperature": temperature,
        }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
