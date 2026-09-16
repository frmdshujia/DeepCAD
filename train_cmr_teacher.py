#!/usr/bin/env python3
"""Train the supervised 16-frame CMR teacher used before Stage I alignment."""
from __future__ import annotations

import argparse
import math

import torch
from torch.utils.data import DataLoader

from deepcad.data import CMRDataset
from deepcad.losses import masked_multitask_loss
from deepcad.models import CMREncoderV4, MultiTaskHead
from deepcad.training.common import (
    autocast_context, choose_device, classification_pos_weights, dump_run_config,
    make_output_dir, parse_columns, regression_statistics, save_checkpoint,
    trainable_parameters,
)
from deepcad.utils import seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backbone-checkpoint", required=True)
    parser.add_argument("--backbone", default="medsam2_heart")
    parser.add_argument("--fusion-mode", default="hierarchical",
                        choices=["hierarchical", "sax_t1", "global_only"])
    parser.add_argument("--classification-columns", default="")
    parser.add_argument("--regression-columns", default="")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--classification-weight", type=float, default=0.5)
    parser.add_argument("--regression-weight", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "none"],
                        default="bfloat16")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-existing", action="store_true")
    return parser.parse_args()


def run_epoch(model, head, loader, device, optimizer, reg_mean, reg_std,
              pos_weight, classification_weight: float,
              regression_weight: float, amp_dtype: str) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    head.train(training)
    total_loss = 0.0
    total_classification = 0.0
    total_regression = 0.0
    total_items = 0
    for batch in loader:
        cmr = batch["cmr"].to(device)
        t1_available = batch["t1_available"].to(device)
        cls_target = batch["classification_targets"].to(device)
        reg_target = batch["regression_targets"].to(device)
        cls_mask = batch["classification_mask"].to(device)
        reg_mask = batch["regression_mask"].to(device)
        if reg_target.numel():
            reg_target = (reg_target - reg_mean) / reg_std

        with torch.set_grad_enabled(training), autocast_context(
                device, amp_dtype):
            _, features = model(cmr, t1_available=t1_available)
            cls_logits, reg_prediction = head(features)
            loss, classification_loss, regression_loss = masked_multitask_loss(
                cls_logits, reg_prediction, cls_target, reg_target,
                cls_mask, reg_mask,
                classification_pos_weight=pos_weight,
                classification_weight=classification_weight,
                regression_weight=regression_weight,
                return_components=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        batch_size = cmr.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_classification += float(classification_loss.detach()) * batch_size
        total_regression += float(regression_loss.detach()) * batch_size
        total_items += batch_size
    denominator = max(total_items, 1)
    return {
        "total": total_loss / denominator,
        "classification": total_classification / denominator,
        "regression": total_regression / denominator,
    }


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    output = make_output_dir(args.output_dir, args.allow_existing)
    device = choose_device(args.device)
    classification_columns = parse_columns(args.classification_columns)
    regression_columns = parse_columns(args.regression_columns)
    if not classification_columns and not regression_columns:
        raise ValueError("At least one supervised target column is required.")

    train_dataset = CMRDataset(
        args.manifest, "train", classification_columns, regression_columns, True)
    validation_dataset = CMRDataset(
        args.manifest, "val", classification_columns, regression_columns, False)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=device.type == "cuda")
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=device.type == "cuda")

    model_config = {
        "backbone": args.backbone,
        "backbone_ckpt": args.backbone_checkpoint,
        "fusion_mode": args.fusion_mode,
        "proj_dim": 256,
        "embed_dim": 768,
        "grad_checkpoint": args.gradient_checkpointing,
    }
    model = CMREncoderV4(**model_config).to(device)
    head = MultiTaskHead(
        768, len(classification_columns), len(regression_columns)).to(device)
    optimizer = torch.optim.AdamW(
        trainable_parameters([model, head]), lr=args.learning_rate,
        weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(args.epochs, 1))
    mean_np, std_np = regression_statistics(args.manifest, regression_columns)
    pos_weight_np = classification_pos_weights(
        args.manifest, classification_columns)
    reg_mean = torch.as_tensor(mean_np, device=device)
    reg_std = torch.as_tensor(std_np, device=device)
    pos_weight = torch.as_tensor(pos_weight_np, device=device)
    dump_run_config(output, args, {
        "classification_columns_parsed": classification_columns,
        "regression_columns_parsed": regression_columns,
        "regression_mean": mean_np.tolist(),
        "regression_std": std_np.tolist(),
        "classification_pos_weight": pos_weight_np.tolist(),
    })

    best = math.inf
    epochs_without_improvement = 0
    for epoch in range(args.epochs):
        train_metrics = run_epoch(
            model, head, train_loader, device, optimizer, reg_mean, reg_std,
            pos_weight, args.classification_weight, args.regression_weight,
            args.amp_dtype)
        validation_metrics = run_epoch(
            model, head, validation_loader, device, None, reg_mean, reg_std,
            pos_weight, args.classification_weight, args.regression_weight,
            args.amp_dtype)
        validation_loss = validation_metrics["total"]
        print(
            f"epoch={epoch + 1} "
            f"train_total={train_metrics['total']:.6f} "
            f"train_cls={train_metrics['classification']:.6f} "
            f"train_reg={train_metrics['regression']:.6f} "
            f"val_total={validation_metrics['total']:.6f} "
            f"val_cls={validation_metrics['classification']:.6f} "
            f"val_reg={validation_metrics['regression']:.6f}", flush=True)
        payload = dict(
            epoch=epoch + 1, encoder=model.state_dict(), head=head.state_dict(),
            model_config=model_config,
            classification_columns=classification_columns,
            regression_columns=regression_columns,
            regression_mean=mean_np, regression_std=std_np,
            classification_pos_weight=pos_weight_np,
            validation_components=validation_metrics,
            validation_loss=validation_loss)
        save_checkpoint(output / "last.pt", **payload)
        if validation_loss < best:
            best = validation_loss
            epochs_without_improvement = 0
            save_checkpoint(output / "best.pt", **payload)
        else:
            epochs_without_improvement += 1
        scheduler.step()
        if epochs_without_improvement >= args.patience:
            break


if __name__ == "__main__":
    main()
