#!/usr/bin/env python3
"""Stage I: align retinal images with frozen CMR teacher embeddings by InfoNCE."""
from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from deepcad.config import parse_args_with_config
from deepcad.data import Stage1ContrastiveDataset, UniqueParticipantSampler
from deepcad.losses import symmetric_info_nce
from deepcad.models import RETFoundFundusEncoder
from deepcad.training.common import (
    assert_disjoint_participants, assert_manifest_schema, autocast_context,
    choose_device, dump_run_config, make_output_dir, save_checkpoint,
    warmup_cosine_scheduler,
)
from deepcad.training.alignment import (
    alignment_metrics, collect_participant_embeddings,
)
from deepcad.utils import seed_everything


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--cmr-embeddings", required=True)
    parser.add_argument("--cmr-eids", required=True)
    parser.add_argument("--retfound-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-validation-manifest", nargs="+")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--new-module-learning-rate", "--learning-rate",
                        dest="new_module_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--unfreeze-last-blocks", type=int, default=12)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--min-learning-rate-ratio", type=float, default=0.01)
    parser.add_argument(
        "--require-t1", action="store_true",
        help="Sensitivity analysis restricted to participants with observed T1")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--amp-dtype", choices=["bfloat16", "float16", "none"],
                        default="bfloat16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-existing", action="store_true")
    return parse_args_with_config(parser)


def run_training_epoch(fundus_encoder, cmr_projector, temperature, loader,
                       sampler, device, optimizer, epoch: int,
                       amp_dtype: str) -> float:
    fundus_encoder.train(True)
    cmr_projector.train(True)
    sampler.set_epoch(epoch)
    total_loss = 0.0
    total_items = 0
    for batch in loader:
        # One image per participant prevents within-participant false negatives.
        if len(set(batch["eid"].tolist())) != len(batch["eid"]):
            raise RuntimeError("Duplicate participant within an InfoNCE batch.")
        images = batch["image"].to(device)
        cmr = batch["cmr_embedding"].to(device)
        with autocast_context(device, amp_dtype):
            fundus_projection, _ = fundus_encoder(images)
            cmr_projection = F.normalize(cmr_projector(cmr), dim=-1)
            loss = symmetric_info_nce(
                fundus_projection, cmr_projection, temperature)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        count = images.shape[0]
        total_loss += float(loss.detach()) * count
        total_items += count
    return total_loss / max(total_items, 1)


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    assert_manifest_schema(
        args.manifest, ["fundus_path", "t1_available"],
        allow_repeated_within_split=True)
    assert_disjoint_participants(
        args.manifest, args.external_validation_manifest)
    output = make_output_dir(args.output_dir, args.allow_existing)
    device = choose_device(args.device)
    dump_run_config(output, args)

    train_dataset = Stage1ContrastiveDataset(
        args.manifest, "train", args.cmr_embeddings, args.cmr_eids, True,
        require_t1=args.require_t1)
    validation_dataset = Stage1ContrastiveDataset(
        args.manifest, "val", args.cmr_embeddings, args.cmr_eids, False,
        require_t1=args.require_t1)
    train_sampler = UniqueParticipantSampler(train_dataset, seed=args.seed)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=device.type == "cuda", drop_last=True)
    validation_loader = DataLoader(
        validation_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda")

    fundus_encoder = RETFoundFundusEncoder(
        adapter_dim=args.adapter_dim, projection_dim=args.projection_dim)
    fundus_encoder.load_retfound_weights(args.retfound_checkpoint)
    fundus_encoder.configure_trainable(args.unfreeze_last_blocks)
    cmr_projector = nn.Sequential(
        nn.Linear(768, 512), nn.GELU(),
        nn.Linear(512, args.projection_dim))
    logit_scale = torch.tensor(1.0 / args.temperature, device=device).log()
    fundus_encoder.to(device)
    cmr_projector.to(device)
    backbone_parameters = [
        parameter for parameter in fundus_encoder.backbone.parameters()
        if parameter.requires_grad]
    new_module_parameters = (
        list(fundus_encoder.adapters.parameters())
        + list(fundus_encoder.pool_norm.parameters())
        + list(fundus_encoder.projection.parameters())
        + list(cmr_projector.parameters()))
    optimizer = torch.optim.AdamW([
        {"name": "retfound_backbone", "params": backbone_parameters,
         "lr": args.backbone_learning_rate},
        {"name": "adapters_and_projection", "params": new_module_parameters,
         "lr": args.new_module_learning_rate},
    ], weight_decay=args.weight_decay)
    scheduler = warmup_cosine_scheduler(
        optimizer, args.warmup_epochs, args.epochs,
        args.min_learning_rate_ratio)

    best = math.inf
    stale = 0
    for epoch in range(args.epochs):
        train_loss = run_training_epoch(
            fundus_encoder, cmr_projector, args.temperature, train_loader,
            train_sampler, device, optimizer, epoch, args.amp_dtype)
        _, val_fundus, val_cmr, _ = collect_participant_embeddings(
            fundus_encoder, cmr_projector, validation_loader, device,
            args.amp_dtype)
        validation_metrics = alignment_metrics(
            val_fundus, val_cmr, logit_scale)
        validation_loss = validation_metrics["infonce"]
        print(f"epoch={epoch + 1} train_loss={train_loss:.6f} "
              f"val_loss={validation_loss:.6f} "
              f"val_f2c_top1={validation_metrics['fundus_to_cmr_top1']:.4f} "
              f"temperature={args.temperature:.4f}",
              flush=True)
        payload = dict(
            epoch=epoch + 1,
            fundus_encoder=fundus_encoder.state_dict(),
            cmr_projector=cmr_projector.state_dict(),
            logit_scale=logit_scale.detach().cpu(),
            temperature=args.temperature,
            validation_loss=validation_loss,
            validation_metrics=validation_metrics,
            require_t1=args.require_t1,
            encoder_config={"adapter_dim": args.adapter_dim,
                            "projection_dim": args.projection_dim})
        save_checkpoint(output / "last.pt", **payload)
        if validation_loss < best:
            best = validation_loss
            stale = 0
            save_checkpoint(output / "best.pt", **payload)
        else:
            stale += 1
        scheduler.step()
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
