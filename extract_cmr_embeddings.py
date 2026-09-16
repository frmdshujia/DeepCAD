#!/usr/bin/env python3
"""Extract frozen 768-dimensional CMR teacher embeddings for Stage I."""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from deepcad.data import CMRDataset
from deepcad.models import CMREncoderV4
from deepcad.manifest import read_manifest
from deepcad.training.common import choose_device, make_output_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, nargs="+")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split", default="all")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()

    output = make_output_dir(args.output_dir, args.allow_existing)
    device = choose_device(args.device)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    model_config = dict(checkpoint["model_config"])
    # The complete trained backbone is in the encoder state dictionary.
    model_config["backbone_ckpt"] = None
    model = CMREncoderV4(**model_config)
    model.load_state_dict(checkpoint["encoder"], strict=True)
    model.to(device).eval()

    frame = read_manifest(args.manifest)
    splits = sorted(frame["split"].unique()) if args.split == "all" else [args.split]
    all_embeddings, all_eids = [], []
    for split in splits:
        dataset = CMRDataset(args.manifest, split, [], [], False)
        loader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=device.type == "cuda")
        with torch.no_grad():
            for batch in loader:
                _, embedding = model(
                    batch["cmr"].to(device),
                    t1_available=batch["t1_available"].to(device))
                all_embeddings.append(embedding.cpu().numpy())
                all_eids.append(batch["eid"].numpy())
    embeddings = np.concatenate(all_embeddings).astype(np.float32)
    eids = np.concatenate(all_eids).astype(np.int64)
    np.save(output / "cmr_teacher_embeddings.npy", embeddings)
    np.save(output / "cmr_teacher_eids.npy", eids)
    print(f"saved {len(eids)} embeddings with shape {embeddings.shape}")


if __name__ == "__main__":
    main()
