"""
Re-evaluate Task1 CMR checkpoints on val / test splits using the same forward + metric
definitions as training.

Usage:
  conda activate modeltrain
  cd .../RETFound_MAE-main
  python contrastive_pretrain/eval_cmr_task1_ckpts.py

Requires:
  - task1_cmr_{val,test}.csv with existing npy paths (v3 under --data_dir_v3 per-row {eid}.npy)
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

HERE = pathlib.Path(__file__).resolve().parent

from contrastive_pretrain.models_dualtower import TaskHead  # noqa: E402
from contrastive_pretrain.models_cmr import CMREncoder  # noqa: E402
from contrastive_pretrain.models_cmr_v3 import CMREncoderV3  # noqa: E402
from contrastive_pretrain.train_cmr_v3 import (  # noqa: E402
    CLS_COLS,
    REG_COLS,
    REG_NORM,
    CMRDatasetV3,
    evaluate as evaluate_v3,
    MEDSAM_CKPT,
)

VAL_CSV = str(HERE / "task_reports/task1_cmr_val.csv")
TEST_CSV = str(HERE / "task_reports/task1_cmr_test.csv")

# Default v3 preprocess dir (override if needed)
DEFAULT_DATA_DIR_V3 = "/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3"


class CMRDataset4(Dataset):
    """4-frame lax_sax npy from CSV `path` column (same cohort as standalone/v2)."""

    def __init__(self, csv_path: str):
        df = pd.read_csv(csv_path)
        df = df[df["path"].apply(lambda p: pathlib.Path(p).exists())].reset_index(drop=True)
        self.rows = df.to_dict("records")
        print(f"[dataset4] {csv_path}: {len(self.rows)} samples with existing npy")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        npy = np.load(r["path"]).astype(np.float32)
        assert npy.shape[0] == 4, f"expected 4 frames, got {npy.shape} for {r['path']}"
        cmr = torch.from_numpy(npy).unsqueeze(1)
        for i in range(cmr.shape[0]):
            lo, hi = cmr[i].min(), cmr[i].max()
            if hi > lo:
                cmr[i] = (cmr[i] - lo) / (hi - lo)

        cls_labels = [float(r.get(c, -1)) for c in CLS_COLS]
        reg_vals, reg_mask = [], []
        for c in REG_COLS:
            v = r.get(c, float("nan"))
            if v is None or (isinstance(v, float) and np.isnan(v)):
                reg_vals.append(0.0)
                reg_mask.append(0)
            else:
                reg_vals.append(float(v))
                reg_mask.append(1)

        return {
            "eid": int(r["eid"]),
            "cmr": cmr,
            "cls_labels": torch.tensor(cls_labels, dtype=torch.float32),
            "reg_labels": torch.tensor(reg_vals, dtype=torch.float32),
            "reg_mask": torch.tensor(reg_mask, dtype=torch.bool),
        }


@torch.no_grad()
def evaluate_4frame(encoder: CMREncoder, head: TaskHead, loader: DataLoader, device: torch.device):
    encoder.eval()
    head.eval()
    all_cls = [[] for _ in CLS_COLS]
    all_lbl = [[] for _ in CLS_COLS]
    all_reg = [[] for _ in REG_COLS]
    all_rgt = [[] for _ in REG_COLS]
    all_rmk = [[] for _ in REG_COLS]

    for batch in loader:
        cmr = batch["cmr"].to(device)
        cls_gt = batch["cls_labels"].to(device)
        reg_gt = batch["reg_labels"].to(device)
        reg_mk = batch["reg_mask"].to(device)
        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            _, z = encoder(cmr)
        cls_out, reg_out = head(z)
        for i in range(len(CLS_COLS)):
            all_cls[i].append(cls_out[i].cpu())
            all_lbl[i].append(cls_gt[:, i].cpu())
        for i in range(len(REG_COLS)):
            all_reg[i].append(reg_out[i].cpu())
            all_rgt[i].append(reg_gt[:, i].cpu())
            all_rmk[i].append(reg_mk[:, i].cpu())

    metrics = {}
    auc_vals = []
    for i, name in enumerate(CLS_COLS):
        logits = torch.cat(all_cls[i]).numpy()
        labels = torch.cat(all_lbl[i]).numpy()
        valid = labels >= 0
        try:
            auc = roc_auc_score(labels[valid], logits[valid]) if valid.sum() >= 2 else float("nan")
        except Exception:
            auc = float("nan")
        metrics[f"auc/{name}"] = auc
        if not np.isnan(auc):
            auc_vals.append(auc)
    metrics["mean_auc"] = float(np.mean(auc_vals)) if auc_vals else float("nan")

    for i, name in enumerate(REG_COLS):
        pred = torch.cat(all_reg[i]).numpy()
        gt = torch.cat(all_rgt[i]).numpy()
        mask = torch.cat(all_rmk[i]).numpy().astype(bool)
        if mask.sum() == 0:
            metrics[f"mae/{name}"] = float("nan")
            continue
        mu, std = REG_NORM[i]["mean"], REG_NORM[i]["std"]
        metrics[f"mae/{name}"] = float(np.abs(pred[mask] * std + mu - gt[mask]).mean())

    return metrics


def load_and_eval_v3(ckpt_path: pathlib.Path, csv_path: str, data_dir_v3: str, device):
    ds = CMRDatasetV3(csv_path, data_dir_v3, is_train=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2, pin_memory=False)
    encoder = CMREncoderV3(
        proj_dim=256,
        embed_dim=768,
        spatial_pool=4,
        transformer_heads=8,
        transformer_depth=2,
        medsam_ckpt=MEDSAM_CKPT,
        freeze_backbone=False,
    ).to(device)
    head = TaskHead(in_dim=768, n_cls=len(CLS_COLS), n_reg=len(REG_COLS)).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    metrics = evaluate_v3(encoder, head, loader, device)
    return metrics, ckpt.get("epoch"), ckpt.get("metrics")


def load_and_eval_4f(ckpt_path: pathlib.Path, csv_path: str, device):
    ds = CMRDataset4(csv_path)
    loader = DataLoader(ds, batch_size=24, shuffle=False, num_workers=2, pin_memory=False)
    encoder = CMREncoder(
        proj_dim=256,
        num_frames=4,
        embed_dim=768,
        img_size=224,
        medsam_ckpt=MEDSAM_CKPT,
        freeze_backbone=False,
    ).to(device)
    head = TaskHead(in_dim=768, n_cls=len(CLS_COLS), n_reg=len(REG_COLS)).to(device)
    ckpt = torch.load(str(ckpt_path), map_location=device)
    encoder.load_state_dict(ckpt["encoder"], strict=True)
    head.load_state_dict(ckpt["head"], strict=True)
    metrics = evaluate_4frame(encoder, head, loader, device)
    return metrics, ckpt.get("epoch"), ckpt.get("metrics")


def fmt(m: dict) -> str:
    lines = []
    for k in sorted(m.keys()):
        v = m[k]
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir_v3", type=str, default=DEFAULT_DATA_DIR_V3)
    args = ap.parse_args()

    v3dir = pathlib.Path(args.data_dir_v3)
    for split in ("val", "test"):
        df = pd.read_csv(HERE / f"task_reports/task1_cmr_{split}.csv")
        n_all = len(df)
        n_v3 = sum(v3dir.joinpath(f"{e}.npy").exists() for e in df["eid"])
        print(f"[coverage] task1_cmr_{split}: rows={n_all}, v3_npy_present={n_v3}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}\n")

    jobs = [
        ("CMR v3 (multi-view)", HERE / "checkpoints_cmr_v3/best.pth", "v3"),
        ("CMR v2 aug+freeze5 (4-frame)", HERE / "checkpoints_cmr_v2_aug_freeze5/best.pth", "4f"),
        ("CMR standalone (4-frame)", HERE / "checkpoints_cmr_standalone/best.pth", "4f"),
    ]

    for split_name, csvp in [("VAL", VAL_CSV), ("TEST", TEST_CSV)]:
        print(f"========== {split_name}: {csvp} ==========")
        for label, ckpt_p, kind in jobs:
            if not ckpt_p.is_file():
                print(f"[SKIP] missing {ckpt_p}")
                continue
            print(f"--- {label} | {ckpt_p.name} ---")
            if kind == "v3":
                m_new, ep_old, m_ckpt = load_and_eval_v3(ckpt_p, csvp, args.data_dir_v3, device)
            else:
                m_new, ep_old, m_ckpt = load_and_eval_4f(ckpt_p, csvp, device)
            print(f"  checkpoint_epoch={ep_old}")
            print("  recomputed:")
            print(fmt(m_new))
            if m_ckpt:
                print("  embedded_in_ckpt_at_save (reference):")
                print(fmt(m_ckpt))
            print()


if __name__ == "__main__":
    main()
