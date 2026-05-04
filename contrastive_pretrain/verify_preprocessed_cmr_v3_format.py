#!/usr/bin/env python3
"""
verify_preprocessed_cmr_v3_format.py

Check that preprocessed_cmr_v3/*.npy files match the pipeline contract:
  shape (16, 224, 224), dtype float16, finite values; optional JSON sidecar.

Also compares distribution summaries against a set of reference EIDs (existing
good files) so new batches stay consistent.

Usage:
  conda activate modeltrain  # optional
  python contrastive_pretrain/verify_preprocessed_cmr_v3_format.py \\
      --ref_eids 1000191,3257462 \\
      --check_csv contrastive_pretrain/task_reports/task1_cmr_test.csv \\
      --data_dir /data/home/shujia/UKB/CMRI/preprocessed_cmr_v3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize_array(a: np.ndarray) -> dict:
    af = a.astype(np.float32)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "finite_frac": float(np.isfinite(af).mean()),
        "global_min": float(np.nanmin(af)),
        "global_max": float(np.nanmax(af)),
        "per_frame_mean_std": [
            float(af[i].mean()) for i in range(a.shape[0])
        ],
    }


def load_ref_summaries(data_dir: Path, eids: list[int]) -> dict[int, dict]:
    out = {}
    for e in eids:
        p = data_dir / f"{e}.npy"
        if not p.exists():
            raise FileNotFoundError(p)
        a = np.load(p)
        out[e] = summarize_array(a)
    return out


def check_one_structural(path: Path) -> tuple[bool, list[str]]:
    errs = []
    a = np.load(path)
    if a.dtype != np.float16:
        errs.append(f"dtype {a.dtype} != float16")
    if a.shape != (16, 224, 224):
        errs.append(f"shape {a.shape} != (16,224,224)")
    if not np.isfinite(a.astype(np.float32)).all():
        errs.append("non-finite values")
    # not completely blank cine stack (T1 frame 15 may be zeros if absent)
    if float(a[:15].astype(np.float32).max()) < 1e-6:
        errs.append("cine frames nearly all zero")

    ok = len(errs) == 0
    return ok, errs


def global_brightness(a: np.ndarray) -> float:
    return float(a.astype(np.float32).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3")
    ap.add_argument(
        "--check_csv",
        type=str,
        default="",
        help="CSV with eid column to verify (e.g. task1_cmr_test.csv)",
    )
    ap.add_argument(
        "--ref_eids",
        type=str,
        default="",
        help="Comma-separated reference EIDs with known-good npy",
    )
    ap.add_argument("--max_report_errors", type=int, default=30)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)

    ref_eids = [int(x) for x in args.ref_eids.split(",") if x.strip()]
    if not ref_eids:
        # default: pick first 3 test rows that already had v3 before bulk fill
        repo = Path(__file__).resolve().parents[1]
        test_csv = repo / "contrastive_pretrain/task_reports/task1_cmr_test.csv"
        df = pd.read_csv(test_csv)
        ref_eids = []
        for e in df["eid"].astype(int):
            if (data_dir / f"{int(e)}.npy").exists():
                ref_eids.append(int(e))
                if len(ref_eids) >= 3:
                    break
        print(f"[auto ref_eids] {ref_eids}")

    ref_summ = load_ref_summaries(data_dir, ref_eids)
    print("=== Reference summaries (legacy / known-good EIDs) ===")
    for e, s in ref_summ.items():
        print(f"  eid={e} shape={s['shape']} dtype={s['dtype']} min={s['global_min']:.4f} max={s['global_max']:.4f}")

    if not args.check_csv:
        print("No --check_csv; done after ref dump.")
        return

    csv_path = Path(args.check_csv)
    if not csv_path.is_file():
        csv_path = Path(__file__).resolve().parents[1] / args.check_csv
    df = pd.read_csv(csv_path)
    eids = df["eid"].astype(int).tolist()

    bad = []
    missing = []
    json_missing = []
    brightness_new = []
    brightness_legacy = []
    new_batch_path = csv_path.parent / "task1_cmr_test_eids_missing_v3.txt"
    new_batch: set[int] = set()
    if new_batch_path.is_file():
        new_batch = {int(l.strip()) for l in new_batch_path.read_text().splitlines() if l.strip()}
        print(f"[cohort split] new_batch from {new_batch_path.name} n={len(new_batch)}")
    legacy_ref = set(ref_eids)
    for e in eids:
        npy = data_dir / f"{e}.npy"
        js = data_dir / f"{e}.json"
        if not npy.exists():
            missing.append(e)
            continue
        ok, errs = check_one_structural(npy)
        arr = np.load(npy)
        gb = global_brightness(arr)
        if new_batch:
            if e in new_batch:
                brightness_new.append(gb)
            else:
                brightness_legacy.append(gb)
        elif e in legacy_ref:
            brightness_legacy.append(gb)
        else:
            brightness_new.append(gb)
        if not js.exists():
            json_missing.append(e)
        if not ok:
            bad.append((e, errs))

    print(f"\n=== Checked {len(eids)} EIDs from {csv_path.name} ===")
    print(f"  missing npy: {len(missing)}")
    print(f"  missing json sidecar: {len(json_missing)}")
    print(f"  failed format checks: {len(bad)}")
    if missing[:15]:
        print(f"  sample missing: {missing[:15]}...")
    if json_missing[:15]:
        print(f"  sample no json: {json_missing[:15]}...")
    if brightness_legacy and brightness_new:
        print(
            "\n=== Brightness sanity (mean pixel value [0,1], cohort-level; not identical across patients) ==="
        )
        print(
            f"  legacy subset n={len(brightness_legacy)}  "
            f"mean={np.mean(brightness_legacy):.4f} std={np.std(brightness_legacy):.4f}"
        )
        print(
            f"  new-batch rows n={len(brightness_new)}  "
            f"mean={np.mean(brightness_new):.4f} std={np.std(brightness_new):.4f}"
        )
    for e, errs in bad[: args.max_report_errors]:
        print(f"  BAD eid={e}: {'; '.join(errs)}")
    if bad:
        raise SystemExit(1)
    if missing:
        raise SystemExit(2)
    print("\nOK — all rows have npy and match structural checks.")


if __name__ == "__main__":
    main()
