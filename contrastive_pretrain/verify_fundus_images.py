"""One-shot PIL verify() pass over fundus PNGs listed in fundus_table_extended.csv.

Removes rows whose fundus_image_path fails to load, and writes:
  - fundus_table_extended.csv  (cleaned in-place, keeps previous backup)
  - contrastive_pretrain/bad_fundus_images.txt  (log of bad paths + error)

Designed to be re-runnable; uses multiprocessing pool for throughput.
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import pandas as pd
from PIL import Image

REPO = Path("/data/home/shujia/CHD/model_train/RETFound_MAE-main")
DEFAULT_TABLE = REPO / "contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv"
DEFAULT_BAD_LOG = REPO / "contrastive_pretrain/bad_fundus_images.txt"


def verify_one(path: str) -> tuple[str, str | None]:
    try:
        if not os.path.isfile(path):
            return path, "missing"
        sz = os.path.getsize(path)
        if sz == 0:
            return path, "empty"
        with Image.open(path) as im:
            im.verify()
        # re-open to also test decode (verify() only checks headers for some formats)
        with Image.open(path) as im:
            im.load()
        return path, None
    except Exception as e:
        return path, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    ap.add_argument("--bad_log", type=Path, default=DEFAULT_BAD_LOG)
    ap.add_argument("--num_workers", type=int, default=16)
    args = ap.parse_args()

    df = pd.read_csv(args.table)
    paths = df["fundus_image_path"].tolist()
    print(f"[verify] total rows: {len(paths)}", flush=True)

    bad = []
    total = len(paths)
    with Pool(processes=args.num_workers) as pool:
        for i, (p, err) in enumerate(pool.imap_unordered(verify_one, paths, chunksize=64)):
            if err is not None:
                bad.append((p, err))
            if (i + 1) % 5000 == 0:
                print(f"[verify] {i+1}/{total}  bad so far: {len(bad)}", flush=True)

    print(f"[verify] done. bad images: {len(bad)}")

    if bad:
        with args.bad_log.open("w") as f:
            for p, err in bad:
                f.write(f"{p}\t{err}\n")
        print(f"[verify] wrote bad log -> {args.bad_log}")

    bad_set = {p for p, _ in bad}
    if bad_set:
        backup = args.table.with_suffix(".csv.preverify.bak")
        if not backup.exists():
            df.to_csv(backup, index=False)
            print(f"[verify] backup original -> {backup}")
        df_clean = df[~df["fundus_image_path"].isin(bad_set)].reset_index(drop=True)
        df_clean.to_csv(args.table, index=False)
        print(f"[verify] cleaned table: {len(df)} -> {len(df_clean)} rows ({args.table})")
    else:
        print("[verify] no bad images, table unchanged.")


if __name__ == "__main__":
    main()
