#!/usr/bin/env python3
"""
训练前运行一次：对每个 train fundus 的 EID，计算与 train CMR 全表的 S_GT，
按固定阈值分桶（高 / 中 / 低），供 CMRBank 分层负样本采样。

阈值（与实验设计一致）：
  high: S_GT > thresh_high (默认 0.8)
  low:  S_GT < thresh_low  (默认 0.3)
  mid:  其余

输出 pickle：meta + buckets[eid_key] = {'high','low','mid'} 每个为 int64 数组，
索引为 **与 cmr_csv train split 行顺序一致的全表行号**（0..N_cmr-1）。

用法：
  python contrastive_pretrain/precompute_stratified_buckets.py \\
    --fundus_csv ... --cmr_csv ... --pc_cols ... --sigma 6.5893 \\
    --out_pkl output_dir/stratified_buckets.pkl
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from contrastive_pretrain.datasets_contrast import CMRBank as _CMRBankRef  # noqa: E402


def _norm_eid(eid):
    return _CMRBankRef._normalize_eid_key(eid)


def _filter_fundus_train(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["split"] == "train"].reset_index(drop=True)
    if "fundus_image_path" in df.columns:
        m = df["fundus_image_path"].apply(os.path.exists)
        df = df[m].reset_index(drop=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fundus_csv", required=True)
    ap.add_argument("--cmr_csv", required=True)
    ap.add_argument("--pc_cols", required=True, help="逗号分隔")
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--thresh_high", type=float, default=0.8)
    ap.add_argument("--thresh_low", type=float, default=0.3)
    ap.add_argument("--out_pkl", required=True)
    args = ap.parse_args()

    pc_cols = [c.strip() for c in args.pc_cols.split(",")]

    fu = pd.read_csv(args.fundus_csv)
    cm = pd.read_csv(args.cmr_csv)
    fu_t = _filter_fundus_train(fu)
    cm_t = cm[cm["split"] == "train"].reset_index(drop=True)

    for c in pc_cols:
        if c not in fu_t.columns or c not in cm_t.columns:
            raise ValueError(f"PC 列缺失: {c}")

    C = cm_t[pc_cols].values.astype(np.float64)
    Nc = C.shape[0]
    denom = 2.0 * args.sigma**2

    # 每个唯一 EID 只算一次（同人多行 fundus 共用 PC）；键与 CMRBank._normalize_eid_key 一致
    eid_to_pc: dict = {}
    for _, row in fu_t.iterrows():
        k = _norm_eid(row["eid"])
        if k not in eid_to_pc:
            eid_to_pc[k] = row[pc_cols].values.astype(np.float64)

    buckets: dict = {}
    for eid, pc_f in eid_to_pc.items():
        d = np.linalg.norm(C - pc_f, axis=1)
        s = np.exp(-(d**2) / denom)
        high = np.where(s > args.thresh_high)[0].astype(np.int64)
        low = np.where(s < args.thresh_low)[0].astype(np.int64)
        mid = np.where((s >= args.thresh_low) & (s <= args.thresh_high))[0].astype(np.int64)
        buckets[eid] = {"high": high, "low": low, "mid": mid}

    meta = {
        "sigma": args.sigma,
        "thresh_high": args.thresh_high,
        "thresh_low": args.thresh_low,
        "n_cmr_train": int(Nc),
        "n_fundus_train_rows": int(len(fu_t)),
        "n_unique_eid": len(eid_to_pc),
        "pc_cols": pc_cols,
        "fundus_csv": os.path.abspath(args.fundus_csv),
        "cmr_csv": os.path.abspath(args.cmr_csv),
    }

    payload = {"meta": meta, "buckets": buckets}

    os.makedirs(os.path.dirname(os.path.abspath(args.out_pkl)) or ".", exist_ok=True)
    with open(args.out_pkl, "wb") as f:
        pickle.dump(payload, f, protocol=4)

    print(f"[precompute_stratified_buckets] wrote {args.out_pkl}")
    print(f"  meta: {meta}")
    # 抽样打印桶大小
    sample_eid = next(iter(buckets.keys()))
    b0 = buckets[sample_eid]
    print(
        f"  例 EID {sample_eid}: high={len(b0['high'])}, "
        f"mid={len(b0['mid'])}, low={len(b0['low'])}"
    )


if __name__ == "__main__":
    main()
