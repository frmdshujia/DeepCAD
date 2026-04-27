#!/usr/bin/env python3
"""
同人 fundus–CMR 的 S_GT 分布：同一次 imaging visit vs 同人不同 visit。

定义（与 cmr_table 的 eid + instance 列一致；UKB 中 instance 表示 imaging visit）：
  - 同一次检查：同一 (eid, instance)，眼底行与 CMR 行做 inner join。
  - 不同次检查、同人：同一 eid，但 CMR 的 instance ≠ 眼底行的 instance。

S_GT：exp(-||p_fu - p_cmr||^2 / (2σ^2))，与 loss_contrast.compute_sgt 一致。

用法：
  python contrastive_pretrain/diagnostics_sgt_same_visit_vs_cross.py \\
    --fundus_csv ... --cmr_csv ... --pc_cols M1_PC1,... --sigma 6.5893 --out_dir ...
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _sgt_from_pc(
    pc_a: np.ndarray, pc_b: np.ndarray, sigma: float
) -> np.ndarray:
    d = np.linalg.norm(pc_a - pc_b, axis=1)
    return np.exp(-(d**2) / (2.0 * sigma**2))


def _filter_fundus_existing(df: pd.DataFrame) -> pd.DataFrame:
    if "fundus_image_path" not in df.columns:
        return df
    m = df["fundus_image_path"].apply(os.path.exists)
    return df[m].reset_index(drop=True)


def run(
    fundus_csv: str,
    cmr_csv: str,
    pc_cols: list[str],
    sigma: float,
    fundus_split: str,
    out_dir: str,
    cmr_pool_for_cross: str,
):
    os.makedirs(out_dir, exist_ok=True)
    fu = pd.read_csv(fundus_csv)
    cm = pd.read_csv(cmr_csv)

    for c in pc_cols:
        if c not in fu.columns or c not in cm.columns:
            raise ValueError(f"PC 列缺失: {c}")

    fu_s = fu[fu["split"] == fundus_split].copy()
    fu_s = _filter_fundus_existing(fu_s)
    if len(fu_s) == 0:
        raise ValueError(f"fundus split={fundus_split} 无有效行")

    if cmr_pool_for_cross == "all":
        cm_pool = cm.copy()
    elif cmr_pool_for_cross == "train":
        cm_pool = cm[cm["split"] == "train"].copy()
    else:
        raise ValueError("cmr_pool_for_cross 须为 all 或 train")

    # --- 同 visit：inner join (eid, instance)；CMR 用 cm_pool ---
    merged = fu_s.merge(
        cm_pool,
        on=["eid", "instance"],
        suffixes=("_fu", "_cm"),
        how="inner",
    )
    cols_fu = [f"{c}_fu" for c in pc_cols]
    cols_cm = [f"{c}_cm" for c in pc_cols]
    missing = [c for c in cols_fu + cols_cm if c not in merged.columns]
    if missing:
        raise ValueError(f"merge 后缺少列 {missing[:5]}...（检查 fundus/cmr 是否含相同 PC 列名）")
    pf = merged[cols_fu].values.astype(np.float64)
    pc = merged[cols_cm].values.astype(np.float64)
    s_same_visit = _sgt_from_pc(pf, pc, sigma)
    max_pc_diff = float(np.abs(pf - pc).max())
    same_pc_rows = max_pc_diff == 0.0

    # --- 同人不同 instance ---
    s_cross: list[float] = []
    for _, fr in fu_s.iterrows():
        eid, inst = fr["eid"], fr["instance"]
        sub = cm_pool[(cm_pool["eid"] == eid) & (cm_pool["instance"] != inst)]
        if len(sub) == 0:
            continue
        pfu = fr[pc_cols].values.astype(np.float64).reshape(1, -1)
        pcm = sub[pc_cols].values.astype(np.float64)
        s_cross.extend(_sgt_from_pc(np.repeat(pfu, len(sub), axis=0), pcm, sigma).tolist())

    s_cross_arr = np.array(s_cross, dtype=np.float64)

    summary = {
        "sigma": sigma,
        "fundus_split": fundus_split,
        "fundus_rows_used": int(len(fu_s)),
        "cmr_pool_for_cross": cmr_pool_for_cross,
        "n_same_visit_pairs": int(len(s_same_visit)),
        "n_cross_visit_pairs": int(len(s_cross_arr)),
        "same_visit_max_abs_pc_diff": max_pc_diff,
        "same_visit_pc_identical_in_tables": same_pc_rows,
        "same_visit_note": (
            "若 fundus/cmr 在相同 (eid,instance) 下 PC 完全一致，则 S_GT 恒为 1（与 sigma 无关）。"
            if same_pc_rows
            else "同 visit 的 fundus/CMR PC 存在差异，S_GT 可为 (0,1]。"
        ),
        "same_visit_percentiles": {
            k: float(np.percentile(s_same_visit, p))
            for k, p in [("P5", 5), ("P50", 50), ("P95", 95)]
        },
    }
    if len(s_cross_arr) > 0:
        summary["cross_visit_percentiles"] = {
            k: float(np.percentile(s_cross_arr, p))
            for k, p in [("P5", 5), ("P50", 50), ("P95", 95)]
        }

    with open(os.path.join(out_dir, "sgt_same_vs_cross_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    np.savez_compressed(
        os.path.join(out_dir, "sgt_same_cross_values.npz"),
        s_same_visit=s_same_visit.astype(np.float32),
        s_cross_visit=s_cross_arr.astype(np.float32) if len(s_cross_arr) else np.array([], np.float32),
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        bins = np.linspace(0, 1, 81)
        axes[0].hist(s_same_visit, bins=bins, color="steelblue", alpha=0.85, edgecolor="white")
        t0 = f"Same (eid, instance) — paired rows\nN={len(s_same_visit)}"
        if same_pc_rows:
            t0 += "\n(degenerate: PC identical in CSVs => S_GT=1)"
        axes[0].set_title(t0, fontsize=10)
        axes[0].set_xlabel("S_GT")
        axes[0].set_ylabel("count")

        if len(s_cross_arr) > 0:
            axes[1].hist(
                s_cross_arr,
                bins=bins,
                color="coral",
                alpha=0.85,
                edgecolor="white",
            )
        else:
            axes[1].text(
                0.5,
                0.5,
                "No pairs: no second instance\nfor this eid in CMR table",
                ha="center",
                va="center",
                transform=axes[1].transAxes,
                fontsize=10,
                color="#555",
            )
        axes[1].set_title(
            f"Same eid, different instance (CMR)\nN={len(s_cross_arr)}",
            fontsize=10,
        )
        axes[1].set_xlabel("S_GT")
        axes[1].set_ylabel("count")
        if 0 < len(s_cross_arr) <= 5:
            axes[1].text(
                0.5,
                0.95,
                "Very few multi-instance eids\nin current cmr_table",
                transform=axes[1].transAxes,
                ha="center",
                va="top",
                fontsize=9,
                color="#555",
            )

        fig.suptitle(f"S_GT (Gaussian kernel, sigma={sigma})", y=1.02)
        fig.tight_layout()
        p = os.path.join(out_dir, "sgt_same_visit_vs_cross_histogram.png")
        fig.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"[ok] 写出 {p}")
    except Exception as e:
        print(f"[warn] matplotlib: {e}")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"[ok] {out_dir}/sgt_same_vs_cross_summary.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fundus_csv", required=True)
    ap.add_argument("--cmr_csv", required=True)
    ap.add_argument("--pc_cols", required=True, help="逗号分隔，须与训练一致")
    ap.add_argument("--sigma", type=float, required=True)
    ap.add_argument("--fundus_split", default="train")
    ap.add_argument(
        "--cmr_pool_for_cross",
        choices=("all", "train"),
        default="all",
        help="同人不同 visit 时从 CMR 表的哪个子集取「另一 visit」（默认 all 以尽量多配对）",
    )
    ap.add_argument("--out_dir", default="output_dir/sgt_same_cross_diag")
    args = ap.parse_args()
    pc_cols = [c.strip() for c in args.pc_cols.split(",")]
    run(
        args.fundus_csv,
        args.cmr_csv,
        pc_cols,
        args.sigma,
        args.fundus_split,
        args.out_dir,
        args.cmr_pool_for_cross,
    )


if __name__ == "__main__":
    main()
