#!/usr/bin/env python3
"""
S_GT 分布诊断（Step 1）

Part A — 训练前做一次（全局 + per-fundus）：
  - 随机配对子样本的 S_GT 直方图
  - 全局分位数 P1,P5,...,P99（对大矩阵用蒙特卡洛子样本估计）
  - 每个 train fundus：对全库 CMR 的 similarity 向量算 P5/P95 与 range

Part B 统计在训练时由 engine_contrast 写入 CSV；本脚本提供绘图：

  python contrastive_pretrain/diagnostics_sgt_distribution.py plot-batch \\
    --stats_csv output_dir/sgt_batch_stats.csv \\
    --global_json output_dir/sgt_global_percentiles.json \\
    --out_png output_dir/sgt_batch_diagnostic.png

全局分析示例：

  python contrastive_pretrain/diagnostics_sgt_distribution.py global \\
    --fundus_csv contrastive_pretrain/preprocessed_data/fundus_table.csv \\
    --cmr_csv contrastive_pretrain/preprocessed_data/cmr_table.csv \\
    --pc_cols PC1,PC2,... \\
    --sigma 6.5893 \\
    --out_dir output_dir/sgt_diag \\
    --random_pairs 5000000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd


def _load_train_fundus_pc(fundus_csv: str, pc_cols: list) -> tuple[np.ndarray, list]:
    df = pd.read_csv(fundus_csv)
    df = df[df["split"] == "train"].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("train fundus 为空")
    exist_mask = df["fundus_image_path"].apply(os.path.exists)
    df = df[exist_mask].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("train fundus 无有效图像路径")
    F = df[pc_cols].values.astype(np.float64)
    eids = df["eid"].tolist()
    return F, eids


def _load_train_cmr_pc(cmr_csv: str, pc_cols: list) -> np.ndarray:
    df = pd.read_csv(cmr_csv)
    df = df[df["split"] == "train"].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError("train CMR 为空")
    return df[pc_cols].values.astype(np.float64)


def run_global(
    fundus_csv: str,
    cmr_csv: str,
    pc_cols: list,
    sigma: float,
    out_dir: str,
    random_pairs: int,
    per_fundus_chunk: int,
    seed: int,
):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)

    F, eids = _load_train_fundus_pc(fundus_csv, pc_cols)
    C = _load_train_cmr_pc(cmr_csv, pc_cols)
    Nf, Nc, P = F.shape[0], C.shape[0], F.shape[1]
    print(f"[global] N_fundus={Nf}, N_cmr={Nc}, P={P}, sigma={sigma}")

    denom = 2.0 * sigma**2

    # --- 随机配对子样本：用于直方图与近似分位数 ---
    n_s = min(int(random_pairs), 20_000_000)
    ii = rng.integers(0, Nf, size=n_s)
    jj = rng.integers(0, Nc, size=n_s)
    d = np.linalg.norm(F[ii] - C[jj], axis=1)
    s_rand = np.exp(-(d**2) / denom)

    hist_bins = 80
    hist, edges = np.histogram(s_rand, bins=hist_bins, range=(0, 1))
    pct = [1, 5, 25, 50, 75, 95, 99]
    pct_vals = np.percentile(s_rand, pct).tolist()
    global_stats = {f"P{p}": float(v) for p, v in zip(pct, pct_vals)}
    global_stats["random_pairs_used"] = int(n_s)
    global_stats["sigma"] = float(sigma)
    global_stats["N_fundus"] = int(Nf)
    global_stats["N_cmr"] = int(Nc)

    with open(os.path.join(out_dir, "sgt_global_percentiles.json"), "w") as f:
        json.dump(global_stats, f, indent=2)
    np.savez_compressed(
        os.path.join(out_dir, "sgt_random_sample_for_hist.npz"),
        s_sample=s_rand.astype(np.float32),
        hist_counts=hist,
        hist_edges=edges,
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(s_rand, bins=hist_bins, range=(0, 1), color="steelblue", alpha=0.85, edgecolor="white")
        for p, v in zip(pct, pct_vals):
            ax.axvline(v, color="red", alpha=0.35, linestyle="--", linewidth=0.8)
        ax.set_xlabel("S_GT (Gaussian kernel)")
        ax.set_ylabel("count")
        ax.set_title(f"S_GT random pairs (n={n_s:,})  train fundus × train CMR")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "sgt_global_histogram.png"), dpi=150)
        plt.close()
    except Exception as e:
        print(f"[global] matplotlib 跳过: {e}")

    # --- Per-fundus: 每行与全 CMR ---
    p5_list, p95_list, range_list = [], [], []
    # 分块矩阵乘避免一次性 (Nf x Nc) 过大：逐 fundus 与 C
    for i in range(Nf):
        if i % 200 == 0:
            print(f"  per-fundus {i}/{Nf}")
        # d_i: (Nc,)
        diff = C - F[i]
        d = np.linalg.norm(diff, axis=1)
        s_row = np.exp(-(d**2) / denom)
        p5, p95 = np.percentile(s_row, [5, 95])
        p5_list.append(float(p5))
        p95_list.append(float(p95))
        range_list.append(float(p95 - p5))

    per_df = pd.DataFrame(
        {
            "eid": eids,
            "per_row_P5": p5_list,
            "per_row_P95": p95_list,
            "P95_minus_P5": range_list,
        }
    )
    per_df.to_csv(os.path.join(out_dir, "sgt_per_fundus_percentiles.csv"), index=False)

    print("[global] 写出:", os.path.join(out_dir, "sgt_global_percentiles.json"))
    print("[global] 写出:", os.path.join(out_dir, "sgt_per_fundus_percentiles.csv"))
    print("[global] 随机子样本分位数:", global_stats)


def plot_batch(stats_csv: str, global_json: str, out_png: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(stats_csv)
    with open(global_json) as f:
        g = json.load(f)
    p5 = g.get("P5", g.get("P1"))
    p95 = g.get("P95", g.get("P99"))

    x = np.arange(len(df))
    mean = df["sgt_mean_masked"].values
    std = df["sgt_std_masked"].values

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.errorbar(x, mean, yerr=std, fmt="o", markersize=2, alpha=0.6, capsize=0, elinewidth=0.6)
    if p5 is not None:
        ax.axhline(float(p5), color="green", linestyle="--", label=f"global P5={float(p5):.4f}")
    if p95 is not None:
        ax.axhline(float(p95), color="orange", linestyle="--", label=f"global P95={float(p95):.4f}")
    ax.set_xlabel("batch index (order in log)")
    ax.set_ylabel("S_GT (masked: exclude diagonal same-EID)")
    ax.set_title("Batch-level S_GT mean ± std (negatives + cross-batch pairs, no (i,i))")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close()
    print(f"[plot-batch] Wrote {out_png}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("global", help="Part A: 全局 + per-fundus（训练前运行）")
    g.add_argument("--fundus_csv", required=True)
    g.add_argument("--cmr_csv", required=True)
    g.add_argument("--pc_cols", required=True, help="逗号分隔")
    g.add_argument("--sigma", type=float, required=True)
    g.add_argument("--out_dir", default="output_dir/sgt_diag")
    g.add_argument("--random_pairs", type=int, default=5_000_000)
    g.add_argument("--seed", type=int, default=0)

    p = sub.add_parser("plot-batch", help="根据训练日志 CSV 画 batch 诊断图")
    p.add_argument("--stats_csv", required=True)
    p.add_argument("--global_json", required=True)
    p.add_argument("--out_png", required=True)

    args = ap.parse_args()

    if args.cmd == "global":
        pc_cols = [c.strip() for c in args.pc_cols.split(",")]
        run_global(
            args.fundus_csv,
            args.cmr_csv,
            pc_cols,
            args.sigma,
            args.out_dir,
            args.random_pairs,
            0,
            args.seed,
        )
    else:
        plot_batch(args.stats_csv, args.global_json, args.out_png)


if __name__ == "__main__":
    main()
