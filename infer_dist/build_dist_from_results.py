"""
从 infer_dist/results/*.csv 聚合推理结果，生成风险分布 JSON 供 web 使用。
- UKB → 欧洲人群分布 (Europe / UK Biobank)
- SDPP + SHCC + WHTM + PUDM → 中国队列分布 (Chinese cohort)

支持推理进行中的部分结果（会读取当前已有 CSV，不要求全部完成）。

用法:
  python infer_dist/build_dist_from_results.py
  python infer_dist/build_dist_from_results.py --results-dir infer_dist/results
"""

import csv
import json
import os
import argparse
from pathlib import Path
import numpy as np

BINS = 50  # 与前端 app.js 一致

CN_SETS = {"SDPP", "SHCC", "WHTM", "PUDM"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="gpu*.csv 所在目录",
    )
    parser.add_argument(
        "--out",
        default=str(Path(__file__).parent / "dist_data.json"),
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    ukb_probs, cn_probs = [], []

    for f in sorted(Path(args.results_dir).glob("gpu*.csv")):
        with open(f) as fp:
            for row in csv.DictReader(fp):
                prob_str = row.get("prob", "").strip()
                if not prob_str:
                    continue
                try:
                    prob = float(prob_str)
                except ValueError:
                    continue
                ds = row.get("dataset", "").strip()
                if ds == "UKB":
                    ukb_probs.append(prob)
                elif ds in CN_SETS:
                    cn_probs.append(prob)

    def to_hist(probs):
        if not probs:
            return [0] * BINS
        arr = np.array(probs)
        hist, _ = np.histogram(arr, bins=BINS, range=(0.0, 1.0))
        return hist.tolist()

    data = {
        "bins": BINS,
        "ukb": to_hist(ukb_probs),
        "cn": to_hist(cn_probs),
        "ukb_n": len(ukb_probs),
        "cn_n": len(cn_probs),
    }

    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)

    print(f"分布已生成: {args.out}")
    print(f"  UKB (欧洲人群):     n={len(ukb_probs):,}")
    print(f"  中国队列 (SDPP+SHCC+WHTM+PUDM): n={len(cn_probs):,}")

    for name, probs in [("UKB (European)", ukb_probs), ("中国队列 (Chinese)", cn_probs)]:
        if probs:
            arr = np.array(probs)
            print(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}, median={np.median(arr):.4f}")

    return args.out


if __name__ == "__main__":
    main()
