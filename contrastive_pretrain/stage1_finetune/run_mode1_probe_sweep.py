#!/usr/bin/env python3
"""
在固定 embedding（如 emb_retfound.pt）上，对线性探针做多套超参「套餐」顺序评估，
汇总 mean test AUROC，并与 baseline 对比，便于快速找更有希望的训练设定。

示例：
  python contrastive_pretrain/stage1_finetune/run_mode1_probe_sweep.py \\
    --embedding_pt output_dir/stage1_emb_cache/emb_retfound.pt \\
    --sweep_root output_dir/mode1_probe_sweep --gpus 0,1,2,3

说明：
- 医疗二分类常见改进：pos_weight（类不平衡）、略调 lr/wd、去掉平滑改硬标签等。
- 非全参数搜索；若需更广，可复制 PACKAGES 列表自行增删。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from contrastive_pretrain.stage1_finetune.stage1_paths import (  # noqa: E402
    COMPOSITE_TASKS,
    ICD_PREVALENT_TASKS,
)


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", s)


# 套餐：名称 + 传给 train_mode1_from_embeddings.py 的额外参数（已含 --hparam_tag）
PACKAGES: list[tuple[str, list[str]]] = [
    (
        "baseline",
        [
            "--hparam_tag",
            "baseline",
            "--loss",
            "bce_smooth",
            "--lr",
            "0.0001",
            "--weight_decay",
            "0.05",
            "--smoothing",
            "0.1",
        ],
    ),
    (
        "pos_weight_auto",
        [
            "--hparam_tag",
            "pos_weight_auto",
            "--loss",
            "bce_pos_weight",
            "--pos_weight",
            "auto",
            "--smoothing",
            "0.0",
            "--lr",
            "0.0001",
            "--weight_decay",
            "0.05",
        ],
    ),
    (
        "low_weight_decay",
        [
            "--hparam_tag",
            "low_weight_decay",
            "--loss",
            "bce_smooth",
            "--lr",
            "0.0001",
            "--weight_decay",
            "0.01",
            "--smoothing",
            "0.1",
        ],
    ),
    (
        "lr3e4_wd1e2",
        [
            "--hparam_tag",
            "lr3e4_wd1e2",
            "--loss",
            "bce_smooth",
            "--lr",
            "0.0003",
            "--weight_decay",
            "0.01",
            "--smoothing",
            "0.05",
        ],
    ),
    (
        "bce_hard",
        [
            "--hparam_tag",
            "bce_hard",
            "--loss",
            "bce_hard",
            "--lr",
            "0.0001",
            "--weight_decay",
            "0.05",
            "--smoothing",
            "0.0",
        ],
    ),
]


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--embedding_pt",
        type=str,
        required=True,
        help="相对 REPO 或绝对路径",
    )
    ap.add_argument("--sweep_root", type=str, default="output_dir/mode1_probe_sweep")
    ap.add_argument("--init_subdir", type=str, default="retfound", help="输出子目录名")
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16384)
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument(
        "--packages",
        type=str,
        default="",
        help='逗号分隔套餐名，默认跑全部。例："baseline,pos_weight_auto"',
    )
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def load_test_aurocs(out_root: Path, tasks: list[str]) -> dict[str, float]:
    m: dict[str, float] = {}
    for t in tasks:
        p = out_root / safe_name(t) / "metrics_mode1_emb.json"
        if not p.is_file():
            m[t] = float("nan")
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        m[t] = float(d.get("test_auroc", float("nan")))
    return m


def summarize(
    baseline: dict[str, float], cur: dict[str, float]
) -> tuple[float, float, int, int]:
    """mean delta, median delta, n_improved, n_paired"""
    deltas = []
    for k in baseline:
        if k not in cur:
            continue
        b, c = baseline[k], cur[k]
        if (
            b != b
            or c != c  # nan
        ):
            continue
        deltas.append(c - b)
    if not deltas:
        return float("nan"), float("nan"), 0, 0
    return (
        float(statistics.mean(deltas)),
        float(statistics.median(deltas)),
        sum(1 for d in deltas if d > 0),
        len(deltas),
    )


def run_one_package(
    pkg_name: str,
    extra: list[str],
    emb_pt: str,
    out_root: Path,
    gpus: list[int],
    epochs: int,
    batch_size: int,
    delivery: str,
    skip_done: bool,
    dry_run: bool,
) -> None:
    tasks = list(COMPOSITE_TASKS) + list(ICD_PREVALENT_TASKS)
    py = os.environ.get("PYTHON", sys.executable)
    train_py = REPO / "contrastive_pretrain/stage1_finetune/train_mode1_from_embeddings.py"

    jobs = []
    for t in tasks:
        od = out_root / safe_name(t)
        jobs.append({"target": t, "output_dir": str(od), "embedding_pt": emb_pt})

    pending = []
    for j in jobs:
        if skip_done and (Path(j["output_dir"]) / "DONE").is_file():
            print(f"[skip DONE] {j['output_dir']}")
            continue
        pending.append(j)

    print(f"\n>>> 套餐 [{pkg_name}]  待训 {len(pending)}/{len(jobs)}  GPU={gpus}\n")

    if not pending:
        print(f"[{pkg_name}] 无需训练（已全部 DONE 或为空）")
        return

    if dry_run:
        print(extra[:20], "...")
        return

    slots: list[tuple[subprocess.Popen | None, dict | None, int]] = [
        (None, None, gpus[i]) for i in range(len(gpus))
    ]

    def start_job(j: dict, gpu: int) -> subprocess.Popen:
        Path(j["output_dir"]).mkdir(parents=True, exist_ok=True)
        cmd = [
            py,
            "-u",
            str(train_py),
            "--embedding_pt",
            j["embedding_pt"],
            "--target_col",
            j["target"],
            "--gpu",
            "0",
            "--output_dir",
            j["output_dir"],
            "--delivery_final_csv",
            delivery,
            "--epochs",
            str(epochs),
            "--batch_size",
            str(batch_size),
        ]
        cmd.extend(extra)
        logf = open(Path(j["output_dir"]) / "train_stdout.log", "a", encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        print(f"[start gpu={gpu}] {pkg_name} {j['target']}")
        return subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(REPO),
            env=env,
        )

    qi = 0
    while qi < len(pending) or any(s[0] is not None for s in slots):
        for i, (proc, meta, gpu) in enumerate(slots):
            if proc is None:
                if qi < len(pending):
                    j = pending[qi]
                    qi += 1
                    p = start_job(j, gpu)
                    slots[i] = (p, j, gpu)
                continue
            assert proc is not None and meta is not None
            ret = proc.poll()
            if ret is None:
                continue
            print(f"[done gpu={gpu} ret={ret}] {pkg_name} {meta['target']}")
            slots[i] = (None, None, gpu)
            if qi < len(pending):
                j = pending[qi]
                qi += 1
                p = start_job(j, gpu)
                slots[i] = (p, j, gpu)
        time.sleep(2)


def main():
    args = parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    emb = args.embedding_pt
    if not os.path.isabs(emb):
        emb = str(REPO / emb)
    if not os.path.isfile(emb):
        print(f"[错误] 找不到 embedding: {emb}")
        sys.exit(1)

    sweep_root = REPO / args.sweep_root
    sweep_root.mkdir(parents=True, exist_ok=True)

    if args.packages.strip():
        want = {x.strip() for x in args.packages.split(",") if x.strip()}
        packages = [(n, e) for n, e in PACKAGES if n in want]
        missing = want - {n for n, _ in packages}
        if missing:
            print(f"[错误] 未知套餐名: {missing}")
            sys.exit(1)
    else:
        packages = PACKAGES

    tasks = list(COMPOSITE_TASKS) + list(ICD_PREVALENT_TASKS)
    summary_rows: list[dict] = []
    baseline_aucs: dict[str, float] | None = None

    for pkg_name, extra in packages:
        out_pkg = sweep_root / pkg_name / args.init_subdir
        delivery = str(sweep_root / pkg_name / "delivery_probe.csv")
        run_one_package(
            pkg_name,
            extra,
            emb,
            out_pkg,
            gpus,
            args.epochs,
            args.batch_size,
            delivery,
            args.skip_done,
            args.dry_run,
        )
        if args.dry_run:
            continue
        aucs = load_test_aurocs(out_pkg, tasks)
        vals = [aucs[t] for t in tasks if aucs[t] == aucs[t]]
        mean_te = sum(vals) / len(vals) if vals else float("nan")

        row = {
            "package": pkg_name,
            "mean_test_auroc": mean_te,
            "n_valid_tasks": len(vals),
        }
        if baseline_aucs is None:
            baseline_aucs = aucs
            row["mean_delta_vs_baseline"] = 0.0
            row["median_delta_vs_baseline"] = 0.0
            row["n_tasks_improved"] = 0
            row["n_paired"] = len(vals)
        else:
            md, med, n_imp, n_pr = summarize(baseline_aucs, aucs)
            row["mean_delta_vs_baseline"] = md
            row["median_delta_vs_baseline"] = med
            row["n_tasks_improved"] = n_imp
            row["n_paired"] = n_pr
        summary_rows.append(row)
        print(
            f"\n[汇总] {pkg_name}: mean_test_auroc={mean_te:.4f}  "
            f"vs_baseline mean_delta={row.get('mean_delta_vs_baseline', 0):.4f}  "
            f"improved {row.get('n_tasks_improved', 0)}/{row.get('n_paired', 0)} tasks\n"
        )

    if args.dry_run:
        return

    if not summary_rows:
        print("[sweep] 无汇总行（未跑任何套餐）")
        return

    sum_path = sweep_root / "sweep_summary.csv"
    import csv

    with open(sum_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    print("=" * 72)
    print(f"[sweep] 汇总已写入: {sum_path}")
    best = max(summary_rows, key=lambda r: r["mean_test_auroc"])
    print(
        f"[sweep] 按 mean_test_auroc 最高套餐: {best['package']} = {best['mean_test_auroc']:.4f}"
    )
    if len(summary_rows) > 1:
        rest = [r for r in summary_rows if r["package"] != summary_rows[0]["package"]]
        best_delta = max(rest, key=lambda r: r.get("mean_delta_vs_baseline") or 0.0)
        print(
            f"[sweep] 相对 baseline 平均提升最大: {best_delta['package']} "
            f"(mean_delta={best_delta.get('mean_delta_vs_baseline', 0):.4f})"
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
