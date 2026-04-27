#!/usr/bin/env python3
"""
多卡任务队列：在 4 张 GPU 上并行跑 stage1 全矩阵（3 init × 3 mode × N 任务）。
任务顺序：先复合标签（COMPOSITE_TASKS），再 ICD（ICD_PREVALENT_TASKS）；
对每个 (target, init, mode) 组合只跑一次，output 下存在 DONE 则跳过。

断点：若 checkpoint_last.pth 存在且尚无 DONE，子进程带 --resume。
"""
from __future__ import annotations

import argparse
import os
import re
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


def build_jobs(
    inits: list,
    modes: list,
    tasks: list,
    base_out: Path,
) -> list[dict]:
    jobs = []
    for t in tasks:
        for init in inits:
            for mode in modes:
                od = base_out / init / f"mode{mode}" / safe_name(t)
                jobs.append(
                    {
                        "target_col": t,
                        "init_source": init,
                        "mode": mode,
                        "output_dir": str(od),
                    }
                )
    return jobs


def should_skip(od: Path) -> bool:
    return (od / "DONE").is_file()


def needs_resume(od: Path) -> bool:
    return (od / "checkpoint_last.pth").is_file() and not (od / "DONE").is_file()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_out", type=str, default="output_dir/stage1_finetune_matrix")
    ap.add_argument(
        "--stage1_csv",
        type=str,
        default="contrastive_pretrain/preprocessed_data/stage1_fundus_downstream_finetuning_with_image_paths.csv",
    )
    ap.add_argument(
        "--fundus_root",
        type=str,
        default="/data/home/home6/fundus_data/UKB/fundus_images",
    )
    ap.add_argument(
        "--inits",
        type=str,
        default="retfound,controlled,no_residual",
        help="逗号分隔",
    )
    ap.add_argument("--modes", type=str, default="1,2,3")
    ap.add_argument(
        "--tasks",
        type=str,
        default="",
        help="逗号分隔；默认同 train_one：先全部复合再全部 ICD",
    )
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument(
        "--delivery_live",
        type=str,
        default="output_dir/stage1_finetune_delivery_live.csv",
    )
    ap.add_argument(
        "--delivery_final",
        type=str,
        default="output_dir/stage1_finetune_delivery_final.csv",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument("--max_jobs", type=int, default=0, help=">0 时仅跑前 K 个任务（烟测）")
    ap.add_argument(
        "--loss_type",
        type=str,
        default="paper_bce",
        choices=["paper_bce", "bce_pos_weight"],
        help="透传给 stage1_train_one（默认与论文 BCE+平滑 一致）",
    )
    ap.add_argument(
        "--pos_weight",
        type=float,
        default=0.0,
        help="仅 loss_type=bce_pos_weight：0=训练集自动 neg/pos",
    )
    args = ap.parse_args()

    inits = [x.strip() for x in args.inits.split(",") if x.strip()]
    modes = [int(x) for x in args.modes.split(",") if x.strip()]
    if args.tasks.strip():
        tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    else:
        tasks = list(COMPOSITE_TASKS) + list(ICD_PREVALENT_TASKS)

    base_out = REPO / args.base_out
    base_out.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(inits, modes, tasks, base_out)
    if args.max_jobs > 0:
        jobs = jobs[: args.max_jobs]

    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    n_slots = len(gpus)

    pending = []
    for j in jobs:
        od = Path(j["output_dir"])
        if should_skip(od):
            print(f"[skip DONE] {od}")
            continue
        pending.append(j)

    print(f"[queue] 待跑 {len(pending)} / 总定义 {len(jobs)} 槽位={n_slots} GPU={gpus}")

    if args.dry_run:
        for j in pending[:20]:
            print(j)
        if len(pending) > 20:
            print("...")
        return

    script = REPO / "contrastive_pretrain/stage1_finetune/stage1_train_one.py"
    py = os.environ.get("PYTHON", sys.executable)

    slots: list[tuple[subprocess.Popen | None, dict | None, int]] = [
        (None, None, gpus[i]) for i in range(n_slots)
    ]

    def start_job(j: dict, gpu: int) -> subprocess.Popen:
        od = Path(j["output_dir"])
        od.mkdir(parents=True, exist_ok=True)
        cmd = [
            py,
            str(script),
            "--stage1_csv",
            str(REPO / args.stage1_csv),
            "--fundus_root",
            args.fundus_root,
            "--target_col",
            j["target_col"],
            "--init_source",
            j["init_source"],
            "--mode",
            str(j["mode"]),
            "--output_dir",
            str(od),
            "--gpu",
            str(gpu),
            "--delivery_csv",
            str(REPO / args.delivery_live),
            "--delivery_final_csv",
            str(REPO / args.delivery_final),
            "--loss_type",
            args.loss_type,
            "--pos_weight",
            str(args.pos_weight),
        ]
        if needs_resume(od):
            cmd.append("--resume")
        logf = open(od / "runner_stdout.log", "a", encoding="utf-8")
        print(f"[start gpu={gpu}] {' '.join(cmd)}")
        return subprocess.Popen(
            cmd,
            stdout=logf,
            stderr=subprocess.STDOUT,
            cwd=str(REPO),
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
            od = Path(meta["output_dir"])
            print(f"[done gpu={gpu} ret={ret}] {meta['init_source']} m{meta['mode']} {meta['target_col']}")
            slots[i] = (None, None, gpu)
            if qi < len(pending):
                j = pending[qi]
                qi += 1
                p = start_job(j, gpu)
                slots[i] = (p, j, gpu)
        time.sleep(3)

    print("[matrix] 全部任务已结束。")


if __name__ == "__main__":
    main()
