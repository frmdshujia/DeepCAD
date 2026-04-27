#!/usr/bin/env python3
"""
在 negative_downsampled_tasks/ 每任务单表上跑 stage1_train_one（默认 Mode2≈解冻约 1/3 ViT block）。

默认：冠心病相关 3 任务 ×（retfound + controlled）× 组合超参
  bce_pos_weight(auto) + lr=3e-4 + wd=0.01（与线性探针 combo_pw_lr 一致）

四卡并行：子进程设置 CUDA_VISIBLE_DEVICES，train_one 内传 --gpu 0。
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

# 与冠心病/缺血性心脏病最相关的三个默认任务（表需存在于 negative_downsampled_tasks/）
CHD_TASKS_DEFAULT = [
    "composite_ischemic_hd",  # 缺血性心脏病复合
    "prevalent_I20",  # 心绞痛等
    "prevalent_I25",  # 慢性缺血性心脏病
]

NARROW_ROOT = (
    REPO / "contrastive_pretrain/preprocessed_data/negative_downsampled_tasks"
)


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", s)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--tasks",
        type=str,
        default=",".join(CHD_TASKS_DEFAULT),
        help="逗号分隔；对应 negative_downsampled_tasks/<task>.csv",
    )
    ap.add_argument("--inits", type=str, default="retfound,controlled")
    ap.add_argument("--mode", type=int, default=2, choices=[1, 2, 3])
    ap.add_argument(
        "--base_out",
        type=str,
        default="output_dir/stage1_mode2_combo_chd_narrow",
    )
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--warmup_epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min_lr", type=float, default=1e-6)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--layer_decay", type=float, default=0.75)
    ap.add_argument(
        "--delivery_live",
        type=str,
        default="output_dir/stage1_narrow_combo_delivery_live.csv",
    )
    ap.add_argument(
        "--delivery_final",
        type=str,
        default="output_dir/stage1_narrow_combo_delivery_final.csv",
    )
    ap.add_argument("--skip_done", action="store_true")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


def should_skip(od: Path) -> bool:
    return (od / "DONE").is_file()


def needs_resume(od: Path) -> bool:
    return (od / "checkpoint_last.pth").is_file() and not (od / "DONE").is_file()


def main():
    args = parse_args()
    tasks = [x.strip() for x in args.tasks.split(",") if x.strip()]
    inits = [x.strip() for x in args.inits.split(",") if x.strip()]
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    n_slots = len(gpus)

    jobs = []
    for t in tasks:
        csv_path = NARROW_ROOT / f"{t}.csv"
        if not csv_path.is_file():
            print(f"[错误] 缺少表: {csv_path}")
            sys.exit(1)
        for init in inits:
            od = REPO / args.base_out / init / f"mode{args.mode}" / safe_name(t)
            jobs.append(
                {
                    "target_col": t,
                    "init_source": init,
                    "narrow_csv": str(csv_path.relative_to(REPO)),
                    "output_dir": str(od),
                }
            )

    pending = []
    for j in jobs:
        if args.skip_done and should_skip(Path(j["output_dir"])):
            print(f"[skip DONE] {j['output_dir']}")
            continue
        pending.append(j)

    print(
        f"[queue] narrow+combo_ft 待跑 {len(pending)}/{len(jobs)} "
        f"槽位={n_slots} GPU={gpus} mode={args.mode} lr={args.lr} wd={args.weight_decay}"
    )

    if args.dry_run:
        for j in pending:
            print(j)
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
            "-u",
            str(script),
            "--narrow_task_csv",
            j["narrow_csv"],
            "--fundus_root",
            "/data/home/home6/fundus_data/UKB/fundus_images",
            "--target_col",
            j["target_col"],
            "--init_source",
            j["init_source"],
            "--mode",
            str(args.mode),
            "--output_dir",
            j["output_dir"],
            "--gpu",
            "0",
            "--epochs",
            str(args.epochs),
            "--warmup_epochs",
            str(args.warmup_epochs),
            "--batch_size",
            str(args.batch_size),
            "--lr",
            str(args.lr),
            "--min_lr",
            str(args.min_lr),
            "--weight_decay",
            str(args.weight_decay),
            "--layer_decay",
            str(args.layer_decay),
            "--loss_type",
            "bce_pos_weight",
            "--pos_weight",
            "0",
            "--smoothing",
            "0.0",
            "--delivery_csv",
            str(REPO / args.delivery_live)
            if not os.path.isabs(args.delivery_live)
            else args.delivery_live,
            "--delivery_final_csv",
            str(REPO / args.delivery_final)
            if not os.path.isabs(args.delivery_final)
            else args.delivery_final,
        ]
        if needs_resume(od):
            cmd.append("--resume")
        logf = open(od / "runner_stdout.log", "a", encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        print(f"[start gpu={gpu}] {j['init_source']} {j['target_col']}")
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
            print(
                f"[done gpu={gpu} ret={ret}] {meta['init_source']} "
                f"{meta['target_col']}"
            )
            slots[i] = (None, None, gpu)
            if qi < len(pending):
                j = pending[qi]
                qi += 1
                p = start_job(j, gpu)
                slots[i] = (p, j, gpu)
        time.sleep(3)

    print("[narrow_combo_ft] 全部任务已结束。")


if __name__ == "__main__":
    main()
