#!/usr/bin/env python3
"""
Mode1 线性头（embedding 已存在）：多作业在多张 GPU 上并行调度。

默认 --gpus 0,1,2,3：最多同时跑 len(gpus) 个任务，每个子进程设置
CUDA_VISIBLE_DEVICES 为对应物理卡；请勿与 train_mode1 内再次覆盖冲突（已修复）。
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

INIT_TO_EMB = {
    "retfound": "emb_retfound.pt",
    "controlled": "emb_controlled.pt",
    "no_residual": "emb_no_residual.pt",
}

# 与 run_mode1_probe_sweep 中验证过的套餐对齐，传给 train_mode1_from_embeddings.py
PROBE_PRESET_ARGS = {
    "default": [],
    "pos_weight_auto": [
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
        "--hparam_tag",
        "pos_weight_auto",
    ],
    # 第一候选 pos_weight_auto + 第二候选 lr3e4_wd1e2（加权 BCE 保留；lr/wd 用 sweep 里温和提升那套）
    "combo_pw_lr": [
        "--loss",
        "bce_pos_weight",
        "--pos_weight",
        "auto",
        "--smoothing",
        "0.0",
        "--lr",
        "0.0003",
        "--weight_decay",
        "0.01",
        "--hparam_tag",
        "combo_pw_lr",
    ],
}


def safe_name(s: str) -> str:
    return re.sub(r"[^0-9a-zA-Z._-]+", "_", s)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_cache_dir", type=str, default="output_dir/stage1_emb_cache")
    ap.add_argument("--out_matrix_dir", type=str, default="output_dir/stage1_mode1_emb_matrix")
    ap.add_argument(
        "--delivery_final_csv",
        type=str,
        default="output_dir/stage1_mode1_emb_delivery_final.csv",
    )
    ap.add_argument("--gpus", type=str, default="0,1,2,3")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch_size", type=int, default=16384)
    ap.add_argument(
        "--skip_done",
        action="store_true",
        help="若输出目录已有 DONE 则跳过",
    )
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--inits",
        type=str,
        default="retfound,controlled,no_residual",
        help="逗号分隔，只调度这些 init（需对应 emb_*.pt 已存在）。例：仅先训 retfound 时写 --inits retfound",
    )
    ap.add_argument(
        "--probe_preset",
        type=str,
        default="default",
        choices=list(PROBE_PRESET_ARGS.keys()),
        help=(
            "default: BCE+标签平滑；pos_weight_auto: 类不平衡加权；"
            "combo_pw_lr: pos_weight_auto + lr3e4_wd1e2（加权 BCE + lr=3e-4, wd=0.01）"
        ),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    n_slots = len(gpus)
    tasks = list(COMPOSITE_TASKS) + list(ICD_PREVALENT_TASKS)

    inits = [x.strip() for x in args.inits.split(",") if x.strip()]
    for init in inits:
        if init not in INIT_TO_EMB:
            print(f"[错误] 未知 init={init!r}，可选: {list(INIT_TO_EMB.keys())}")
            sys.exit(1)
    if not inits:
        print("[错误] --inits 为空")
        sys.exit(1)

    emb_root = REPO / args.emb_cache_dir
    out_root = REPO / args.out_matrix_dir
    deliv = (
        args.delivery_final_csv
        if os.path.isabs(args.delivery_final_csv)
        else str(REPO / args.delivery_final_csv)
    )

    jobs = []
    for init in inits:
        emb_pt = emb_root / INIT_TO_EMB[init]
        if not emb_pt.is_file() and not args.dry_run:
            print(f"[错误] 缺少 embedding 文件: {emb_pt} 请先运行 extract_stage1_embeddings.py")
            sys.exit(1)
        for t in tasks:
            od = out_root / init / safe_name(t)
            jobs.append(
                {
                    "init": init,
                    "target": t,
                    "embedding_pt": str(emb_pt),
                    "output_dir": str(od),
                }
            )

    pending = []
    for j in jobs:
        if args.skip_done and (Path(j["output_dir"]) / "DONE").is_file():
            print(f"[skip DONE] {j['output_dir']}")
            continue
        pending.append(j)

    extra = PROBE_PRESET_ARGS[args.probe_preset]
    print(
        f"[queue] Mode1 训练 {len(pending)}/{len(jobs)} 槽位={n_slots} GPU={gpus} "
        f"probe_preset={args.probe_preset}"
    )

    if args.dry_run:
        for j in pending[:8]:
            print(j)
        return

    py = os.environ.get("PYTHON", sys.executable)
    train_py = REPO / "contrastive_pretrain/stage1_finetune/train_mode1_from_embeddings.py"

    slots: list[tuple[subprocess.Popen | None, dict | None, int]] = [
        (None, None, gpus[i]) for i in range(n_slots)
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
            deliv,
            "--epochs",
            str(args.epochs),
            "--batch_size",
            str(args.batch_size),
        ]
        cmd.extend(extra)
        logf = open(Path(j["output_dir"]) / "train_stdout.log", "a", encoding="utf-8")
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "PYTHONUNBUFFERED": "1"}
        print(f"[start gpu={gpu}] {j['init']} {j['target']}")
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
            print(f"[done gpu={gpu} ret={ret}] {meta['init']} {meta['target']}")
            slots[i] = (None, None, gpu)
            if qi < len(pending):
                j = pending[qi]
                qi += 1
                p = start_job(j, gpu)
                slots[i] = (p, j, gpu)
        time.sleep(2)

    print("[mode1-parallel] 全部训练作业已结束。")


if __name__ == "__main__":
    main()
