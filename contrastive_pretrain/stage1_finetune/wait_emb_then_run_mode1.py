#!/usr/bin/env python3
"""
在后台等待三份 embedding 全部落盘且可加载后，自动启动 Mode1 四卡并行训练。

典型用法（睡前一条命令）：
  cd REPO && nohup env PYTHONUNBUFFERED=1 \\
    python contrastive_pretrain/stage1_finetune/wait_emb_then_run_mode1.py \\
    --poll_seconds 120 --gpus 0,1,2,3 --skip_done \\
    >> output_dir/wait_then_mode1.log 2>&1 &

说明：
- extract 脚本只在结束时一次性 torch.save，因此「文件出现且能 load」≈ 该份导出完成。
- 其余参数会原样传给 run_mode1_emb_train_parallel.py。
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

EMB_NAMES = ("emb_retfound.pt", "emb_controlled.pt", "emb_no_residual.pt")


def _torch_load_dict(path: Path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def embedding_ok(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.stat().st_size < 1024 * 1024:
        return False
    try:
        d = _torch_load_dict(path)
        if not isinstance(d, dict) or "embeddings" not in d:
            return False
        e = d["embeddings"]
        return hasattr(e, "shape") and e.ndim == 2 and e.shape[0] > 0
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser(
        description="等待三份 embedding 就绪后启动 run_mode1_emb_train_parallel.py"
    )
    ap.add_argument(
        "--emb_cache_dir",
        type=str,
        default="output_dir/stage1_emb_cache",
        help="相对 REPO 根目录",
    )
    ap.add_argument(
        "--poll_seconds",
        type=float,
        default=120.0,
        help="轮询间隔（秒）",
    )
    ap.add_argument(
        "--repo",
        type=str,
        default=str(REPO),
        help="仓库根目录",
    )
    ap.add_argument(
        "--status_log",
        type=str,
        default="",
        help="可选：将状态行追加写入该文件",
    )
    args, train_rest = ap.parse_known_args()

    repo = Path(args.repo)
    emb_dir = repo / args.emb_cache_dir
    targets = [emb_dir / name for name in EMB_NAMES]

    train_py = repo / "contrastive_pretrain/stage1_finetune/run_mode1_emb_train_parallel.py"
    if not train_py.is_file():
        print(f"[错误] 未找到 {train_py}", file=sys.stderr)
        sys.exit(1)

    py = os.environ.get("PYTHON", sys.executable)

    def log(msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        if args.status_log:
            p = repo / args.status_log if not os.path.isabs(args.status_log) else Path(args.status_log)
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    log(f"开始等待 embedding（轮询 {args.poll_seconds}s）: {[str(p) for p in targets]}")

    while True:
        ok = [embedding_ok(p) for p in targets]
        if all(ok):
            break
        miss = [targets[i].name for i in range(3) if not ok[i]]
        log(f"尚未就绪: {miss}")
        time.sleep(max(5.0, float(args.poll_seconds)))

    log("三份 embedding 已就绪，启动 Mode1 四卡训练…")

    cmd = [py, "-u", str(train_py)]
    if train_rest:
        cmd.extend(train_rest)
    else:
        cmd.extend(["--gpus", "0,1,2,3", "--skip_done"])

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    ret = subprocess.run(cmd, cwd=str(repo), env=env)
    log(f"run_mode1_emb_train_parallel 退出码={ret.returncode}")
    sys.exit(ret.returncode)


if __name__ == "__main__":
    main()
