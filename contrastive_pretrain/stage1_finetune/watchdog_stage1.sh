#!/usr/bin/env bash
# 若矩阵调度进程未在运行，则重新拉起（用于机器定时重启后恢复队列）。
# 用法：在 crontab 中每 5 分钟执行一次，或 systemd timer。
set -euo pipefail
REPO="/data/home/shujia/CHD/model_train/RETFound_MAE-main"
PY="${PYTHON:-/data/home/shujia/miniconda3/envs/retfound/bin/python}"
RUNNER="$REPO/contrastive_pretrain/stage1_finetune/run_stage1_matrix.py"
MARKER="${1:-stage1_matrix_runner}"

if pgrep -f "$RUNNER" >/dev/null 2>&1; then
  echo "[watchdog] $RUNNER 已在运行"
  exit 0
fi

LOG="$REPO/output_dir/${MARKER}_watchdog.log"
mkdir -p "$(dirname "$LOG")"
echo "[$(date -Iseconds)] 启动 run_stage1_matrix" >>"$LOG"
cd "$REPO"
nohup "$PY" -u "$RUNNER" >>"$LOG" 2>&1 &
echo "[watchdog] 已后台启动，日志: $LOG"
