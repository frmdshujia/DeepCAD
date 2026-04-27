#!/usr/bin/env bash
# 停止矩阵调度与所有 stage1_train_one 子进程（紧急用）
set -euo pipefail
echo "[stop] 结束 run_stage1_matrix / stage1_train_one ..."
pkill -f "run_stage1_matrix.py" 2>/dev/null || true
pkill -f "stage1_train_one.py" 2>/dev/null || true
sleep 1
pgrep -af "run_stage1_matrix|stage1_train_one" || echo "[stop] 已无相关进程"
