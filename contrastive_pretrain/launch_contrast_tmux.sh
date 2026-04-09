#!/bin/bash
# ============================================================
# launch_contrast_tmux.sh — 在 tmux 会话中启动对比学习，便于断线后续看日志
#
# 用法：
#   bash contrastive_pretrain/launch_contrast_tmux.sh
#   bash contrastive_pretrain/launch_contrast_tmux.sh contrastive_pretrain/run_contrast.sh
#
# 环境变量：
#   TMUX_SESSION  会话名（默认 contrast_train）
#
# 连接会话：  tmux attach -t contrast_train
# 断开会话：  Ctrl+b 然后按 d（训练继续在后台跑）
# 列表：      tmux ls
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SESSION="${TMUX_SESSION:-contrast_4gpu}"
# 默认全量 4 卡；烟雾测可传：contrastive_pretrain/run_contrast_smoke.sh
RUN_SCRIPT="${1:-${ROOT}/contrastive_pretrain/run_contrast.sh}"

if [[ ! -f "$RUN_SCRIPT" ]]; then
  echo "错误：找不到脚本 $RUN_SCRIPT"
  exit 1
fi

if command -v tmux >/dev/null 2>&1 && tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "tmux 会话 [$SESSION] 已存在。"
  echo "  连接查看: tmux attach -t $SESSION"
  echo "  或先结束: tmux kill-session -t $SESSION"
  exit 0
fi

if ! command -v tmux >/dev/null 2>&1; then
  echo "未安装 tmux，直接前台运行:"
  echo "  bash \"$RUN_SCRIPT\""
  exec bash "$RUN_SCRIPT"
fi

# -d 后台启动；-c 工作目录
tmux new-session -d -s "$SESSION" -c "$ROOT" \
  "bash -c 'echo 工作目录: \$(pwd); echo 运行: $RUN_SCRIPT; echo; bash \"$RUN_SCRIPT\" || true; echo; echo === 已结束。按回车关闭窗口 ===; read _'"

echo "已在 tmux 会话 [$SESSION] 中启动训练脚本。"
echo ""
echo "  查看进度:    tmux attach -t $SESSION"
echo "  断开（后台继续）: Ctrl+b 再按 d"
echo "  结束会话:    tmux kill-session -t $SESSION"
echo ""
