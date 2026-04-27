#!/usr/bin/env bash
# 单进程顺序跑 4 目标 × (retfound + contrastive) = 8 次训练，降低内存/供电压力。
# 保守设置：小 batch、梯度累积、较低学习率、num_workers=0。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
export PYTHONUNBUFFERED=1

: "${CONDA_SH:=${HOME}/miniconda3/etc/profile.d/conda.sh}"
# shellcheck source=/dev/null
source "$CONDA_SH"
conda activate retfound

CKPT="${CKPT:-output_dir/contrast_finetune_e2e_20260412_174935/checkpoint_best.pth}"
DEVICE="${DEVICE:-cuda:0}"
OUT="${OUT:-output_dir/end2end_conservative_4exp_AB.csv}"
CKPT_DIR="${CKPT_DIR:-output_dir/end2end_conservative_ckpt}"

exec python -u contrastive_pretrain/end2end_stage2_ceiling.py \
  --run_all_inits \
  --targets_file contrastive_pretrain/conservative_4exp_targets.txt \
  --contrastive_ckpt "$CKPT" \
  --ckpt_dir "$CKPT_DIR" \
  --output_csv "$OUT" \
  --epochs 40 \
  --patience 10 \
  --batch_size 2 \
  --accumulation_steps 8 \
  --lr_backbone 1e-5 \
  --lr_head_group 1e-4 \
  --weight_decay 0.05 \
  --clip_grad 1.0 \
  --num_workers 0 \
  --device "$DEVICE"
