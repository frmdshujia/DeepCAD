#!/usr/bin/env bash
# 端到端「天花板」实验：retfound + contrastive × 全部 discover 目标
# 预计耗时很长（数十小时级，视 GPU 而定）；建议 nohup 或 tmux。
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source ~/miniconda3/etc/profile.d/conda.sh
conda activate retfound

CKPT="${CKPT:-${ROOT}/output_dir/contrast_finetune_e2e_20260412_174935/checkpoint_best.pth}"
OUT="${OUT:-${ROOT}/output_dir/end2end_ceiling_AB.csv}"

python contrastive_pretrain/end2end_stage2_ceiling.py \
  --contrastive_ckpt "${CKPT}" \
  --run_all_inits \
  --output_csv "${OUT}" \
  --epochs "${EPOCHS:-40}" \
  --patience "${PATIENCE:-10}" \
  --batch_size "${BATCH:-4}" \
  --accumulation_steps "${ACC:-4}" \
  --lr_backbone "${LRB:-2e-5}" \
  --lr_head_group "${LRH:-2e-4}" \
  --device cuda:0

echo "Done: ${OUT}"
