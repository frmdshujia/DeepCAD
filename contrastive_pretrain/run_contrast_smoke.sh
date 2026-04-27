#!/bin/bash
# ============================================================
# run_contrast_smoke.sh — 保守「烟雾测」：小样本 + 轻负载 + 单卡
#
# 目的：快速验证 loss 能降、显存不炸，再改用 run_contrast.sh 全量。
#
# 用法：
#   bash contrastive_pretrain/run_contrast_smoke.sh
# 或 tmux 里跑（见 launch_contrast_tmux.sh）
# ============================================================

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${ROOT}/contrastive_pretrain/preprocessed_data"
FUNDUS_CSV="${DATA_DIR}/fundus_table.csv"
CMR_CSV="${DATA_DIR}/cmr_table.csv"
PC_COLS="M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3"
SIGMA=6.5893
FINETUNE="${ROOT}/RETFound_cfp_weights.pth"

# 单卡保守设置（可按机器改 GPU_IDS）
GPU_IDS="${GPU_IDS:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${ROOT}/output_dir/contrast_smoke_${TIMESTAMP}"
LOG_FILE="${OUTPUT_DIR}/train.log"
mkdir -p "${OUTPUT_DIR}"

echo "=============================================="
echo "  对比学习 — 烟雾测（保守）"
echo "  ROOT: ${ROOT}"
echo "  GPU:  ${GPU_IDS}"
echo "  OUT:  ${OUTPUT_DIR}"
echo "  日志: ${LOG_FILE}"
echo "=============================================="

# 说明：
#   --fundus_max_train_samples 500     训练集最多 500 行（路径存在前提下）
#   --cmr_train_max_rows 12000         train CMR bank 随机 1.2 万行（减负）
#   --sgt_temp 0.5                     目标 softmax(S_GT/τ_g) 锐化（1.0=旧版均匀目标）
#   --cmr_sample_k 1024                每步负样本数降低
#   --skip_full_eval                   不做全量 retrieval（省时间）；用 val_loss 选 best
# 验证集仍为全量 val（监控稳定）

# 若当前 shell 的 python 无 torch，请用：
#   conda run -n retfound --no-capture-output bash contrastive_pretrain/run_contrast_smoke.sh
PYTHON="${PYTHON:-python}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${PYTHON}" contrastive_pretrain/main_contrast.py \
    --fundus_csv "${FUNDUS_CSV}" \
    --cmr_csv "${CMR_CSV}" \
    --pc_cols "${PC_COLS}" \
    --sigma "${SIGMA}" \
    --finetune "${FINETUNE}" \
    \
    --fundus_max_train_samples 500 \
    --subset_seed 42 \
    --cmr_train_max_rows 12000 \
    \
    --proj_dim 256 \
    --temperature 0.07 \
    --sgt_temp 0.5 \
    --cmr_sample_k 1024 \
    \
    --batch_size 16 \
    --epochs 12 \
    --warmup_epochs 2 \
    \
    --blr 1e-5 \
    --min_lr 1e-7 \
    --weight_decay 0.05 \
    --layer_decay 0.75 \
    --proj_lr_scale 10.0 \
    --cmr_lr_scale 100.0 \
    --clip_grad 1.0 \
    \
    --patience 8 \
    --metric_for_best val_loss \
    --eval_freq 999 \
    --skip_full_eval \
    --save_freq 999 \
    \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 4 \
    --gpu "${GPU_IDS}" \
    --desc "smoke_${TIMESTAMP}" \
    2>&1 | tee "${LOG_FILE}"

echo ""
echo "烟雾测结束。若 loss 正常、无 OOM，可加大样本并运行:"
echo "  bash contrastive_pretrain/run_contrast.sh"
echo "兼容 encoder（若产生 best）:"
echo "  ${OUTPUT_DIR}/contrast_pretrain_encoder_best.pth"
