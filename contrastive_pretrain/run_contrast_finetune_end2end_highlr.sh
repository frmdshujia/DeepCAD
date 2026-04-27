#!/bin/bash
# ============================================================
# run_contrast_finetune_end2end_highlr.sh
#
# 在「基线端到端实验」之上，用 **更大 blr** 重新训练一版，其它设置尽量与
# contrast_finetune_e2e_20260412_174935 一致（4 卡、batch 16、K=2048、300 epoch 等）。
#
# **输出目录始终带时间戳**，不会写入 preserved_weights/，也不会覆盖旧目录。
# 基线最佳权重已归档至:
#   output_dir/preserved_weights/e2e_run_20260412_174935/
#
# 默认 blr=8e-5（约为原 4e-5 的 2×）；可用环境变量 BLR=1e-4 等覆盖。
#
# 用法（仓库根目录）:
#   bash contrastive_pretrain/run_contrast_finetune_end2end_highlr.sh
#   BLR=1e-4 bash contrastive_pretrain/run_contrast_finetune_end2end_highlr.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "/data/home/shujia/miniconda3/etc/profile.d/conda.sh" ]; then
  source "/data/home/shujia/miniconda3/etc/profile.d/conda.sh"
  conda activate retfound 2>/dev/null || true
fi
PYTHON="${PYTHON:-$(command -v python)}"

DATA_DIR="${REPO_ROOT}/contrastive_pretrain/preprocessed_data"
FUNDUS_CSV="${DATA_DIR}/fundus_table.csv"
CMR_CSV="${DATA_DIR}/cmr_table.csv"
PC_COLS="M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3"
FINETUNE="${REPO_ROOT}/RETFound_cfp_weights.pth"

SIGMA="${SIGMA:-6.5893}"
SGT_TEMP="${SGT_TEMP:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.15}"

# 与基线 e2e 一致：四卡
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
N_GPU=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c . || true)

BATCH_SIZE="${BATCH_SIZE:-16}"
CMR_SAMPLE_K="${CMR_SAMPLE_K:-2048}"

# 基线为 4e-5；此处默认加倍，可按机器与稳定性再调
BLR="${BLR:-8e-5}"

METRIC_FOR_BEST="${METRIC_FOR_BEST:-gt_pred_spearman}"
EPOCHS="${EPOCHS:-300}"
EVAL_FREQ="${EVAL_FREQ:-5}"
PATIENCE="${PATIENCE:-35}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
# 独立目录，绝不覆盖旧实验
OUTPUT_DIR="${REPO_ROOT}/output_dir/contrast_finetune_e2e_highlr_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/tb_logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

EFF_BS=$(( BATCH_SIZE * N_GPU ))

echo "=============================================="
echo "  端到端对比学习 | 高学习率实验（新目录）"
echo "  eff_bs=${EFF_BS}  batch/GPU=${BATCH_SIZE}  K=${CMR_SAMPLE_K}"
echo "  blr=${BLR}  (基线曾为 4e-5；本脚本默认 8e-5)"
echo "  sigma=${SIGMA}  sgt_temp=${SGT_TEMP}  temp=${TEMPERATURE}"
echo "  metric_for_best=${METRIC_FOR_BEST}"
echo "  OUT: ${OUTPUT_DIR}"
echo "  基线权重归档: output_dir/preserved_weights/e2e_run_20260412_174935/"
echo "=============================================="

MASTER_PORT="${MASTER_PORT:-29512}"

ARGS=(
  contrastive_pretrain/main_contrast.py
  --fundus_csv "${FUNDUS_CSV}"
  --cmr_csv "${CMR_CSV}"
  --pc_cols "${PC_COLS}"
  --sigma "${SIGMA}"
  --finetune "${FINETUNE}"
  --loss_type soft
  --sgt_temp "${SGT_TEMP}"
  --temperature "${TEMPERATURE}"
  --proj_dim 256
  --cmr_sample_k "${CMR_SAMPLE_K}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --warmup_epochs 5
  --blr "${BLR}"
  --min_lr 1e-7
  --weight_decay 0.05
  --layer_decay 0.75
  --proj_lr_scale 10.0
  --cmr_lr_scale 100.0
  --clip_grad 1.0
  --patience "${PATIENCE}"
  --eval_freq "${EVAL_FREQ}"
  --metric_for_best "${METRIC_FOR_BEST}"
  --save_freq 10
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --num_workers 4
  --gpu "${GPU_IDS}"
  --desc "e2e_highlr_${TIMESTAMP}"
)

if [ "${N_GPU}" -gt 1 ]; then
  if command -v torchrun >/dev/null 2>&1; then
    echo "[launch] torchrun --nproc_per_node=${N_GPU} port=${MASTER_PORT}"
    torchrun --nproc_per_node="${N_GPU}" --master_port="${MASTER_PORT}" "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
  else
    "${PYTHON}" -m torch.distributed.launch --nproc_per_node="${N_GPU}" --master_port="${MASTER_PORT}" --use_env "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
  fi
else
  "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

echo ""
echo "结束。日志: ${OUTPUT_DIR}/train.log"
echo "本实验权重仅写入上述目录，不会覆盖 preserved_weights 或旧 contrast_finetune_e2e_* 目录。"
