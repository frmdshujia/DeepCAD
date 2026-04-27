#!/bin/bash
# ============================================================
# run_contrast_finetune_end2end.sh
#
# RETFound ViT 全参微调 + 投影头 + CMR MLP（双塔均可训练），不再使用冻结特征。
# 损失与 fast 上较优设置对齐：soft InfoNCE + σ / τ_g。
# 验证指标：默认按 val 上 gt_pred Spearman 保存 checkpoint_best（可用环境变量改）。
#
# 学习率说明（与 main_contrast 一致）：
#   实际 backbone 顶层参考 lr = blr × (batch×GPU数) / 256
#   投影头 lr = 上式 × proj_lr_scale（默认 10）
#   CMR MLP lr = 上式 × cmr_lr_scale（默认 100）
# 多卡 eff_bs 大时 blr 要小；单卡 eff_bs 小时需略增大 blr 才能与「4×16≈64」量级对齐。
#
# 探索建议（每次改 BLR 或 BATCH_SIZE 单独实验、看 TensorBoard eval/*）：
#   - 相关系数震荡或降：略降 blr（如 8e-5 → 5e-5）或减小 proj_lr_scale / cmr_lr_scale
#   - loss 几乎不动：略升 blr 或增大 batch（显存允许时）
#   - OOM：降 BATCH_SIZE 或 CMR_SAMPLE_K，并相应按 eff_bs 重算 blr
#
# 用法（仓库根目录）：
#   bash contrastive_pretrain/run_contrast_finetune_end2end.sh
#   METRIC_FOR_BEST=gt_pred_pearson BLR=8e-5 bash contrastive_pretrain/run_contrast_finetune_end2end.sh
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "/data/home/shujia/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "/data/home/shujia/miniconda3/etc/profile.d/conda.sh"
  conda activate retfound 2>/dev/null || true
fi
PYTHON="${PYTHON:-$(command -v python)}"

DATA_DIR="${REPO_ROOT}/contrastive_pretrain/preprocessed_data"
FUNDUS_CSV="${DATA_DIR}/fundus_table.csv"
CMR_CSV="${DATA_DIR}/cmr_table.csv"
PC_COLS="M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3"
FINETUNE="${REPO_ROOT}/RETFound_cfp_weights.pth"

# 与 fast 上 Pearson 较优的 soft 设置对齐（可改环境变量覆盖）
SIGMA="${SIGMA:-6.5893}"
SGT_TEMP="${SGT_TEMP:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.15}"

# 单卡默认；多卡请设 GPU_IDS="0,1,2,3" 并相应调小 BLR
GPU_IDS="${GPU_IDS:-0}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
N_GPU=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c . || true)

BATCH_SIZE="${BATCH_SIZE:-16}"
# 全图微调显存压力大时可改为 1024 或 512
CMR_SAMPLE_K="${CMR_SAMPLE_K:-1024}"

# 单卡 batch=16 → eff=16 → blr=1.6e-4 时 顶层 lr≈1e-5（与 eff=64, blr=4e-5 同量级）
EFF_BS=$(( BATCH_SIZE * N_GPU ))
BLR="${BLR:-1.6e-4}"

METRIC_FOR_BEST="${METRIC_FOR_BEST:-gt_pred_spearman}"
EPOCHS="${EPOCHS:-80}"
EVAL_FREQ="${EVAL_FREQ:-5}"
PATIENCE="${PATIENCE:-15}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${REPO_ROOT}/output_dir/contrast_finetune_e2e_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/tb_logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "=============================================="
echo "  端到端对比学习 | RETFound 全参 + 双塔"
echo "  eff_bs=${EFF_BS}  batch/GPU=${BATCH_SIZE}  K=${CMR_SAMPLE_K}"
echo "  blr=${BLR}  sigma=${SIGMA}  sgt_temp=${SGT_TEMP}  temp=${TEMPERATURE}"
echo "  metric_for_best=${METRIC_FOR_BEST}"
echo "  OUT: ${OUTPUT_DIR}"
echo "=============================================="

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
  --desc "e2e_${TIMESTAMP}"
)

# 显式不传 --freeze_backbone，ViT 参与训练
if [ "${N_GPU}" -gt 1 ]; then
  if command -v torchrun >/dev/null 2>&1; then
    echo "[launch] torchrun --nproc_per_node=${N_GPU}"
    torchrun --nproc_per_node="${N_GPU}" --master_port=29511 "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
  else
    echo "[launch] python -m torch.distributed.launch"
    "${PYTHON}" -m torch.distributed.launch --nproc_per_node="${N_GPU}" --master_port=29511 --use_env "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
  fi
else
  "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

echo ""
echo "结束。日志: ${OUTPUT_DIR}/train.log"
echo "最佳权重: ${OUTPUT_DIR}/checkpoint_best.pth（指标: ${METRIC_FOR_BEST}）"
echo "兼容 encoder: ${OUTPUT_DIR}/contrast_pretrain_encoder_best.pth"
