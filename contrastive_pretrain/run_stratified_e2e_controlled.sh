#!/usr/bin/env bash
# =============================================================================
# 分层负采样对照实验：与 output_dir/contrast_finetune_e2e_20260412_174935/args.json
# 中记录的训练设置逐项对齐；唯一故意改变的变量是 CMR 负样本策略（random → stratified）。
#
# 对齐项（与当时 e2e 一致）：
#   - 4×GPU × batch_size 16 × cmr_sample_k 2048，blr=4e-5，temperature=0.15，sgt_temp=0.5
#   - epochs=300，warmup=5，patience=35，eval_freq=5
#   - metric_for_best=gt_pred_spearman，不做 skip_full_eval（需周期性完整 eval）
#   - sigma、proj、优化器与正则与 args.json 一致
#
# 唯一实验变量：
#   --cmr_sample_mode stratified + 预计算桶 pkl
#   STRAT_NEG_FRAC_HIGH / STRAT_NEG_FRAC_LOW：负样本池中 high/low 桶目标占比，余量为 mid
#   默认 0.40 / 0.40（中间桶约 20%，比各占 1/3 时更少）
#
# 用法（仓库根）：
#   bash contrastive_pretrain/run_stratified_e2e_controlled.sh
#   GPU_IDS=0,1,2,3 STRAT_NEG_FRAC_HIGH=0.35 STRAT_NEG_FRAC_LOW=0.35 bash ...
# =============================================================================
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

# —— 与 e2e_20260412_174935 args.json 一致 —— #
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
N_GPU=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c . || true)

BATCH_SIZE="${BATCH_SIZE:-16}"
CMR_SAMPLE_K="${CMR_SAMPLE_K:-2048}"
SIGMA="${SIGMA:-6.5893}"
SGT_TEMP="${SGT_TEMP:-0.5}"
TEMPERATURE="${TEMPERATURE:-0.15}"
BLR="${BLR:-4e-5}"
EPOCHS="${EPOCHS:-300}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-5}"
PATIENCE="${PATIENCE:-35}"
EVAL_FREQ="${EVAL_FREQ:-5}"
METRIC_FOR_BEST="${METRIC_FOR_BEST:-gt_pred_spearman}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# —— 仅分层实验可调：high/low 占比略升 → mid 减少 —— #
STRAT_NEG_FRAC_HIGH="${STRAT_NEG_FRAC_HIGH:-0.40}"
STRAT_NEG_FRAC_LOW="${STRAT_NEG_FRAC_LOW:-0.40}"

TS=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${REPO_ROOT}/output_dir/exp_stratified_e2e_controlled_${TS}"
LOG_DIR="${OUTPUT_DIR}/tb_logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"
BUCKET_PKL="${OUTPUT_DIR}/stratified_buckets.pkl"

echo "=============================================="
echo "  分层负采样 | 其余与 e2e_20260412_174935 对齐"
echo "  OUT=${OUTPUT_DIR}"
echo "  GPU_IDS=${GPU_IDS}  N_GPU=${N_GPU}"
echo "  batch/GPU=${BATCH_SIZE}  K=${CMR_SAMPLE_K}  blr=${BLR}"
echo "  temp=${TEMPERATURE}  sgt_temp=${SGT_TEMP}  metric_for_best=${METRIC_FOR_BEST}"
echo "  strat: high=${STRAT_NEG_FRAC_HIGH} low=${STRAT_NEG_FRAC_LOW} (mid=余量)"
echo "=============================================="

echo "[1/2] 预计算分层桶（sigma=${SIGMA}，与训练一致）..."
"${PYTHON}" contrastive_pretrain/precompute_stratified_buckets.py \
  --fundus_csv "${FUNDUS_CSV}" \
  --cmr_csv "${CMR_CSV}" \
  --pc_cols "${PC_COLS}" \
  --sigma "${SIGMA}" \
  --thresh_high 0.8 \
  --thresh_low 0.3 \
  --out_pkl "${BUCKET_PKL}"

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
  --cmr_train_max_rows 0
  --cmr_sample_k "${CMR_SAMPLE_K}"
  --cmr_sample_mode stratified
  --stratified_buckets_pkl "${BUCKET_PKL}"
  --strat_neg_frac_high "${STRAT_NEG_FRAC_HIGH}"
  --strat_neg_frac_low "${STRAT_NEG_FRAC_LOW}"
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS}"
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
  --num_workers "${NUM_WORKERS}"
  --gpu "${GPU_IDS}"
  --desc "stratified_e2e_controlled_${TS}"
)

echo "[2/2] 启动训练（与 e2e 相同 launch 方式，无 --skip_full_eval）..."
if [ "${N_GPU}" -gt 1 ]; then
  if command -v torchrun >/dev/null 2>&1; then
    echo "[launch] torchrun --nproc_per_node=${N_GPU}"
    torchrun --nproc_per_node="${N_GPU}" --master_port="${MASTER_PORT:-29521}" "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
  else
    "${PYTHON}" -m torch.distributed.launch --nproc_per_node="${N_GPU}" --master_port="${MASTER_PORT:-29521}" --use_env "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
  fi
else
  "${PYTHON}" "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

echo ""
echo "结束。日志: ${OUTPUT_DIR}/train.log"
echo "最佳权重: ${OUTPUT_DIR}/checkpoint_best.pth（指标: ${METRIC_FOR_BEST}）"
