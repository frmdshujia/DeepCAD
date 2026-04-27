#!/usr/bin/env bash
# 实验三：分层 S_GT 采样（stratified）对比学习训练 — 后台长期运行
# 注意：此脚本曾为省显存单卡小 batch，与 e2e 基线超参不一致。
# 若要做「唯一变量=分层采样」的对照，请用 run_stratified_e2e_controlled.sh
# 用法：bash contrastive_pretrain/run_exp3_stratified_background.sh
# 关闭终端后进程仍继续（nohup）；日志：output_dir/exp3_stratified_<时间戳>/nohup.log

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DATA_DIR="${ROOT}/contrastive_pretrain/preprocessed_data"
FUNDUS_CSV="${DATA_DIR}/fundus_table.csv"
CMR_CSV="${DATA_DIR}/cmr_table.csv"
PC_COLS="M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3"
SIGMA=6.5893
FINETUNE="${ROOT}/RETFound_cfp_weights.pth"
# 默认优先使用 conda 环境 retfound（与 run_contrast_smoke.sh 一致）；可 export PYTHON=... 覆盖
if [[ -z "${PYTHON:-}" ]]; then
  if command -v conda >/dev/null 2>&1; then
    PYTHON="conda run -n retfound --no-capture-output python"
  else
    PYTHON="python3"
  fi
fi

TS="$(date +%Y%m%d_%H%M%S)"
OUT="${ROOT}/output_dir/exp3_stratified_${TS}"
BUCKET_PKL="${OUT}/stratified_buckets.pkl"
mkdir -p "${OUT}"

echo "=========================================="
echo "  Exp3 Stratified | OUT=${OUT}"
echo "=========================================="

echo "[1/2] 预计算分层分桶（train fundus EID × train CMR 全表）..."
${PYTHON} contrastive_pretrain/precompute_stratified_buckets.py \
  --fundus_csv "${FUNDUS_CSV}" \
  --cmr_csv "${CMR_CSV}" \
  --pc_cols "${PC_COLS}" \
  --sigma "${SIGMA}" \
  --thresh_high 0.8 \
  --thresh_low 0.3 \
  --out_pkl "${BUCKET_PKL}"

echo "[2/2] 启动对比学习（nohup，cmr_train_max_rows=0 全量 bank）..."
# 显存：默认 batch_size=16 张 fundus / step，cmr_sample_k=2048（原 64×4096 易 OOM）
# 可 export BATCH_SIZE=8 CMR_SAMPLE_K=1024 再试
BATCH_SIZE="${BATCH_SIZE:-16}"
CMR_SAMPLE_K="${CMR_SAMPLE_K:-2048}"
GPU_IDS="${GPU_IDS:-0}"
CUDA_VISIBLE_DEVICES="${GPU_IDS}" nohup ${PYTHON} contrastive_pretrain/main_contrast.py \
  --fundus_csv "${FUNDUS_CSV}" \
  --cmr_csv "${CMR_CSV}" \
  --pc_cols "${PC_COLS}" \
  --sigma "${SIGMA}" \
  --finetune "${FINETUNE}" \
  --cmr_train_max_rows 0 \
  --cmr_sample_mode stratified \
  --stratified_buckets_pkl "${BUCKET_PKL}" \
  --strat_neg_frac_high 0.333333 \
  --strat_neg_frac_low 0.333333 \
  --cmr_sample_k "${CMR_SAMPLE_K}" \
  --batch_size "${BATCH_SIZE}" \
  --epochs 100 \
  --metric_for_best val_loss \
  --eval_freq 999 \
  --skip_full_eval \
  --output_dir "${OUT}" \
  --num_workers 8 \
  --gpu "${GPU_IDS}" \
  --desc "exp3_stratified_${TS}" \
  >> "${OUT}/nohup.log" 2>&1 &

echo $! > "${OUT}/train.pid"
echo "PID $(cat "${OUT}/train.pid") 已写入 ${OUT}/train.pid"
echo "日志: ${OUT}/nohup.log"
echo "查看: tail -f ${OUT}/nohup.log"
