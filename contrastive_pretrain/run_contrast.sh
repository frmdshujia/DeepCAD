#!/bin/bash
# ============================================================
# run_contrast.sh — 对比学习预训练（默认 4 卡 DDP）
#
# 启动（在仓库根目录或任意目录均可）：
#   bash contrastive_pretrain/run_contrast.sh
#
# 环境：默认使用 retfound conda 的 Python（与 requirement.txt 一致）。
#       可覆盖：export PYTHON=/path/to/python
# ============================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# 必须激活 retfound，否则 torch 可能 ImportError（与项目 main_finetune 一致）
if [ -f "/data/home/shujia/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck source=/dev/null
  source "/data/home/shujia/miniconda3/etc/profile.d/conda.sh"
  conda activate retfound
fi
PYTHON="${PYTHON:-$(command -v python)}"

# PyTorch 1.8 无 torchrun 时，用 torch.distributed.launch

# ─── 数据路径 ────────────────────────────────────────────────
DATA_DIR="${REPO_ROOT}/contrastive_pretrain/preprocessed_data"
FUNDUS_CSV="${DATA_DIR}/fundus_table.csv"
CMR_CSV="${DATA_DIR}/cmr_table.csv"
PC_COLS="M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3"
SIGMA=6.5893

FINETUNE="${REPO_ROOT}/RETFound_cfp_weights.pth"

# ─── GPU（默认 4 卡）──────────────────────────────────────────
GPU_IDS="${GPU_IDS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
N_GPU=$(echo "${GPU_IDS}" | tr ',' '\n' | grep -c . || true)

# ─── 输出 ────────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="${REPO_ROOT}/output_dir/contrast_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/tb_logs"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "=============================================="
echo "  对比学习预训练 | 工作目录: ${REPO_ROOT}"
echo "  GPU: ${GPU_IDS} (×${N_GPU})"
echo "  Output: ${OUTPUT_DIR}"
echo "  Sigma: ${SIGMA}"
echo "  Python: ${PYTHON}"
echo "=============================================="

ARGS=(
  contrastive_pretrain/main_contrast.py
  --fundus_csv    "${FUNDUS_CSV}"
  --cmr_csv       "${CMR_CSV}"
  --pc_cols       "${PC_COLS}"
  --sigma         "${SIGMA}"
  --finetune      "${FINETUNE}"
  --proj_dim      256
  --temperature   0.07
  --cmr_sample_k  4096
  --batch_size    64
  --epochs        100
  --warmup_epochs 10
  --blr           1e-5
  --min_lr        1e-7
  --weight_decay  0.05
  --layer_decay   0.75
  --proj_lr_scale 10.0
  --cmr_lr_scale  100.0
  --clip_grad     1.0
  --patience      12
  --eval_freq     5
  --save_freq     10
  --output_dir    "${OUTPUT_DIR}"
  --log_dir       "${LOG_DIR}"
  --num_workers   8
  --gpu           "${GPU_IDS}"
  --desc          "run_${TIMESTAMP}"
)

if command -v torchrun >/dev/null 2>&1; then
  echo "[launch] torchrun (${N_GPU} proc)"
  torchrun --nproc_per_node="${N_GPU}" --master_port=29501 "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
else
  echo "[launch] ${PYTHON} -m torch.distributed.launch (${N_GPU} proc, PyTorch 1.x)"
  "${PYTHON}" -m torch.distributed.launch \
    --nproc_per_node="${N_GPU}" \
    --master_port=29501 \
    --use_env \
    "${ARGS[@]}" 2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

echo ""
echo "训练结束（或早停）。日志: ${OUTPUT_DIR}/train.log"
echo "下游 encoder: ${OUTPUT_DIR}/contrast_pretrain_encoder_best.pth（若产生 best）"
