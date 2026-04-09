#!/bin/bash
# ============================================================
# run_contrast.sh — 对比学习预训练启动脚本
#
# 使用前请根据实际情况修改以下变量：
#   FUNDUS_CSV, CMR_CSV, PC_COLS, SIGMA, FINETUNE, GPU_IDS
#
# 启动方式：
#   bash contrastive_pretrain/run_contrast.sh
# ============================================================

set -e  # 任意命令失败即退出

# ─── 必填项：数据 Agent 交付后填入 ───────────────────────────
FUNDUS_CSV="/path/to/fundus_table.csv"           # Fundus 表路径
CMR_CSV="/path/to/cmr_table.csv"                 # CMR 表路径
PC_COLS="M1_PC1,M1_PC2,M1_PC3,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M3_PC3,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1"
                                                  # ← 替换为实际 14 列列名（逗号分隔）
SIGMA=12.374                                      # ← 替换为数据 Agent 计算的 σ 值

# ─── 模型权重（已知）───────────────────────────────────────
FINETUNE="/data/home/shujia/CHD/model_train/RETFound_MAE-main/RETFound_cfp_weights.pth"

# ─── GPU 设置 ───────────────────────────────────────────────
GPU_IDS="0,1,2,3"                                # 可用 GPU 编号（逗号分隔）
N_GPU=$(echo $GPU_IDS | tr ',' '\n' | wc -l)    # 自动计算 GPU 数量

# ─── 输出目录 ────────────────────────────────────────────────
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="./output_dir/contrast_${TIMESTAMP}"
LOG_DIR="${OUTPUT_DIR}/tb_logs"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "=============================================="
echo "  对比学习预训练"
echo "  GPU: ${GPU_IDS} (×${N_GPU})"
echo "  Output: ${OUTPUT_DIR}"
echo "  Sigma: ${SIGMA}"
echo "=============================================="

# ─── 启动训练 ────────────────────────────────────────────────
torchrun \
    --nproc_per_node=${N_GPU} \
    --master_port=29501 \
    contrastive_pretrain/main_contrast.py \
    --fundus_csv    "${FUNDUS_CSV}" \
    --cmr_csv       "${CMR_CSV}" \
    --pc_cols       "${PC_COLS}" \
    --sigma         ${SIGMA} \
    --finetune      "${FINETUNE}" \
    \
    --proj_dim      256 \
    --temperature   0.07 \
    --cmr_sample_k  4096 \
    \
    --batch_size    64 \
    --epochs        100 \
    --warmup_epochs 10 \
    \
    --blr           1e-5 \
    --min_lr        1e-7 \
    --weight_decay  0.05 \
    --layer_decay   0.75 \
    --proj_lr_scale 10.0 \
    --cmr_lr_scale  100.0 \
    --clip_grad     1.0 \
    \
    --patience      12 \
    --eval_freq     5 \
    --save_freq     10 \
    \
    --output_dir    "${OUTPUT_DIR}" \
    --log_dir       "${LOG_DIR}" \
    --num_workers   8 \
    --gpu           "${GPU_IDS}" \
    --desc          "run_${TIMESTAMP}" \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "训练完成！"
echo "下游微调兼容 encoder 已保存至："
echo "  ${OUTPUT_DIR}/contrast_pretrain_encoder_best.pth"
echo ""
echo "下游微调示例命令："
echo "  python main_finetune.py \\"
echo "    --finetune ${OUTPUT_DIR}/contrast_pretrain_encoder_best.pth \\"
echo "    --data_path /path/to/downstream_data \\"
echo "    [其他参数...]"
