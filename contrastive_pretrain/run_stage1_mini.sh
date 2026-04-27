#!/bin/bash
# run_stage1_mini.sh
#
# Stage 1 快速消融实验启动脚本（modeltrain conda env）
#
# 内存说明：
#   - Exp A (CMR 单塔):  单卡 ~2GB  → 可单卡或多卡运行
#   - Exp B (眼底单塔):  单卡 ~4.5GB (冻结前20层) → 需 ≥5GB 空闲
#   - Exp D (双塔):      单卡 ~4.5GB (两编码器均冻结主干) → 需 ≥5GB 空闲
#
# 用法：
#   EXP=A bash contrastive_pretrain/run_stage1_mini.sh   # CMR 单塔
#   EXP=B bash contrastive_pretrain/run_stage1_mini.sh   # 眼底单塔
#   EXP=D bash contrastive_pretrain/run_stage1_mini.sh   # 双塔
#
# 全参数微调（需要更多显存，4卡 torchrun）:
#   EXP=B FREEZE_FUNDUS=0 FREEZE_CMR=0 GPUS=4 bash contrastive_pretrain/run_stage1_mini.sh

set -e
cd /data/home/shujia/CHD/model_train/RETFound_MAE-main

PYTHON=/data/home/shujia/miniconda3/envs/modeltrain/bin/python
EXP=${EXP:-A}
EPOCHS=${EPOCHS:-15}
BATCH=${BATCH:-4}
LR=${LR:-1e-4}
PATIENCE=${PATIENCE:-5}
MAX_SAMPLES=${MAX_SAMPLES:-0}   # 0=all available npy/png
GPUS=${GPUS:-1}
OUT_DIR=contrastive_pretrain/checkpoints_stage1_mini

# 全参微调：freeze_fundus_blocks=0, freeze_cmr=0
# 4卡×24GB，每卡 batch=4，有效 batch=16
# grad_ckpt 启用梯度检查点，将 ViT-L 激活显存从 ~10GB 降到 ~6GB
FREEZE_FUNDUS=${FREEZE_FUNDUS:-0}
FREEZE_CMR_FLAG=""
if [ "${FREEZE_CMR:-0}" = "1" ] && [ "$EXP" = "D" ]; then
    FREEZE_CMR_FLAG="--freeze_cmr_backbone"
fi
GRAD_CKPT_FLAG="--grad_ckpt"

echo "====================================================="
echo "Exp=$EXP  epochs=$EPOCHS  batch/GPU=$BATCH  gpus=$GPUS"
echo "全参微调: freeze_fundus_blocks=$FREEZE_FUNDUS  grad_ckpt=ON"
echo "====================================================="

torchrun \
    --nproc_per_node=$GPUS \
    --master_port=29500 \
    contrastive_pretrain/train_stage1_mini.py \
    --exp $EXP \
    --epochs $EPOCHS \
    --batch_size $BATCH \
    --lr $LR \
    --patience $PATIENCE \
    --max_samples $MAX_SAMPLES \
    --freeze_fundus_blocks $FREEZE_FUNDUS \
    $FREEZE_CMR_FLAG \
    $GRAD_CKPT_FLAG \
    --out_dir $OUT_DIR
