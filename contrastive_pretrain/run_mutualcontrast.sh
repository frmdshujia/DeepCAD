#!/bin/bash
# run_mutualcontrast.sh
#
# Mutual Contrastive + Downstream Task joint training
#
# Quick start (4-GPU full run):
#   bash contrastive_pretrain/run_mutualcontrast.sh
#
# Smoke test (CPU/single-GPU, 2 epochs, 8 samples):
#   SMOKE=1 bash contrastive_pretrain/run_mutualcontrast.sh
#
# Tune lambda (weaker contrastive):
#   LAM=0.1 bash contrastive_pretrain/run_mutualcontrast.sh

set -e
cd /data/home/shujia/CHD/model_train/RETFound_MAE-main

PYTHON=/data/home/shujia/miniconda3/envs/modeltrain/bin/python
GPUS=${GPUS:-4}
EPOCHS=${EPOCHS:-30}
BATCH=${BATCH:-8}
LR=${LR:-1e-4}
LAM=${LAM:-0.3}
GAMMA=${GAMMA:-1.0}
TEMP=${TEMP:-0.07}
PATIENCE=${PATIENCE:-5}
SMOKE=${SMOKE:-0}
OUT_DIR=contrastive_pretrain/checkpoints_mutualcontrast

GRAD_CKPT="--grad_ckpt"   # keeps ViT-L peak mem ~3GB vs ~10GB

mkdir -p $OUT_DIR
LOG=$OUT_DIR/train.log

echo "============================================================"
echo "Mutual Contrastive + Downstream Task Training"
echo "  GPUs=$GPUS  epochs=$EPOCHS  batch/GPU=$BATCH  lr=$LR"
echo "  gamma=$GAMMA  lam=$LAM  temperature=$TEMP"
echo "  grad_ckpt=ON  patience=$PATIENCE"
echo "  log -> $LOG"
echo "============================================================"

if [ "$SMOKE" = "1" ]; then
    echo ">>> SMOKE TEST MODE <<<"
    $PYTHON contrastive_pretrain/train_mutualcontrast.py \
        --epochs 2 \
        --batch_size 8 \
        --lr $LR \
        --gamma $GAMMA \
        --lam $LAM \
        --temperature $TEMP \
        --patience 99 \
        --out_dir $OUT_DIR \
        $GRAD_CKPT \
        --smoke_test \
        --no_dist 2>&1 | tee $LOG
else
    torchrun \
        --nproc_per_node=$GPUS \
        --master_port=29501 \
        contrastive_pretrain/train_mutualcontrast.py \
        --epochs $EPOCHS \
        --batch_size $BATCH \
        --lr $LR \
        --gamma $GAMMA \
        --lam $LAM \
        --temperature $TEMP \
        --patience $PATIENCE \
        --out_dir $OUT_DIR \
        $GRAD_CKPT 2>&1 | tee $LOG
fi
