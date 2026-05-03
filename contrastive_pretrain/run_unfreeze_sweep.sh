#!/bin/bash
# run_unfreeze_sweep.sh — fully unfrozen RETFound backbone, sweep LR
#
# freeze_blocks=0: all 24 ViT-L blocks trainable (~307M params)
# GPU 0: low  LR  — fundus_lr=1e-5,  head_lr=1e-4  (conservative baseline)
# GPU 1: mid  LR  — fundus_lr=1e-5,  head_lr=1e-3  (high head only)
# GPU 2: high LR  — fundus_lr=3e-5,  head_lr=1e-3  (moderate backbone)
# GPU 3: high LR  — fundus_lr=1e-4,  head_lr=1e-3  (aggressive)

set -e
cd "$(dirname "$0")/.."

CMR_CKPT=${CMR_CKPT:-contrastive_pretrain/checkpoints_cmr_v3/best.pth}
CMR_NPY_DIR=${CMR_NPY_DIR:-/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3}
PYTHON=${PYTHON:-/data/home/shujia/miniconda3/envs/modeltrain/bin/python}
RETFOUND_CKPT=${RETFOUND_CKPT:-RETFound_cfp_weights.pth}
SCRIPT=contrastive_pretrain/train_fundus_cmr.py

COMMON="--cmr_ckpt $CMR_CKPT --cmr_npy_dir $CMR_NPY_DIR --cmr_version v3 \
  --retfound_ckpt $RETFOUND_CKPT \
  --lam1 1.0 --lam2 0.1 --lam3 1.0 \
  --freeze_blocks 0 \
  --epochs 50 --batch_size 16 --warmup_epochs 5 --patience 10 \
  --weight_decay 0.05"

launch() {
  local GPU=$1 FLRB=$2 HLR=$3 TAG=$4
  local OUT=contrastive_pretrain/checkpoints_unfreeze_${TAG}
  mkdir -p $OUT
  echo "  GPU $GPU: freeze=0  fundus_lr=$FLRB  head_lr=$HLR  → $OUT"
  CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON $SCRIPT $COMMON \
    --fundus_lr $FLRB --head_lr $HLR \
    --out_dir $OUT \
    > $OUT/train.log 2>&1 &
  echo "    PID=$!"
}

echo "=== Fully-Unfrozen Sweep (freeze_blocks=0) ==="
launch 0  1e-5  1e-4  "flr1e5_hlr1e4"
launch 1  1e-5  1e-3  "flr1e5_hlr1e3"
launch 2  3e-5  1e-3  "flr3e5_hlr1e3"
launch 3  1e-4  1e-3  "flr1e4_hlr1e3"

echo ""
echo "Monitor: tail -f contrastive_pretrain/checkpoints_unfreeze_*/train.log"
