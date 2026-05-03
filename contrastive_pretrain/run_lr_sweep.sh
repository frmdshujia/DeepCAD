#!/bin/bash
# run_lr_sweep.sh — LR sweep with best lam (lam2=0.1, lam3=1.0)
#
# Hypothesis: AUC stuck at 0.57 because head_lr=1e-4 too small,
#             fundus_lr=1e-5 too conservative.
# Fix: head_lr → 1e-3, sweep fundus_lr and freeze_blocks.
#
# GPU 0: head_lr=1e-3, fundus_lr=1e-5,  freeze=20  (isolate head effect)
# GPU 1: head_lr=1e-3, fundus_lr=3e-5,  freeze=20  (moderate backbone)
# GPU 2: head_lr=1e-3, fundus_lr=1e-4,  freeze=20  (aggressive backbone)
# GPU 3: head_lr=1e-3, fundus_lr=3e-5,  freeze=16  (unfreeze more blocks)

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
  --epochs 50 --batch_size 16 --warmup_epochs 5 --patience 10 \
  --weight_decay 0.05"

launch() {
  local GPU=$1 FLBK=$2 FLRB=$3 HLR=$4 TAG=$5
  local OUT=contrastive_pretrain/checkpoints_lr_sweep_${TAG}
  mkdir -p $OUT
  echo "  GPU $GPU: freeze=$FLBK  fundus_lr=$FLRB  head_lr=$HLR  → $OUT"
  CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON $SCRIPT $COMMON \
    --freeze_blocks $FLBK --fundus_lr $FLRB --head_lr $HLR \
    --out_dir $OUT \
    > $OUT/train.log 2>&1 &
  echo "    PID=$!"
}

echo "=== LR Sweep (lam2=0.1 lam3=1.0, head_lr→1e-3) ==="
launch 0  20  1e-5  1e-3  "f20_flr1e5_hlr1e3"
launch 1  20  3e-5  1e-3  "f20_flr3e5_hlr1e3"
launch 2  20  1e-4  1e-3  "f20_flr1e4_hlr1e3"
launch 3  16  3e-5  1e-3  "f16_flr3e5_hlr1e3"

echo ""
echo "Monitor: tail -f contrastive_pretrain/checkpoints_lr_sweep_*/train.log"
