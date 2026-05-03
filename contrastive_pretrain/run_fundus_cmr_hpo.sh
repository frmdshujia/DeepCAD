#!/bin/bash
# run_fundus_cmr_hpo.sh  —  Branch 1: Fundus-CMR 3-loss grid search on 2582
#
# Grid: fix lam1=1.0, sweep lam2 × lam3 ∈ {0.1, 1.0, 5.0}² = 9 combos
# 4 GPUs → Batch 1 (4 jobs), then Batch 2 (4 jobs), then Batch 3 (1 job)
#
# Rationale: lam1=1.0 sets the scale; lam2/lam3 vary 50× range so we can
# observe whether RKD structure or task supervision drives fundus AUC higher.
# Second round can vary lam1 if needed.
#
# Usage:
#   # Batch 1 (run immediately):
#   BATCH=1 bash contrastive_pretrain/run_fundus_cmr_hpo.sh
#   # Batch 2 (run when Batch 1 GPUs free):
#   BATCH=2 bash contrastive_pretrain/run_fundus_cmr_hpo.sh
#   # Batch 3:
#   BATCH=3 bash contrastive_pretrain/run_fundus_cmr_hpo.sh

set -e
cd "$(dirname "$0")/.."

CMR_CKPT=${CMR_CKPT:-contrastive_pretrain/checkpoints_cmr_v3/best.pth}
CMR_NPY_DIR=${CMR_NPY_DIR:-/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3}
PYTHON=${PYTHON:-/data/home/shujia/miniconda3/envs/modeltrain/bin/python}
SCRIPT=contrastive_pretrain/train_fundus_cmr.py
# RETFOUND_CKPT: default to local path on 2582; override if needed
RETFOUND_CKPT=${RETFOUND_CKPT:-RETFound_cfp_weights.pth}
RETFOUND_ARG="--retfound_ckpt $RETFOUND_CKPT"
COMMON="--cmr_ckpt $CMR_CKPT --cmr_npy_dir $CMR_NPY_DIR --cmr_version v3 \
  --freeze_blocks 20 --epochs 50 --batch_size 16 \
  --fundus_lr 1e-5 --head_lr 1e-4 --warmup_epochs 2 --patience 10 \
  $RETFOUND_ARG"

BATCH=${BATCH:-1}
echo "=== Fundus-CMR HPO — Batch $BATCH  (CMR ckpt: $CMR_CKPT) ==="

launch() {
  local GPU=$1 LAM1=$2 LAM2=$3 LAM3=$4
  local TAG="lam1_${LAM1}_lam2_${LAM2}_lam3_${LAM3}"
  local OUT=contrastive_pretrain/checkpoints_fundus_cmr_${TAG}
  mkdir -p $OUT
  echo "  GPU $GPU: lam1=$LAM1  lam2=$LAM2  lam3=$LAM3  → $OUT"
  CUDA_VISIBLE_DEVICES=$GPU nohup $PYTHON $SCRIPT $COMMON \
    --out_dir $OUT \
    --lam1 $LAM1 --lam2 $LAM2 --lam3 $LAM3 \
    > $OUT/train.log 2>&1 &
  echo "    PID=$!"
}

if [ "$BATCH" = "1" ]; then
  # lam2\lam3  0.1  1.0  5.0
  #      0.1   [1]  [2]  [3]
  #      1.0   [4]  ...
  launch 0  1.0  0.1  0.1
  launch 1  1.0  0.1  1.0
  launch 2  1.0  0.1  5.0
  launch 3  1.0  1.0  0.1
fi

if [ "$BATCH" = "2" ]; then
  launch 0  1.0  1.0  1.0
  launch 1  1.0  1.0  5.0
  launch 2  1.0  5.0  0.1
  launch 3  1.0  5.0  1.0
fi

if [ "$BATCH" = "3" ]; then
  launch 0  1.0  5.0  5.0
fi

echo ""
echo "Monitor logs in contrastive_pretrain/checkpoints_fundus_cmr_*/train.log"
echo "Compare results with:"
echo "  python contrastive_pretrain/compare_hpo_results.py --pattern 'checkpoints_fundus_cmr_*'"
