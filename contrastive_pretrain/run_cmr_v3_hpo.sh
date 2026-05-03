#!/bin/bash
# run_cmr_v3_hpo.sh  —  Branch 2: CMR v3 HPO on 2572
#
# GPU 0: original baseline (train_cmr_v3.py, composite+I21) already running
# GPU 1: B1 — composite-only, baseline LR, unfreeze_bs=8
# GPU 2: B2 — composite-only, 3× LR,       unfreeze_bs=10  (larger batch)
# GPU 3: B3 — composite-only, depth=4,      unfreeze_bs=6   (deeper transformer)
#
# Usage: bash contrastive_pretrain/run_cmr_v3_hpo.sh

set -e
cd "$(dirname "$0")/.."

DATA_DIR=/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3
PYTHON=/data/home/shujia/miniconda3/envs/modeltrain/bin/python
SCRIPT=contrastive_pretrain/train_cmr_v3_hpo.py
COMMON="--data_dir $DATA_DIR --freeze_epochs 5 --warmup_epochs 3 --patience 8 --epochs 40"

echo "=== CMR v3 HPO sweep (GPUs 1,2,3) ==="
echo "GPU 0: original baseline already running — untouched"

# B1: composite-only baseline, unfreeze_bs=8
mkdir -p contrastive_pretrain/checkpoints_cmr_hpo_B1_bs8
echo "[B1] GPU 1: enc_lr=1e-5  bs=24/8  depth=2"
CUDA_VISIBLE_DEVICES=1 nohup $PYTHON $SCRIPT $COMMON \
  --out_dir              contrastive_pretrain/checkpoints_cmr_hpo_B1_bs8 \
  --batch_size           24 \
  --unfreeze_batch_size  8 \
  --enc_lr               1e-5 \
  --head_lr              1e-4 \
  --transformer_depth    2 \
  > contrastive_pretrain/checkpoints_cmr_hpo_B1_bs8/train.log 2>&1 &
echo "  PID=$!"

# B2: higher LR + larger batch
mkdir -p contrastive_pretrain/checkpoints_cmr_hpo_B2_lrhigh
echo "[B2] GPU 2: enc_lr=3e-5  bs=32/10  depth=2"
CUDA_VISIBLE_DEVICES=2 nohup $PYTHON $SCRIPT $COMMON \
  --out_dir              contrastive_pretrain/checkpoints_cmr_hpo_B2_lrhigh \
  --batch_size           32 \
  --unfreeze_batch_size  10 \
  --enc_lr               3e-5 \
  --head_lr              3e-4 \
  --transformer_depth    2 \
  > contrastive_pretrain/checkpoints_cmr_hpo_B2_lrhigh/train.log 2>&1 &
echo "  PID=$!"

# B3: deeper transformer
mkdir -p contrastive_pretrain/checkpoints_cmr_hpo_B3_depth4
echo "[B3] GPU 3: enc_lr=1e-5  bs=24/6  depth=4"
CUDA_VISIBLE_DEVICES=3 nohup $PYTHON $SCRIPT $COMMON \
  --out_dir              contrastive_pretrain/checkpoints_cmr_hpo_B3_depth4 \
  --batch_size           24 \
  --unfreeze_batch_size  6 \
  --enc_lr               1e-5 \
  --head_lr              1e-4 \
  --transformer_depth    4 \
  > contrastive_pretrain/checkpoints_cmr_hpo_B3_depth4/train.log 2>&1 &
echo "  PID=$!"

echo ""
echo "Monitor:"
echo "  tail -f contrastive_pretrain/checkpoints_cmr_hpo_B1_bs8/train.log"
echo "  tail -f contrastive_pretrain/checkpoints_cmr_hpo_B2_lrhigh/train.log"
echo "  tail -f contrastive_pretrain/checkpoints_cmr_hpo_B3_depth4/train.log"
