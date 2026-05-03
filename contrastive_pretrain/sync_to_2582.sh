#!/bin/bash
# sync_to_2582.sh  —  Sync code + CMR npy files from 2572 → 2582
#
# Edit TARGET_HOST before running.
# Run from the root of the repo on 2572.

TARGET_HOST=${1:-"FILL_IN_2582_IP_OR_HOSTNAME"}
TARGET_USER=${2:-"shujia"}
REMOTE="${TARGET_USER}@${TARGET_HOST}"

REPO_SRC=/data/home/shujia/CHD/model_train/RETFound_MAE-main
NPY_SRC=/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3
REPO_DST=/data/home/shujia/CHD/model_train/RETFound_MAE-main
NPY_DST=/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3

echo "=== Syncing code to ${REMOTE}:${REPO_DST} ==="
rsync -avz --progress \
  --include='*.py' \
  --include='*.sh' \
  --include='*.json' \
  --include='*.csv' \
  --include='*.pth' \
  --exclude='__pycache__' \
  --exclude='*.npy' \
  --exclude='checkpoints_*/' \
  --exclude='.git/' \
  "${REPO_SRC}/" "${REMOTE}:${REPO_DST}/"

echo ""
echo "=== Syncing CMR npy files to ${REMOTE}:${NPY_DST} ==="
echo "(~32K files × ~0.7MB = ~22GB, will take a while)"
rsync -avz --progress \
  "${NPY_SRC}/" "${REMOTE}:${NPY_DST}/"

echo ""
echo "=== Syncing best CMR checkpoint ==="
rsync -avz --progress \
  "${REPO_SRC}/contrastive_pretrain/checkpoints_cmr_v3/best.pth" \
  "${REMOTE}:${REPO_DST}/contrastive_pretrain/checkpoints_cmr_v3/best.pth"

echo "Sync complete."
