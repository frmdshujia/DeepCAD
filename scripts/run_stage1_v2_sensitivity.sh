#!/usr/bin/env bash
set -euo pipefail

: "${MANIFEST_DIR:?Set MANIFEST_DIR to final_manifests_v2_20260916}"
: "${CMR_TEACHER_CHECKPOINT:?Set CMR_TEACHER_CHECKPOINT to selected hierarchical teacher}"
: "${RETFOUND_CHECKPOINT:?Set RETFOUND_CHECKPOINT to RETFound weights}"
: "${EXTERNAL_MANIFEST:?Set EXTERNAL_MANIFEST to the final 67,590-person list}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new versioned result directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"

"${PYTHON_BIN}" verify_manifests.py \
  --checksum-file "${MANIFEST_DIR}/SHA256SUMS.txt"

PARTICIPANT_MANIFESTS=(
  "${MANIFEST_DIR}/stage1_train_participants_16frame.csv"
  "${MANIFEST_DIR}/stage1_val_participants_16frame.csv"
  "${MANIFEST_DIR}/stage1_test_participants_16frame.csv"
)
IMAGE_MANIFESTS=(
  "${MANIFEST_DIR}/stage1_train_image_rows_16frame.csv"
  "${MANIFEST_DIR}/stage1_val_image_rows_16frame.csv"
  "${MANIFEST_DIR}/stage1_test_image_rows_16frame.csv"
)

EMBEDDING_DIR="${OUTPUT_ROOT}/teacher_embeddings_hierarchical_seed${SEED}"
"${PYTHON_BIN}" extract_cmr_embeddings.py \
  --manifest "${PARTICIPANT_MANIFESTS[@]}" \
  --checkpoint "${CMR_TEACHER_CHECKPOINT}" \
  --output-dir "${EMBEDDING_DIR}"

for POLICY in all require_t1; do
  EXTRA_ARGS=()
  if [[ "${POLICY}" == "require_t1" ]]; then
    EXTRA_ARGS+=(--require-t1)
  fi
  "${PYTHON_BIN}" train_stage1_contrastive.py \
    --config configs/stage1_alignment.yaml \
    --manifest "${IMAGE_MANIFESTS[@]}" \
    --cmr-embeddings "${EMBEDDING_DIR}/cmr_teacher_embeddings.npy" \
    --cmr-eids "${EMBEDDING_DIR}/cmr_teacher_eids.npy" \
    --retfound-checkpoint "${RETFOUND_CHECKPOINT}" \
    --external-validation-manifest "${EXTERNAL_MANIFEST}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_ROOT}/stage1_${POLICY}_seed${SEED}" \
    "${EXTRA_ARGS[@]}"
done
