#!/usr/bin/env bash
set -euo pipefail

: "${MANIFEST_DIR:?Set MANIFEST_DIR to final_manifests_v2_20260916}"
: "${BACKBONE_CHECKPOINT:?Set BACKBONE_CHECKPOINT to MedSAM2-US-Heart weights}"
: "${OUTPUT_ROOT:?Set OUTPUT_ROOT to a new versioned result directory}"

PYTHON_BIN="${PYTHON_BIN:-python}"
SEED="${SEED:-42}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-32}"
UNFREEZE_BATCH_SIZE="${UNFREEZE_BATCH_SIZE:-16}"

"${PYTHON_BIN}" verify_manifests.py \
  --checksum-file "${MANIFEST_DIR}/SHA256SUMS.txt"

MANIFESTS=(
  "${MANIFEST_DIR}/cmr_teacher_train_complete27_nonexternal.csv"
  "${MANIFEST_DIR}/cmr_teacher_val_complete27_nonexternal.csv"
  "${MANIFEST_DIR}/cmr_teacher_test_complete27_nonexternal.csv"
)
CLASSIFICATION="prevalent_I21,prevalent_I25,history_revasc,composite_cardiomyopathy_hf,prevalent_I48"
REGRESSION_22="LVEF,LVEDV,LVESV,LVSV,LV_CO,LVM,LV_WT,GLS,GCS,GRS,RVEF,RVEDV,RVESV,RVSV,LAEF,LA_max,LA_min,LASV,RAEF,RA_max,RA_min,RASV"
REGRESSION_8="GLS,GCS,GRS,LVEF,LV_CO,RVEF,LVSV,RVSV"

for FUSION_MODE in global_only sax_t1 hierarchical; do
  "${PYTHON_BIN}" train_cmr_teacher.py \
    --config configs/cmr_teacher.yaml \
    --manifest "${MANIFESTS[@]}" \
    --backbone-checkpoint "${BACKBONE_CHECKPOINT}" \
    --fusion-mode "${FUSION_MODE}" \
    --classification-columns "${CLASSIFICATION}" \
    --regression-columns "${REGRESSION_22}" \
    --classification-weight 0.5 \
    --regression-weight 0.5 \
    --epochs "${EPOCHS}" \
    --batch-size "${BATCH_SIZE}" \
    --unfreeze-batch-size "${UNFREEZE_BATCH_SIZE}" \
    --seed "${SEED}" \
    --output-dir "${OUTPUT_ROOT}/cmr_27task_${FUSION_MODE}_seed${SEED}"
done

"${PYTHON_BIN}" train_cmr_teacher.py \
  --config configs/cmr_teacher.yaml \
  --manifest "${MANIFESTS[@]}" \
  --backbone-checkpoint "${BACKBONE_CHECKPOINT}" \
  --fusion-mode hierarchical \
  --classification-columns "" \
  --regression-columns "${REGRESSION_8}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --unfreeze-batch-size "${UNFREEZE_BATCH_SIZE}" \
  --seed "${SEED}" \
  --output-dir "${OUTPUT_ROOT}/cmr_legacy8_hierarchical_seed${SEED}"
