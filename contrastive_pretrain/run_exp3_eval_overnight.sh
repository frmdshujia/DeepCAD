#!/usr/bin/env bash
# Exp3（分层采样）训练完成后：在 test 上跑与实验一/二一致的评估链。
# 1) eval_cross_modal_ablation：gt_pred Spearman/Pearson、全库与小候选集 R@k（100/1000）
# 2) linear_probe_clinical：LVEF 回归 + CHD 分类（test 报告）
# 3) linear_probe_pc：14 维 PC 线性探针（proj 表征；指标为 val，与脚本一致）
#
# 用法（仓库根）：
#   nohup bash contrastive_pretrain/run_exp3_eval_overnight.sh >> output_dir/exp3_stratified_20260417_165326/overnight_eval.log 2>&1 &
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

if command -v conda >/dev/null 2>&1; then
  PYTHON=(conda run -n retfound --no-capture-output python)
else
  PYTHON=(python)
fi

CKPT_DIR="${REPO_ROOT}/output_dir/exp3_stratified_20260417_165326"
CKPT="${CKPT_DIR}/checkpoint_best.pth"
RETF="${REPO_ROOT}/RETFound_cfp_weights.pth"
FUNDUS="${REPO_ROOT}/contrastive_pretrain/preprocessed_data/fundus_table.csv"
CMR="${REPO_ROOT}/contrastive_pretrain/preprocessed_data/cmr_table.csv"
STAGE2="${REPO_ROOT}/contrastive_pretrain/preprocessed_data/stage2_cmr.csv"

GPU="${EVAL_GPU:-0}"
export CUDA_VISIBLE_DEVICES="${GPU}"

echo "=============================================="
echo "  Exp3 过夜评估 | $(date -Iseconds)"
echo "  CKPT: ${CKPT}"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "=============================================="

if [[ ! -f "${CKPT}" ]]; then
  echo "ERROR: missing ${CKPT}" >&2
  exit 1
fi

echo ""
echo ">>> [1/3] eval_cross_modal_ablation.py (test)"
"${PYTHON[@]}" contrastive_pretrain/eval_cross_modal_ablation.py \
  --retfound_ckpt "${RETF}" \
  --contrastive_ckpt "${CKPT}" \
  --split test \
  --fundus_csv "${FUNDUS}" \
  --cmr_csv "${CMR}" \
  --gpu 0 \
  --batch_size 32 \
  --num_workers 4 \
  --n_repeat_subsample 5 \
  --pool_sizes "100,1000" \
  --output_json "${CKPT_DIR}/eval_cross_modal_ablation_test.json"

echo ""
echo ">>> [2/3] linear_probe_clinical.py (test 指标)"
"${PYTHON[@]}" contrastive_pretrain/linear_probe_clinical.py \
  --retfound_ckpt "${RETF}" \
  --contrastive_ckpt "${CKPT}" \
  --stage2_csv "${STAGE2}" \
  --fundus_csv "${FUNDUS}" \
  --mlp_hidden_layers 1 \
  --device cuda:0 \
  --batch_size 64 \
  --num_workers 4 \
  --output_json "${CKPT_DIR}/linear_probe_clinical_test_mlp1h.json"

echo ""
echo ">>> [3/3] linear_probe_pc.py（14 PC，proj 表征）"
"${PYTHON[@]}" contrastive_pretrain/linear_probe_pc.py \
  --fundus_csv "${FUNDUS}" \
  --retfound_ckpt "${RETF}" \
  --representation proj \
  --contrast_fundus_full_ckpt "${CKPT}" \
  --proj_dim 256 \
  --drop_path 0.1 \
  --epochs 80 \
  --device cuda:0 \
  --batch_size 64 \
  --num_workers 4 \
  --output_json "${CKPT_DIR}/linear_probe_pc14_proj.json"

echo ""
echo "=============================================="
echo "  全部完成 | $(date -Iseconds)"
echo "  结果 JSON 见: ${CKPT_DIR}/"
echo "=============================================="
