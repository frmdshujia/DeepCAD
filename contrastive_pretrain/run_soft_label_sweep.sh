#!/usr/bin/env bash
# Soft 标签锐化扫描：sgt_temp × σ（基于 fast_contrast_train）
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
OUT_ROOT="${ROOT}/output_dir/soft_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_ROOT"
SUMMARY="${OUT_ROOT}/SWEEP_SUMMARY.md"
BASE_SIG=6.5893
EPOCHS=120
EVAL_FREQ=120

{
  echo "# Soft label 锐化扫描"
  echo ""
  echo "| run | sgt_temp | sigma | loss | tgt_H | gt_Spearman | gt_Pearson | R@5 | paired_cos |"
  echo "|-----|----------|-------|------|-------|-------------|------------|-----|------------|"
} > "$SUMMARY"

run_one () {
  local NAME="$1"
  local ST="$2"
  local SIG="$3"
  local DIR="${OUT_ROOT}/${NAME}"
  mkdir -p "$DIR"
  conda run -n retfound --no-capture-output python contrastive_pretrain/fast_contrast_train.py \
    --train_feat output_dir/feature_cache/train_feats_full.pt \
    --val_feat output_dir/feature_cache/val_feats_full.pt \
    --cmr_csv contrastive_pretrain/preprocessed_data/cmr_table.csv \
    --output_dir "$DIR" \
    --loss_type soft \
    --sigma "$SIG" \
    --sgt_temp "$ST" \
    --temperature 0.15 \
    --cmr_sample_k 256 \
    --batch_size 128 \
    --epochs "$EPOCHS" \
    --eval_freq "$EVAL_FREQ" \
    --lr 1e-3 \
    --linear_proj \
    --gpu 0 \
    2>&1 | tee "${DIR}/train.log"

  python3 - "$NAME" "$ST" "$SIG" "$DIR" >> "$SUMMARY" << 'PY'
import json, sys
name, st, sig, d = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    with open(d + "/log.json") as f:
        rows = json.load(f)
    last = rows[-1] if rows else {}
except Exception:
    print(f"| {name} | {st} | {sig} | err | | | | | |")
    sys.exit(0)
vl = last.get("loss", "")
th = last.get("train_target_entropy", "")
gp = last.get("gt_pred_pearson", "")
gs = last.get("gt_pred_spearman", "")
r5 = last.get("R@5", "")
pc = last.get("paired_cosine", "")
print(f"| {name} | {st} | {sig} | {vl} | {th} | {gs} | {gp} | {r5} | {pc} |")
PY
}

run_one "s1_tau1p0_sig1p0" 1.0 "${BASE_SIG}"
run_one "s2_tau0p5_sig1p0" 0.5 "${BASE_SIG}"
run_one "s3_tau0p3_sig1p0" 0.3 "${BASE_SIG}"
run_one "s4_tau0p2_sig1p0" 0.2 "${BASE_SIG}"
SIG05=$(python3 -c "print(6.5893*0.5)")
SIG20=$(python3 -c "print(6.5893*2)")
run_one "s5_tau0p3_sig0p5" 0.3 "${SIG05}"
run_one "s6_tau0p3_sig2p0" 0.3 "${SIG20}"

{
  echo ""
  echo "说明：K=256 时均匀参考熵 log(K)≈5.55；**tgt_H** 为训练 batch 上 softmax(S_GT/τ) 的平均熵，越低目标越尖。"
  echo "输出目录: $OUT_ROOT"
} >> "$SUMMARY"
echo "Wrote $SUMMARY"
