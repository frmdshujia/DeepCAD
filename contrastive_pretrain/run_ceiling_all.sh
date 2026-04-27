#!/bin/bash
# run_ceiling_all.sh — 4-GPU overnight ceiling experiments
# GPU0: Exp1 linear CMR (all tasks) → Exp5 LAX i21/isch_hd
# GPU1: Exp2 finetune CMR i21 → isch_hd
# GPU2: Exp2 finetune CMR lvef → lvedv → Exp5 SAX i21/isch_hd
# GPU3: Exp3 linear fundus → Exp4 finetune fundus i21 → isch_hd → lvef → lvedv

PY=/data/home/shujia/miniconda3/envs/retfound/bin/python
SCRIPT=contrastive_pretrain/ceiling_probe.py
OUT=output_dir/ceiling_v2

mkdir -p $OUT/logs

run_job() {
    local gpu=$1; local tag=$2; shift 2
    local logfile=$OUT/logs/${tag}.log
    echo "[launch] GPU$gpu  $tag  → $logfile"
    CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT "$@" \
        > $logfile 2>&1
    echo "[done]  GPU$gpu  $tag  exit=$?"
}

# ─────────────────────────────────────────────
# GPU 0: Exp1 linear CMR (4 tasks) → Exp5 LAX sequence
# ─────────────────────────────────────────────
(
  run_job 0 exp1_linear_cmr_i21     --encoder medsam --modality cmr_all --mode linear  --task i21     --out_dir $OUT/exp1_linear_cmr_i21
  run_job 0 exp1_linear_cmr_isch_hd --encoder medsam --modality cmr_all --mode linear  --task isch_hd --out_dir $OUT/exp1_linear_cmr_isch_hd
  run_job 0 exp1_linear_cmr_lvef    --encoder medsam --modality cmr_all --mode linear  --task lvef    --out_dir $OUT/exp1_linear_cmr_lvef
  run_job 0 exp1_linear_cmr_lvedv   --encoder medsam --modality cmr_all --mode linear  --task lvedv   --out_dir $OUT/exp1_linear_cmr_lvedv
  # Exp5: LAX-only sequence
  run_job 0 exp5_lax_i21     --encoder medsam --modality cmr_lax --mode finetune --task i21     --out_dir $OUT/exp5_lax_i21     --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
  run_job 0 exp5_lax_isch_hd --encoder medsam --modality cmr_lax --mode finetune --task isch_hd --out_dir $OUT/exp5_lax_isch_hd --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
) &

# ─────────────────────────────────────────────
# GPU 1: Exp2 finetune CMR i21 → isch_hd
# ─────────────────────────────────────────────
(
  run_job 1 exp2_finetune_cmr_i21     --encoder medsam --modality cmr_all --mode finetune --task i21     --out_dir $OUT/exp2_finetune_cmr_i21     --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
  run_job 1 exp2_finetune_cmr_isch_hd --encoder medsam --modality cmr_all --mode finetune --task isch_hd --out_dir $OUT/exp2_finetune_cmr_isch_hd --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
) &

# ─────────────────────────────────────────────
# GPU 2: Exp2 finetune CMR lvef → lvedv → Exp5 SAX sequence
# ─────────────────────────────────────────────
(
  run_job 2 exp2_finetune_cmr_lvef  --encoder medsam --modality cmr_all --mode finetune --task lvef  --out_dir $OUT/exp2_finetune_cmr_lvef  --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
  run_job 2 exp2_finetune_cmr_lvedv --encoder medsam --modality cmr_all --mode finetune --task lvedv --out_dir $OUT/exp2_finetune_cmr_lvedv --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
  # Exp5: SAX-only sequence
  run_job 2 exp5_sax_i21     --encoder medsam --modality cmr_sax --mode finetune --task i21     --out_dir $OUT/exp5_sax_i21     --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
  run_job 2 exp5_sax_isch_hd --encoder medsam --modality cmr_sax --mode finetune --task isch_hd --out_dir $OUT/exp5_sax_isch_hd --epochs 50 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
) &

# ─────────────────────────────────────────────
# GPU 3: Exp3 linear fundus → Exp4 finetune fundus (4 tasks)
# ─────────────────────────────────────────────
(
  run_job 3 exp3_linear_fundus_i21     --encoder retfound --modality fundus --mode linear  --task i21     --out_dir $OUT/exp3_linear_fundus_i21
  run_job 3 exp3_linear_fundus_isch_hd --encoder retfound --modality fundus --mode linear  --task isch_hd --out_dir $OUT/exp3_linear_fundus_isch_hd
  run_job 3 exp4_finetune_fundus_i21     --encoder retfound --modality fundus --mode finetune --task i21     --out_dir $OUT/exp4_finetune_fundus_i21     --epochs 30 --batch_size 16 --backbone_lr 1e-5 --head_lr 1e-3
  run_job 3 exp4_finetune_fundus_isch_hd --encoder retfound --modality fundus --mode finetune --task isch_hd --out_dir $OUT/exp4_finetune_fundus_isch_hd --epochs 30 --batch_size 16 --backbone_lr 1e-5 --head_lr 1e-3
  run_job 3 exp4_finetune_fundus_lvef    --encoder retfound --modality fundus --mode finetune --task lvef    --out_dir $OUT/exp4_finetune_fundus_lvef    --epochs 30 --batch_size 16 --backbone_lr 1e-5 --head_lr 1e-3
  run_job 3 exp4_finetune_fundus_lvedv   --encoder retfound --modality fundus --mode finetune --task lvedv   --out_dir $OUT/exp4_finetune_fundus_lvedv   --epochs 30 --batch_size 16 --backbone_lr 1e-5 --head_lr 1e-3
) &

wait
echo "=============================="
echo "All ceiling experiments done!"
echo "=============================="
