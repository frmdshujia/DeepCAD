#!/usr/bin/env bash
# 1) 三份 backbone 冻结前向，导出 embedding（可三卡并行各跑一个）
# 2) 对 15 个任务 × 3 个 init 训练线性头（Mode1，仅 logits）
set -euo pipefail
REPO="/data/home/shujia/CHD/model_train/RETFound_MAE-main"
PY="${PYTHON:-/data/home/shujia/miniconda3/envs/retfound/bin/python}"
cd "$REPO"
mkdir -p output_dir/stage1_emb_cache

CSV="contrastive_pretrain/preprocessed_data/stage1_fundus_downstream_finetuning_with_image_paths.csv"
FUNDUS="/data/home/home6/fundus_data/UKB/fundus_images"
DELIV="output_dir/stage1_mode1_emb_delivery_final.csv"

echo "=== [1/2] 导出 embedding（retfound / controlled / no_residual，三卡并行）==="
# 三卡并行；若只有一卡，改为顺序执行（去掉 & 与 wait）
$PY -u contrastive_pretrain/stage1_finetune/extract_stage1_embeddings.py \
  --init_source retfound --gpu 0 --batch_size 128 --fp16_storage \
  --stage1_csv "$CSV" --fundus_root "$FUNDUS" \
  --output_pt output_dir/stage1_emb_cache/emb_retfound.pt &
PID0=$!
$PY -u contrastive_pretrain/stage1_finetune/extract_stage1_embeddings.py \
  --init_source controlled --gpu 1 --batch_size 128 --fp16_storage \
  --stage1_csv "$CSV" --fundus_root "$FUNDUS" \
  --output_pt output_dir/stage1_emb_cache/emb_controlled.pt &
PID1=$!
$PY -u contrastive_pretrain/stage1_finetune/extract_stage1_embeddings.py \
  --init_source no_residual --gpu 2 --batch_size 128 --fp16_storage \
  --stage1_csv "$CSV" --fundus_root "$FUNDUS" \
  --output_pt output_dir/stage1_emb_cache/emb_no_residual.pt &
PID2=$!
wait $PID0 $PID1 $PID2

echo "=== [2/2] Mode1 线性头（45 作业，四卡并行调度）==="
$PY -u contrastive_pretrain/stage1_finetune/run_mode1_emb_train_parallel.py \
  --gpus 0,1,2,3 \
  --emb_cache_dir output_dir/stage1_emb_cache \
  --out_matrix_dir output_dir/stage1_mode1_emb_matrix \
  --delivery_final_csv "$DELIV" \
  --epochs 50 --batch_size 16384 \
  --skip_done

echo "=== 全部完成。汇总见 $DELIV 与各目录 metrics_mode1_emb.json ==="
