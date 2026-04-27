#!/bin/bash
# run_4disease_cmr.sh — 4-GPU 四病分类实验（8,224 CMR样本 MedSAM微调）
# GPU0: I21 (心肌梗死)
# GPU1: isch_hd (缺血性心脏病复合终点)
# GPU2: I25 (慢性缺血性心脏病)
# GPU3: I20 (心绞痛)

PY=/data/home/shujia/miniconda3/envs/retfound/bin/python
SCRIPT=contrastive_pretrain/ceiling_probe.py
OUT=output_dir/4disease_cmr

mkdir -p $OUT/logs

run_job() {
    local gpu=$1; local tag=$2; shift 2
    local logfile=$OUT/logs/${tag}.log
    echo "[launch] GPU$gpu  $tag  → $logfile"
    CUDA_VISIBLE_DEVICES=$gpu $PY $SCRIPT "$@" \
        --out_dir $OUT/$tag \
        > $logfile 2>&1
    local ec=$?
    echo "[done]  GPU$gpu  $tag  exit=$ec"
    return $ec
}

COMMON="--encoder medsam --modality cmr_all --mode finetune
        --epochs 60 --batch_size 16 --backbone_lr 5e-6 --head_lr 1e-3
        --patience 20 --eval_every 5"

(run_job 0 i21     $COMMON --task i21    ) &
(run_job 1 isch_hd $COMMON --task isch_hd) &
(run_job 2 i25     $COMMON --task i25    ) &
(run_job 3 i20     $COMMON --task i20    ) &

wait
echo "=============================="
echo "4-disease experiments done!"
echo "=============================="
