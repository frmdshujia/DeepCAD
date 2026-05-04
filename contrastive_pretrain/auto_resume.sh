#!/bin/bash
# auto_resume.sh — 重启后自动续训 CMR 实验
# 挂在 @reboot crontab，服务器重启后自动执行

WORKDIR="/data/home/shujia/CHD/model_train/RETFound_MAE-main"
CONDA_RUN="/data/home/shujia/miniconda3/bin/conda run -n modeltrain"
SCRIPT="contrastive_pretrain/train_cmr_v3.py"
DATA_DIR="/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3"
LOG_DIR="$WORKDIR/contrastive_pretrain"

# 实验配置：backbone → GPU
declare -A GPU_MAP
GPU_MAP["medsam2_heart"]=0
GPU_MAP["sam2_tiny"]=1
GPU_MAP["medsam2_latest"]=2

cd "$WORKDIR" || exit 1

# 等待 GPU 驱动就绪（最多等 120 秒）
echo "[auto_resume] $(date): waiting for GPU driver..."
for i in $(seq 1 24); do
    nvidia-smi > /dev/null 2>&1 && break
    sleep 5
done

if ! nvidia-smi > /dev/null 2>&1; then
    echo "[auto_resume] $(date): GPU driver not ready after 120s, aborting"
    exit 1
fi
echo "[auto_resume] $(date): GPU driver ready"

for backbone in "${!GPU_MAP[@]}"; do
    gpu="${GPU_MAP[$backbone]}"
    ckpt_dir="$LOG_DIR/checkpoints_cmr_${backbone}"
    log_file="$ckpt_dir/train.log"

    # 检查是否已完成（早停或正常结束）
    if grep -q "Training complete\|early stop" "$log_file" 2>/dev/null; then
        best=$(grep "Training complete" "$log_file" 2>/dev/null | tail -1 | grep -oP "[\d.]+$")
        echo "[auto_resume] $(date): $backbone already DONE (best_auc=$best), skipping"
        continue
    fi

    # 检查是否已在运行
    if ps aux | grep "train_cmr_v3" | grep "$backbone" | grep -v grep > /dev/null 2>&1; then
        echo "[auto_resume] $(date): $backbone already running, skipping"
        continue
    fi

    # 检查 checkpoint 是否存在
    if [ ! -f "$ckpt_dir/last.pth" ]; then
        echo "[auto_resume] $(date): $backbone no checkpoint found, skipping"
        continue
    fi

    # 启动续训
    echo "[auto_resume] $(date): starting $backbone on GPU$gpu (--resume)"
    nohup $CONDA_RUN bash -c \
        "CUDA_VISIBLE_DEVICES=${gpu} python -u ${WORKDIR}/${SCRIPT} \
        --data_dir ${DATA_DIR} \
        --backbone ${backbone} \
        --batch_size 16 \
        --unfreeze_batch_size 8 \
        --resume" \
    >> "$log_file" 2>&1 &
    echo "[auto_resume] $(date): $backbone PID=$!"
done

echo "[auto_resume] $(date): done"
