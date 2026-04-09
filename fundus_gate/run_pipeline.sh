#!/bin/bash
# 眼底门控模型完整训练流程
# 用法: bash run_pipeline.sh [--skip_cifar] [--skip_acdc] [--epochs 30] [--batch_size 64]

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── 默认参数 ──────────────────────────────────────────────────────────────────
EPOCHS=30
BATCH_SIZE=64
ARCH="efficientnet_b0"
LR=1e-4
SKIP_CIFAR=""
SKIP_ACDC=""
DEVICE="cuda"

# ── 解析参数 ──────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --skip_cifar)   SKIP_CIFAR="--skip_cifar"; shift;;
        --skip_acdc)    SKIP_ACDC="--skip_acdc";   shift;;
        --epochs)       EPOCHS="$2";               shift 2;;
        --batch_size)   BATCH_SIZE="$2";           shift 2;;
        --arch)         ARCH="$2";                 shift 2;;
        --lr)           LR="$2";                   shift 2;;
        --cpu)          DEVICE="cpu";              shift;;
        *) echo "未知参数: $1"; exit 1;;
    esac
done

# ── 激活conda环境 ─────────────────────────────────────────────────────────────
source /data/home/shujia/miniconda3/etc/profile.d/conda.sh
conda activate retfound

echo "======================================================"
echo "  眼底门控模型训练流程"
echo "  时间: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================"

# ── 步骤1: 数据准备 ────────────────────────────────────────────────────────────
echo ""
echo "[步骤1] 数据准备..."
python prepare_data.py \
    --max_fundus 10000 \
    --max_neg_cifar 5000 \
    --max_neg_acdc 2000 \
    $SKIP_CIFAR \
    $SKIP_ACDC

# 检查CSV是否生成
if [[ ! -f "train.csv" || ! -f "val.csv" ]]; then
    echo "[错误] 数据准备失败，未生成 train.csv / val.csv"
    exit 1
fi

echo "[步骤1] 完成"

# ── 步骤2: 模型训练 ────────────────────────────────────────────────────────────
echo ""
echo "[步骤2] 开始训练（arch=${ARCH}, epochs=${EPOCHS}, batch=${BATCH_SIZE}）..."
python train.py \
    --train_csv train.csv \
    --val_csv val.csv \
    --output_dir checkpoints \
    --arch "$ARCH" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE" \
    --lr "$LR" \
    --num_workers 8

echo ""
echo "======================================================"
echo "  训练完成！"
echo "  模型保存在: $SCRIPT_DIR/checkpoints/best_fundus_gate.pth"
echo "======================================================"
echo ""
echo "使用方法（Python）:"
echo "  from fundus_gate.inference import FundusGateChecker"
echo "  checker = FundusGateChecker()"
echo "  is_fundus, prob = checker.check('your_image.jpg')"
