#!/bin/bash
# 快速测试脚本 - 小样本训练测试

echo "=========================================="
echo "DeepCAD 快速测试"
echo "=========================================="

# 设置变量（根据实际情况修改）
TRAIN_CSV="data/splits/train_mini.csv"
VAL_CSV="data/splits/train_mini.csv"
RETINAL_PRETRAINED="RETFound_mae"  # 或本地路径
MRI_PRETRAINED="checkpoints/pretrained/medsam/medsam_vit_b.pth"
BATCH_SIZE=2
NUM_EPOCHS=2
LR=1e-4

echo "配置:"
echo "  训练CSV: $TRAIN_CSV"
echo "  批次大小: $BATCH_SIZE"
echo "  训练轮数: $NUM_EPOCHS"
echo "  学习率: $LR"
echo ""

# 检查文件是否存在
if [ ! -f "$TRAIN_CSV" ]; then
    echo "❌ 错误: 训练CSV文件不存在: $TRAIN_CSV"
    echo "   请先创建测试数据"
    exit 1
fi

# 运行训练
echo "开始训练..."
python scripts/train_stage1.py \
    --train_csv "$TRAIN_CSV" \
    --val_csv "$VAL_CSV" \
    --retinal_pretrained "$RETINAL_PRETRAINED" \
    --mri_pretrained "$MRI_PRETRAINED" \
    --batch_size $BATCH_SIZE \
    --num_epochs $NUM_EPOCHS \
    --lr $LR \
    --retinal_base_path data \
    --mri_base_path data \
    --save_dir checkpoints/test \
    --log_dir logs/test \
    --num_workers 0

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ 快速测试通过！"
    echo "=========================================="
    echo ""
    echo "检查点保存在: checkpoints/test/"
    echo "日志保存在: logs/test/"
    echo ""
    echo "下一步: 可以开始正式训练了"
else
    echo ""
    echo "=========================================="
    echo "❌ 测试失败，请检查错误信息"
    echo "=========================================="
fi

