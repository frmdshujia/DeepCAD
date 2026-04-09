#!/bin/bash
# 将当前服务器上的 retfound conda 环境导出，供目标服务器复用
# 在当前服务器（node5）上执行: bash export_env.sh

ENV_NAME="retfound"
OUTPUT_DIR="$(dirname "$0")"

echo "[1] 导出 conda 环境 YAML（含 pip 依赖）..."
conda env export -n "$ENV_NAME" > "$OUTPUT_DIR/retfound_env.yml"
echo "    ✓ 已保存: $OUTPUT_DIR/retfound_env.yml"

echo ""
echo "[2] 目标服务器上的恢复命令："
echo "    conda env create -f retfound_env.yml"
echo "    conda activate retfound"
echo ""
echo "[3] 若目标服务器 CPU 类型不同，建议改用 pip 最小安装："
echo "    pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu"
echo "    pip install flask flask-cors Pillow timm==0.3.2 psutil"
echo ""
echo "完成。"
