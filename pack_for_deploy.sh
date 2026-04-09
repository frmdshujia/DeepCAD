#!/bin/bash
# DeepCAD 部署打包脚本
# 用法: ./pack_for_deploy.sh [输出目录]

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

OUT_DIR="${1:-.}"
PKG_NAME="deepcad_deploy_$(date +%Y%m%d_%H%M).tar.gz"

echo "============================================"
echo "  DeepCAD 部署打包"
echo "============================================"
echo "  项目根目录: $SCRIPT_DIR"
echo "  输出文件:   $OUT_DIR/$PKG_NAME"
echo "============================================"

tar -czvf "$OUT_DIR/$PKG_NAME" \
  --exclude='*.pyc' \
  --exclude='__pycache__' \
  --exclude='.git' \
  --exclude='.specstory' \
  --exclude='.cursor*' \
  --exclude='fundus_gate/.cifar_cache' \
  --exclude='fundus_gate/*.csv' \
  --exclude='fundus_gate/*.log' \
  --exclude='fundus_gate/prob_stats*' \
  --exclude='infer_dist/results' \
  --exclude='infer_dist/*.csv' \
  --exclude='infer_dist/inference_log*' \
  web \
  fundus_gate \
  models_vit.py \
  models_mae.py \
  util \
  infer_dist/dist_data.json \
  deploy_sync.md

echo ""
echo "============================================"
echo "  打包完成: $OUT_DIR/$PKG_NAME"
echo "  大小: $(du -h "$OUT_DIR/$PKG_NAME" | cut -f1)"
echo "============================================"
echo ""
echo "上传到远程: scp $OUT_DIR/$PKG_NAME user@remote:/path/to/"
echo "远程解压:   tar -xzvf $PKG_NAME"
echo "远程启动:   cd web && bash deploy/start.sh cpu 8000"
echo ""
