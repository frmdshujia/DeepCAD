#!/bin/bash
# 将 OCT/Fundus 源目录下所有 PNG 移动到 UKB fundus 图像目录
# 需要 root 权限写入目标目录，请在终端执行：sudo bash 本脚本路径

set -euo pipefail

SRC="/data/home/shujia/Retinal Optical Coherence Tomography"
DST="/data/home/home6/fundus_data/UKB/fundus_images"

if [[ ! -d "$SRC" ]]; then
  echo "错误：源目录不存在: $SRC"
  exit 1
fi
if [[ ! -d "$DST" ]]; then
  echo "错误：目标目录不存在: $DST"
  exit 1
fi

N=$(find "$SRC" -type f -iname '*.png' 2>/dev/null | wc -l)
echo "即将移动 PNG 数量: $N"
echo "源: $SRC"
echo "目标: $DST"

# GNU find：{}+ 自动分批，避免参数过长
find "$SRC" -type f -iname '*.png' -exec mv -t "$DST" {} +

echo "完成。源目录剩余 PNG:"
find "$SRC" -type f -iname '*.png' 2>/dev/null | wc -l
