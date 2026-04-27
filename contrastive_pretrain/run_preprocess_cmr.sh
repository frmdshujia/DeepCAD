#!/bin/bash
# run_preprocess_cmr.sh
#
# 批量将 CMR zip 文件预处理为 (4,224,224) float16 npy，供 Stage1 训练使用。
# 已处理的 EID 自动跳过（幂等）。
#
# 用法：
#   bash contrastive_pretrain/run_preprocess_cmr.sh [eid_file]
#   默认读取 contrastive_pretrain/stage1_cmr_sampled_train.csv 中所有 EID

set -e
PYTHON=/data/home/shujia/miniconda3/bin/python
CMR_DIR=/data/home/shujia/UKB/CMRI/downloaded
OUT_DIR=/data/home/shujia/UKB/CMRI/preprocessed_lax_sax
WORKERS=${WORKERS:-8}
HERE=$(dirname "$0")

# Build EID list from CMR sampled CSV
EID_FILE=/tmp/stage1_cmr_train_eids.txt
$PYTHON -c "
import pandas as pd
df = pd.read_csv('${HERE}/stage1_cmr_sampled_train.csv')
eids = df['eid'].astype(int).unique()
with open('$EID_FILE', 'w') as f:
    for e in eids:
        f.write(str(e) + '\n')
print(f'Written {len(eids)} EIDs to $EID_FILE')
"

echo "Preprocessing CMR: EIDs in $EID_FILE -> $OUT_DIR (workers=$WORKERS)"
mkdir -p $OUT_DIR

$PYTHON $HERE/preprocess_cmr_lax_sax.py \
  --eid_file $EID_FILE \
  --cmr_dir  $CMR_DIR \
  --out_dir  $OUT_DIR \
  --num_workers $WORKERS

echo "Done. Check $OUT_DIR"
