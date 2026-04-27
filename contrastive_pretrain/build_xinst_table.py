"""
build_xinst_table.py

生成 Cross-Instance fundus-CMR 配对表 fundus_cmr_xinst.csv。

列格式（与 fundus_table.csv 完全兼容，可直接被 FundusContrastDataset 读入）：
    eid, fundus_image_path, split,
    fundus_instance, fundus_field, fundus_array,
    cmr_instance, visit_interval_years,
    M1_PC1..M6_PC3  (14 列)

配对策略：
- 对每个同时有 fundus 和 CMR 的 EID：
    - 收集其所有 fundus PNG（左右眼 × 多 instance）
    - 取其 CMR 行中 instance=2 的 PC14（若无 instance=2 则取 instance=3，仍无则跳过）
    - 每个 fundus 图生成一行，PC14 = 该 EID 的 CMR PC14
- visit_interval_years：fundus_instance 到 cmr_instance 的典型年差
    - UKB imaging visit 典型间隔：0→0y, 1→4.3y, 2→8.5y, 3→10.5y
- split 继承自 cmr_table.csv（按 EID，同 EID 必同 split）

执行方式：
    python contrastive_pretrain/build_xinst_table.py \
        --fundus_dir /data/home/home6/fundus_data/UKB/fundus_images \
        --cmr_csv contrastive_pretrain/preprocessed_data/cmr_table.csv \
        --output contrastive_pretrain/preprocessed_data/fundus_cmr_xinst.csv
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


# UKB imaging visit typical assessment center dates (approximated; will be
# refined with field 53 if needed later).
INSTANCE_TO_YEARS_FROM_BASELINE = {
    0: 0.0,   # initial assessment 2006-2010
    1: 4.3,   # repeat assessment 2012-2013
    2: 8.5,   # imaging visit 1 (includes CMR) 2014-2020
    3: 10.5,  # imaging visit 2 (repeat imaging) 2019-ongoing
}


FILE_PAT = re.compile(r'^(\d+)_(\d+)_(\d+)_(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)

PC_COLS = [
    'M1_PC1', 'M1_PC2',
    'M2_PC1', 'M2_PC2', 'M2_PC3',
    'M3_PC1', 'M3_PC2',
    'M4_PC1', 'M4_PC2',
    'M5_PC1', 'M5_PC2',
    'M6_PC1', 'M6_PC2', 'M6_PC3',
]


def parse_fundus_dir(fundus_dir: str) -> pd.DataFrame:
    """扫描 fundus 目录，解析文件名得到 DataFrame。"""
    if not os.path.isdir(fundus_dir):
        raise FileNotFoundError(f'fundus_dir not found: {fundus_dir}')
    files = os.listdir(fundus_dir)
    rows = []
    for f in files:
        m = FILE_PAT.match(f)
        if not m:
            continue
        rows.append({
            'eid': int(m.group(1)),
            'fundus_field': int(m.group(2)),
            'fundus_instance': int(m.group(3)),
            'fundus_array': int(m.group(4)),
            'fundus_image_path': os.path.join(fundus_dir, f),
        })
    df = pd.DataFrame(rows)
    print(f'[fundus] parsed {len(df)} PNGs across {df["eid"].nunique()} EIDs from {fundus_dir}')
    return df


def load_cmr(cmr_csv: str) -> pd.DataFrame:
    df = pd.read_csv(cmr_csv)
    missing = [c for c in PC_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'CMR csv missing cols: {missing}')
    if 'instance' not in df.columns:
        raise ValueError('CMR csv must have "instance" column.')
    if 'split' not in df.columns:
        raise ValueError('CMR csv must have "split" column.')
    print(f'[cmr] rows={len(df)}, unique_eid={df["eid"].nunique()}, '
          f'instance_dist={df["instance"].value_counts().to_dict()}, '
          f'split_dist={df["split"].value_counts().to_dict()}')
    return df


def pick_cmr_row_per_eid(cmr_df: pd.DataFrame) -> pd.DataFrame:
    """每 EID 选一行 CMR：优先 instance=2，否则 instance=3。"""
    order = {2: 0, 3: 1, 0: 2, 1: 3}
    cmr_df = cmr_df.copy()
    cmr_df['_inst_order'] = cmr_df['instance'].map(order).fillna(99).astype(int)
    cmr_df = cmr_df.sort_values(['eid', '_inst_order']).drop_duplicates(subset='eid', keep='first')
    cmr_df = cmr_df.drop(columns=['_inst_order']).reset_index(drop=True)
    print(f'[cmr] after per-EID pick: rows={len(cmr_df)}, '
          f'instance_dist={cmr_df["instance"].value_counts().to_dict()}')
    return cmr_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fundus_dir', required=True, type=str,
                    help='Directory containing fundus PNGs named as {eid}_{field}_{instance}_{array}.png')
    ap.add_argument('--cmr_csv', required=True, type=str,
                    help='cmr_table.csv with eid/instance/split/M*_PC* columns')
    ap.add_argument('--output', required=True, type=str,
                    help='Output csv path.')
    ap.add_argument('--exclude_paired_table', type=str, default=None,
                    help='(Optional) existing same-instance fundus_table.csv. '
                         'Rows with (eid,instance) present here will be EXCLUDED '
                         'so Stage 2 image-level pairing remains held-out.')
    args = ap.parse_args()

    fundus_df = parse_fundus_dir(args.fundus_dir)
    cmr_df = load_cmr(args.cmr_csv)

    cmr_picked = pick_cmr_row_per_eid(cmr_df)

    # Merge: fundus ↔ CMR per EID
    merged = fundus_df.merge(
        cmr_picked[['eid', 'instance', 'split'] + PC_COLS],
        on='eid',
        how='inner',
        suffixes=('', '_cmr'),
    )
    merged = merged.rename(columns={'instance': 'cmr_instance'})
    print(f'[merge] cross-instance paired rows: {len(merged)}, '
          f'unique_eid: {merged["eid"].nunique()}')

    # Optional: exclude rows already used by Stage 2 (same-instance paired)
    if args.exclude_paired_table and os.path.exists(args.exclude_paired_table):
        ex = pd.read_csv(args.exclude_paired_table)
        if {'eid', 'instance'}.issubset(ex.columns):
            ex_keys = set(
                (int(e), int(i)) for e, i in zip(ex['eid'].tolist(), ex['instance'].tolist())
            )
            mask = [(int(e), int(i)) not in ex_keys
                    for e, i in zip(merged['eid'].tolist(), merged['fundus_instance'].tolist())]
            n_before = len(merged)
            merged = merged[mask].reset_index(drop=True)
            print(f'[exclude] removed {n_before - len(merged)} rows '
                  f'already in {args.exclude_paired_table} (same-instance pair reserved for Stage 2)')

    # Compute visit_interval_years
    merged['visit_interval_years'] = (
        merged['cmr_instance'].map(INSTANCE_TO_YEARS_FROM_BASELINE).astype(float)
        - merged['fundus_instance'].map(INSTANCE_TO_YEARS_FROM_BASELINE).astype(float)
    )

    # Reorder columns
    out_cols = [
        'eid', 'fundus_image_path', 'split',
        'fundus_instance', 'fundus_field', 'fundus_array',
        'cmr_instance', 'visit_interval_years',
    ] + PC_COLS
    merged = merged[out_cols]

    # Sanity checks
    if merged[PC_COLS].isna().any().any():
        n_bad = merged[PC_COLS].isna().any(axis=1).sum()
        print(f'[warn] {n_bad} rows have NaN in PC columns; dropping.')
        merged = merged.dropna(subset=PC_COLS).reset_index(drop=True)

    # Existence check (fast: stat a sample of paths)
    sample_paths = merged['fundus_image_path'].sample(
        min(200, len(merged)), random_state=0).tolist()
    n_missing = sum(1 for p in sample_paths if not os.path.exists(p))
    if n_missing:
        print(f'[warn] {n_missing}/{len(sample_paths)} sample paths not accessible; '
              f'FundusContrastDataset will filter automatically.')

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    merged.to_csv(args.output, index=False)

    print('\n=== summary ===')
    print(f'rows: {len(merged)}')
    print(f'unique_eid: {merged["eid"].nunique()}')
    print(f'split dist: {merged["split"].value_counts().to_dict()}')
    print(f'fundus_instance dist: {merged["fundus_instance"].value_counts().to_dict()}')
    print(f'cmr_instance dist: {merged["cmr_instance"].value_counts().to_dict()}')
    print(f'visit_interval_years dist: {merged["visit_interval_years"].value_counts().to_dict()}')
    print(f'saved -> {args.output}')


if __name__ == '__main__':
    main()
