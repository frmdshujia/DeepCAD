"""
diagnostics_cohort_sizes.py

比较三种 fundus-CMR 配对方案的规模：
  (A) 全量跨 instance 配对：所有有 fundus PNG 且有 CMR PC14 的 EID
  (B) 排除间隔期新发事件后：从 (A) 去掉 fundus 日期到 CMR 日期之间有心血管事件首次发生的 EID
  (C) 严格同次就诊：fundus 和 CMR 都在 instance 2 的配对（即当前 fundus_table.csv 的 ~1,983）

事件定义（可配置）:
  - 主要心脏事件：I21 (AMI), I22 (MI subsequent), I42 (cardiomyopathy),
                   I43 (cardiomyopathy in others), I50 (HF)
  - 延展心脏事件：+ I20 I24 I25 (ischemic HD), I47 I48 I49 (arrhythmias),
                     I34 I35 (valvular), I10-I13 I15 (HTN family)

CMR imaging date 近似：
  - fundus@instance 0 的 fundus 日期 ≈ master_long 中 'Date of attending assessment centre'
  - fundus@instance 1 的 fundus 日期 ≈ baseline date + 4.3y
  - fundus@instance 2 的 fundus 日期 ≈ baseline date + 8.5y
  - fundus@instance 3 的 fundus 日期 ≈ baseline date + 10.5y
  - CMR imaging date ≈ baseline date + 8.5y（instance 2）
"""

import os
import re
from datetime import timedelta

import numpy as np
import pandas as pd


MAJOR_HEART_ICDS = ['I21', 'I22', 'I42', 'I43', 'I50']
EXTENDED_HEART_ICDS = MAJOR_HEART_ICDS + [
    'I20', 'I24', 'I25',
    'I47', 'I48', 'I49',
    'I34', 'I35',
]
ALL_ICDS = EXTENDED_HEART_ICDS + ['I10', 'I11', 'I12', 'I13', 'I15',
                                  'E10', 'E11', 'E12', 'E13', 'E14']


INSTANCE_TO_YEARS = {0: 0.0, 1: 4.3, 2: 8.5, 3: 10.5}

FILE_PAT = re.compile(r'^(\d+)_(\d+)_(\d+)_(\d+)\.(png|jpg|jpeg)$', re.IGNORECASE)


def parse_fundus_dir(fundus_dir: str) -> pd.DataFrame:
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
            'fundus_path': os.path.join(fundus_dir, f),
        })
    return pd.DataFrame(rows)


def count_events_in_window(master_df: pd.DataFrame, icd_list: list,
                           window_start_col: str, window_end_col: str,
                           eids: set) -> pd.DataFrame:
    """对 master_df 中 eid in eids 的人，统计 icd_list 里任一事件
    首次报告日期是否落在 [window_start_col, window_end_col] 之间（闭区间）。
    返回 DataFrame: (eid, has_interval_event)。
    """
    sub = master_df[master_df['eid'].isin(eids)].copy()
    n_in = len(sub)
    if n_in == 0:
        return pd.DataFrame(columns=['eid', 'has_interval_event'])

    start = sub[window_start_col]
    end = sub[window_end_col]

    has_event = np.zeros(n_in, dtype=bool)
    for icd in icd_list:
        col = next(
            (c for c in master_df.columns
             if c.startswith(f'Date {icd} first reported')),
            None,
        )
        if col is None:
            continue
        d = pd.to_datetime(sub[col], errors='coerce')
        in_window = (d >= start) & (d <= end)
        has_event = has_event | in_window.fillna(False).values

    sub = sub.assign(has_interval_event=has_event)
    return sub[['eid', 'has_interval_event']]


def main():
    fundus_dir = '/data/home/home6/fundus_data/UKB/fundus_images'
    cmr_csv = 'contrastive_pretrain/preprocessed_data/modeling_delivery/cmr_table.csv'
    master_csv = 'contrastive_pretrain/raw/master_long.csv'
    strict_fundus_table = 'contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table.csv'

    # 1. fundus PNG inventory
    print('===== (1) Fundus PNG inventory =====')
    fundus = parse_fundus_dir(fundus_dir)
    print(f'  fundus PNGs: {len(fundus)} across {fundus["eid"].nunique()} EIDs')

    # 2. CMR PC14 inventory
    print('\n===== (2) CMR PC14 inventory =====')
    cmr = pd.read_csv(cmr_csv)
    cmr_i2 = cmr[cmr['instance'] == 2]
    print(f'  cmr_table rows: {len(cmr)}, unique eid: {cmr["eid"].nunique()}')
    print(f'  cmr instance=2 rows: {len(cmr_i2)}, unique eid: {cmr_i2["eid"].nunique()}')

    # 3. master_long dates
    print('\n===== (3) Master long table =====')
    master = pd.read_csv(master_csv, low_memory=False)
    print(f'  master rows: {len(master)}, unique eid: {master["eid"].nunique()}')
    master['baseline_date'] = pd.to_datetime(
        master['Date of attending assessment centre'], errors='coerce')
    n_has_date = master['baseline_date'].notna().sum()
    print(f'  EIDs with parsable baseline_date: {n_has_date}')

    cmr_eids = set(cmr_i2['eid'].unique())

    # 4. Cohort A: 全量跨 instance 配对（每 fundus 图一条配对）
    print('\n===== (A) 全量跨 instance 配对 =====')
    a = fundus.merge(pd.DataFrame({'eid': list(cmr_eids)}), on='eid', how='inner')
    print(f'  配对行数 (每张 fundus 一行): {len(a)}')
    print(f'  唯一 EID: {a["eid"].nunique()}')
    print(f'  fundus_instance 分布: {a["fundus_instance"].value_counts().to_dict()}')

    # 4.1 进一步限制只在 master_long 有 baseline_date 的人（绝大多数）
    eid_with_date = set(master.loc[master['baseline_date'].notna(), 'eid'].unique())
    a_dated = a[a['eid'].isin(eid_with_date)]
    print(f'  其中有 baseline_date: 行数={len(a_dated)}, EID={a_dated["eid"].nunique()}')

    # 5. Cohort B: 排除间隔期新发事件后
    print('\n===== (B) 排除间隔期新发心脏事件 =====')
    # 构造每个 (eid, fundus_instance) 的 fundus_date 和 cmr_date
    # fundus_date = baseline_date + fundus_instance_offset
    # cmr_date = baseline_date + 8.5y (instance 2)
    master_lookup = master[['eid', 'baseline_date']].drop_duplicates('eid')
    a_dated = a_dated.merge(master_lookup, on='eid', how='inner')
    a_dated['fundus_date'] = a_dated.apply(
        lambda r: r['baseline_date'] + pd.Timedelta(
            days=int(INSTANCE_TO_YEARS[int(r['fundus_instance'])] * 365.25)),
        axis=1,
    )
    a_dated['cmr_date'] = a_dated['baseline_date'] + pd.Timedelta(
        days=int(INSTANCE_TO_YEARS[2] * 365.25))

    # 对每 EID 只判一次事件是否在 [min(fundus_date), cmr_date] 之间
    # 简化：取每 EID 其最早 fundus_date 作为 window_start
    eid_window = a_dated.groupby('eid').agg(
        window_start=('fundus_date', 'min'),
        window_end=('cmr_date', 'first'),
    ).reset_index()
    # 合并到 master 行
    master_w = master.merge(eid_window, on='eid', how='inner')

    # 主要事件
    print('\n  [Rule B-major] 排除 I21/I22/I42/I43/I50 新发:')
    ev_major = count_events_in_window(
        master_w, MAJOR_HEART_ICDS, 'window_start', 'window_end',
        set(eid_window['eid']),
    )
    n_bad_major = ev_major['has_interval_event'].sum()
    good_eids_major = set(ev_major.loc[~ev_major['has_interval_event'], 'eid'])
    b_major = a_dated[a_dated['eid'].isin(good_eids_major)]
    print(f'    间隔期新发事件 EID: {n_bad_major}')
    print(f'    保留配对行数: {len(b_major)}, 唯一 EID: {b_major["eid"].nunique()}')

    # 扩展事件
    print('\n  [Rule B-extended] 排除 I20-I25/I34-I35/I42-I43/I47-I50 新发:')
    ev_ext = count_events_in_window(
        master_w, EXTENDED_HEART_ICDS, 'window_start', 'window_end',
        set(eid_window['eid']),
    )
    n_bad_ext = ev_ext['has_interval_event'].sum()
    good_eids_ext = set(ev_ext.loc[~ev_ext['has_interval_event'], 'eid'])
    b_ext = a_dated[a_dated['eid'].isin(good_eids_ext)]
    print(f'    间隔期新发事件 EID: {n_bad_ext}')
    print(f'    保留配对行数: {len(b_ext)}, 唯一 EID: {b_ext["eid"].nunique()}')

    # 全量扩展 (心脏 + 高血压 + 糖尿病)
    print('\n  [Rule B-all] 排除所有 I10-I50 + E10-E14 新发:')
    ev_all = count_events_in_window(
        master_w, ALL_ICDS, 'window_start', 'window_end',
        set(eid_window['eid']),
    )
    n_bad_all = ev_all['has_interval_event'].sum()
    good_eids_all = set(ev_all.loc[~ev_all['has_interval_event'], 'eid'])
    b_all = a_dated[a_dated['eid'].isin(good_eids_all)]
    print(f'    间隔期新发事件 EID: {n_bad_all}')
    print(f'    保留配对行数: {len(b_all)}, 唯一 EID: {b_all["eid"].nunique()}')

    # 6. Cohort C: 严格同次就诊（instance 2 的 fundus + instance 2 的 CMR）
    print('\n===== (C) 严格同次就诊（fundus@i2 + CMR@i2）=====')
    c = a[(a['fundus_instance'] == 2) & (a['eid'].isin(cmr_eids))]
    print(f'  配对行数: {len(c)}, 唯一 EID: {c["eid"].nunique()}')
    # 与现有 fundus_table.csv 对照
    ft = pd.read_csv(strict_fundus_table)
    print(f'  当前 fundus_table.csv 行数={len(ft)}, EID={ft["eid"].nunique()}')

    # 7. 汇总
    print('\n' + '=' * 60)
    print('Cohort size summary')
    print('=' * 60)
    print(f'{"Cohort":<40} {"Pairs (rows)":>12} {"Unique EID":>12}')
    print('-' * 60)
    print(f'{"A. 全量跨 instance 配对":<40} {len(a):>12,} {a["eid"].nunique():>12,}')
    print(f'{"   └ 有 baseline_date":<40} {len(a_dated):>12,} {a_dated["eid"].nunique():>12,}')
    print(f'{"B-major. 排 I21/22/42/43/50":<40} {len(b_major):>12,} {b_major["eid"].nunique():>12,}')
    print(f'{"B-ext. 排所有心脏事件":<40} {len(b_ext):>12,} {b_ext["eid"].nunique():>12,}')
    print(f'{"B-all. 排心脏+HTN+DM":<40} {len(b_all):>12,} {b_all["eid"].nunique():>12,}')
    print(f'{"C. 严格同次就诊 (instance 2)":<40} {len(c):>12,} {c["eid"].nunique():>12,}')
    print(f'{"   └ 当前 fundus_table.csv":<40} {len(ft):>12,} {ft["eid"].nunique():>12,}')


if __name__ == '__main__':
    main()
