"""
build_training_csv.py

从 stage1_cmr.csv 和 stage1_fundus.csv 生成下采样训练集，并按 8/1/1 切分。

产出文件（保存在本脚本所在目录）：
  stage1_cmr_sampled.csv
  stage1_cmr_sampled_train.csv / _val.csv / _test.csv
  stage1_fundus_sampled.csv
  stage1_fundus_sampled_train.csv / _val.csv / _test.csv

运行：
  /data/home/shujia/miniconda3/bin/python build_training_csv.py
"""

import os
import pathlib
import pandas as pd
import numpy as np

# ── 路径 ──────────────────────────────────────────────────────────────────────
HERE      = pathlib.Path(__file__).parent
DATA_DIR  = HERE / 'preprocessed_data'
STAGE1    = DATA_DIR / 'stage1'

CMR_CSV    = STAGE1 / 'stage1_cmr.csv'
FUNDUS_CSV = STAGE1 / 'stage1_fundus.csv'

# ── 常量 ──────────────────────────────────────────────────────────────────────
CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]
REG_COL = 'LV ejection fraction'
LV_COLS = ['LV ejection fraction', 'LV end diastolic volume', 'LV end systolic volume']
N_NEG   = 10_000
SEED    = 42


# ── 核心函数 ──────────────────────────────────────────────────────────────────
def smart_downsample(df, n_neg=N_NEG, cls_cols=CLS_COLS, reg_col=REG_COL, seed=SEED):
    """三优先级采样（文档 §2.2）"""
    df = df.copy()

    is_pos  = (df[cls_cols] == 1).any(axis=1)
    has_reg = df[reg_col].notna() if reg_col in df.columns else pd.Series(False, index=df.index)

    pos_df       = df[is_pos]
    neg_with_reg = df[~is_pos & has_reg]
    neg_pure     = df[~is_pos & ~has_reg]

    n_from_reg  = min(len(neg_with_reg), n_neg)
    n_from_pure = max(0, n_neg - n_from_reg)

    neg_sampled = pd.concat([
        neg_with_reg.sample(n_from_reg,  random_state=seed),
        neg_pure.sample(min(n_from_pure, len(neg_pure)), random_state=seed),
    ])

    final = pd.concat([pos_df, neg_sampled]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f'  阳性（全保留）:      {len(pos_df):>8,}')
    print(f'  阴性-有回归标签:    {n_from_reg:>8,}')
    print(f'  阴性-纯阴性补充:    {n_from_pure:>8,}')
    print(f'  总计:               {len(final):>8,}')
    print()
    for col in cls_cols:
        n_pos = (final[col] == 1).sum()
        print(f'    {col:<40} 阳性: {n_pos:,} ({n_pos/len(final):.2%})')
    return final


def split_80_10_10(df, seed=SEED):
    """按 EID 切分 80/10/10"""
    eids = df['eid'].unique()
    rng  = np.random.default_rng(seed)
    rng.shuffle(eids)
    n = len(eids)
    n_val  = n // 10
    n_test = n // 10
    val_eids  = set(eids[:n_val])
    test_eids = set(eids[n_val:n_val + n_test])
    train_eids = set(eids[n_val + n_test:])

    train = df[df['eid'].isin(train_eids)].reset_index(drop=True)
    val   = df[df['eid'].isin(val_eids)].reset_index(drop=True)
    test  = df[df['eid'].isin(test_eids)].reset_index(drop=True)
    return train, val, test


# ── CMR ───────────────────────────────────────────────────────────────────────
print('=' * 60)
print('CMR 人群下采样')
print('=' * 60)

cmr = pd.read_csv(CMR_CSV, low_memory=False)
print(f'原始行数: {len(cmr):,}，唯一EID: {cmr["eid"].nunique():,}')

# 去重：同一 EID 多个 instance，优先保留 instance=2；若无则保留最大 instance
cmr_sorted = cmr.sort_values(['eid', 'instance'], ascending=[True, True])
cmr_dedup  = cmr_sorted.groupby('eid', as_index=False).apply(
    lambda g: g[g['instance'] == 2].iloc[0] if (g['instance'] == 2).any() else g.iloc[-1]
).reset_index(drop=True)
print(f'去重后: {len(cmr_dedup):,} 行\n')

cmr_sampled = smart_downsample(cmr_dedup)

out_cmr = HERE / 'stage1_cmr_sampled.csv'
cmr_sampled.to_csv(out_cmr, index=False)
print(f'\n已保存: {out_cmr}')

cmr_train, cmr_val, cmr_test = split_80_10_10(cmr_sampled)
cmr_train.to_csv(HERE / 'stage1_cmr_sampled_train.csv', index=False)
cmr_val.to_csv(HERE / 'stage1_cmr_sampled_val.csv',   index=False)
cmr_test.to_csv(HERE / 'stage1_cmr_sampled_test.csv',  index=False)
print(f'训练/验证/测试 分割: {len(cmr_train):,} / {len(cmr_val):,} / {len(cmr_test):,}')

# 回归标签统计
for col in LV_COLS:
    n_reg = cmr_sampled[col].notna().sum()
    print(f'  {col} 有标签人数: {n_reg:,}')


# ── 眼底 ──────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('眼底人群下采样（从 stage1_cmr 按 EID 补充 LV 标签）')
print('=' * 60)

fundus = pd.read_csv(FUNDUS_CSV, low_memory=False)
print(f'原始行数: {len(fundus):,}，唯一EID: {fundus["eid"].nunique():,}')

# 去重：保留 instance 最大的那条（最新访视）
fundus_dedup = (fundus.sort_values(['eid', 'instance'], ascending=[True, True])
                      .groupby('eid', as_index=False).last()
                      .reset_index(drop=True))
print(f'去重后: {len(fundus_dedup):,} 行')

# 从 stage1_cmr 提取 LV 列（按 EID join，优先取 instance=2）
# 33,666 个 EID 同时存在于眼底和CMR，这就是"宽松配对池（跨visit）"
cmr_lv_raw = cmr[['eid', 'instance'] + LV_COLS].copy()
# 每个EID只保留一条：优先 instance=2（主访），否则取最大 instance
cmr_lv_dedup = (cmr_lv_raw.sort_values(['eid', 'instance'], ascending=[True, True])
                           .groupby('eid', as_index=False)
                           .apply(lambda g: g[g['instance'] == 2].iloc[0]
                                  if (g['instance'] == 2).any() else g.iloc[-1],
                                  include_groups=False)
                           .reset_index(drop=True))
cmr_lv_dedup = cmr_lv_dedup[['eid'] + LV_COLS]

# left join：33,666 个交集EID得到LV列，其余保持NaN
fundus_aug = fundus_dedup.merge(cmr_lv_dedup, on='eid', how='left')
n_with_reg = fundus_aug['LV ejection fraction'].notna().sum()
cross_eids = len(set(fundus_dedup['eid']) & set(cmr_lv_dedup['eid']))
print(f'眼底∩CMR 交集EID数（宽松配对池）: {cross_eids:,}')
print(f'眼底中有LV标签人数: {n_with_reg:,}（共 {len(fundus_aug):,} 人）\n')

fundus_sampled = smart_downsample(fundus_aug)

out_fundus = HERE / 'stage1_fundus_sampled.csv'
fundus_sampled.to_csv(out_fundus, index=False)
print(f'\n已保存: {out_fundus}')

fundus_train, fundus_val, fundus_test = split_80_10_10(fundus_sampled)
fundus_train.to_csv(HERE / 'stage1_fundus_sampled_train.csv', index=False)
fundus_val.to_csv(HERE / 'stage1_fundus_sampled_val.csv',   index=False)
fundus_test.to_csv(HERE / 'stage1_fundus_sampled_test.csv',  index=False)
print(f'训练/验证/测试 分割: {len(fundus_train):,} / {len(fundus_val):,} / {len(fundus_test):,}')

for col in LV_COLS:
    n_reg = fundus_sampled[col].notna().sum()
    print(f'  {col} 有标签人数: {n_reg:,}')

print('\n[DONE] build_training_csv.py 完成')
