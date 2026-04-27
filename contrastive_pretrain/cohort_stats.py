"""cohort_stats.py — 统计需求全表输出"""
import os, sys, json
import numpy as np
import pandas as pd

PY_DATA = 'contrastive_pretrain/preprocessed_data'
CMR_NPY_DIR = '/data/home/shujia/UKB/CMRI/preprocessed_lax_sax'

# 加载四张表
cmr_all  = pd.read_csv(f'{PY_DATA}/stage1/stage1_cmr.csv',    low_memory=False)
fund_all = pd.read_csv(f'{PY_DATA}/stage1/stage1_fundus.csv', low_memory=False)
dual     = pd.read_csv(f'{PY_DATA}/stage2/fundus/stage2_fundus_dual.csv', low_memory=False)

# ── 定义三个基础人群（各取 instance=2 的唯一 EID）──────────────────────────
cmr_pop  = cmr_all[cmr_all['instance']==2].drop_duplicates('eid').copy()   # CMR人群
fund_pop = fund_all.drop_duplicates('eid').copy()                           # 眼底人群（所有visit去重）
dual_pop = dual.drop_duplicates('eid').copy()                               # 配对人群

SEP = '─' * 72

def pct(n, tot):
    return f'{n} / {tot} / {100*n/tot:.2f}%' if tot > 0 else f'{n} / {tot} / —'

def num_stat(series, label=''):
    s = series.dropna()
    if len(s) == 0:
        return f'非空n=0'
    return (f'非空n={len(s)}, 均值={s.mean():.2f}±{s.std():.2f}, '
            f'P25={s.quantile(0.25):.2f}, 中位数={s.median():.2f}, '
            f'P75={s.quantile(0.75):.2f}, 最小={s.min():.2f}, 最大={s.max():.2f}')


# ════════════════════════════════════════════════════════════════════
print(SEP)
print('表1  各模态人群基础规模')
print(SEP)

# 眼底唯一EID
fund_uid = fund_all['eid'].nunique()
print(f'眼底人群唯一EID数 (stage1_fundus):          {fund_uid}')

# CMR唯一EID
cmr_uid  = cmr_all['eid'].nunique()
print(f'CMR人群唯一EID数 (stage1_cmr):              {cmr_uid}')

# 同时有LAX和SAX的CMR人数 (instance=2)
has_sax = cmr_pop['Short axis heart images - DICOM'].notna()
has_lax = cmr_pop['Long axis heart images - DICOM'].notna()
both_seq = (has_sax & has_lax).sum()
print(f'CMR instance=2 同时有LAX+SAX的人数:        {both_seq}')

# 眼底左右眼统计
left_col  = 'Fundus retinal eye image (left)'
right_col = 'Fundus retinal eye image (right)'
has_left  = fund_pop[left_col].notna().sum()
has_right = fund_pop[right_col].notna().sum()
has_both  = (fund_pop[left_col].notna() & fund_pop[right_col].notna()).sum()
print(f'眼底人群中有左眼的人数:                      {has_left}')
print(f'眼底人群中有右眼的人数:                      {has_right}')
print(f'眼底人群中左右眼都有的人数:                  {has_both}')


# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表2  配对人群规模')
print(SEP)

# 严格配对：同一 EID 在 cmr(instance=2) 和 fund(instance=2) 均有记录
fund_inst2 = fund_all[fund_all['instance']==2].drop_duplicates('eid')
strict_eids = set(cmr_pop['eid']) & set(fund_inst2['eid'])
print(f'同visit严格配对 (CMR inst=2 ∩ Fundus inst=2): {len(strict_eids)}')

# 宽松配对：任意 EID 出现在两表
loose_eids = set(cmr_all['eid']) & set(fund_all['eid'])
print(f'跨visit宽松配对 (EID取交集):                 {len(loose_eids)}')

# ── 日期间隔：宽松配对，取 CMR instance=2 日期 vs 眼底最早日期 ──────────
cmr_dates = cmr_pop[['eid','Date of attending assessment centre']].rename(
    columns={'Date of attending assessment centre':'cmr_date'})
fund_dates = fund_all[fund_all['eid'].isin(loose_eids)].copy()
fund_dates = fund_dates.sort_values(['eid','instance']).groupby('eid').first()[
    ['Date of attending assessment centre']].rename(
    columns={'Date of attending assessment centre':'fund_date'}).reset_index()

merged = cmr_dates.merge(fund_dates, on='eid', how='inner')
merged['cmr_date']  = pd.to_datetime(merged['cmr_date'],  errors='coerce')
merged['fund_date'] = pd.to_datetime(merged['fund_date'], errors='coerce')
merged = merged.dropna(subset=['cmr_date','fund_date'])
merged['gap_yr'] = (merged['cmr_date'] - merged['fund_date']).dt.days.abs() / 365.25

gap = merged['gap_yr']
print(f'\n配对人群眼底↔CMR采集时间间隔（年，n={len(gap)}）:')
print(f'  中位数={gap.median():.2f}  P25={gap.quantile(0.25):.2f}  P75={gap.quantile(0.75):.2f}  '
      f'最小={gap.min():.2f}  最大={gap.max():.2f}')

le2   = (gap <= 2).sum()
y2_5  = ((gap > 2) & (gap <= 5)).sum()
gt5   = (gap > 5).sum()
tot_g = len(gap)
print(f'  间隔 ≤2年:  {pct(le2,  tot_g)}')
print(f'  间隔 2-5年: {pct(y2_5, tot_g)}')
print(f'  间隔 >5年:  {pct(gt5,  tot_g)}')


# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表3  分类任务标签统计')
print(SEP)

tasks_cls = [
    ('composite_ischemic_hd',     '缺血性心脏病(复合)'),
    ('prevalent_I21',             '心肌梗死 I21'),
    ('composite_cardiomyopathy_hf','心肌病/心衰(复合)'),
    ('composite_af_arrhythmia',   '房颤/心律失常(复合)'),
    ('prevalent_I25',             '慢性缺血性心脏病 I25'),
    ('prevalent_I50',             '心力衰竭 I50'),
    ('composite_hypertensive_hd', '高血压性心脏病(复合)'),
]

# 三个人群
pops = {
    'CMR人群 (inst=2)': cmr_pop,
    '眼底人群':         fund_pop,
    '配对人群':         dual_pop,
}

fmt_hdr = f"{'任务':<30} {'CMR人群':^28} {'眼底人群':^28} {'配对人群':^28}"
print(fmt_hdr)
print('─' * 118)
for col, label in tasks_cls:
    row = f'{label:<30} '
    for pname, pop in pops.items():
        if col not in pop.columns:
            row += f"{'无此列':^28} "
            continue
        s = pop[col].dropna()
        pos = (s == 1).sum()
        row += f"{pct(pos, len(s)):<28} "
    print(row)


# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表4  回归任务标签统计')
print(SEP)

reg_tasks = [
    ('LV ejection fraction',    'LVEF (%)'),
    ('LV end diastolic volume', 'LVEDV (mL)'),
    ('LV end systolic volume',  'LVESV (mL)'),
]

for col, label in reg_tasks:
    print(f'\n── {label} ──')
    for pname, pop in pops.items():
        if col not in pop.columns:
            print(f'  {pname}: 无此列')
            continue
        print(f'  {pname}: {num_stat(pop[col])}')


# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表5  人口学基本特征')
print(SEP)

age_col = 'Age when attended assessment centre'
sex_col = 'Sex'
bmi_col = 'Body mass index (BMI)'

for pname, pop in pops.items():
    print(f'\n── {pname} (n={len(pop)}) ──')
    # 年龄
    age = pop[age_col].dropna()
    print(f'  年龄: {age.mean():.1f}±{age.std():.1f}  中位数={age.median():.1f} '
          f'(P25={age.quantile(0.25):.1f}, P75={age.quantile(0.75):.1f})')
    # 性别 (UKB 字符串 'Male'/'Female')
    if sex_col in pop.columns:
        male   = (pop[sex_col] == 'Male').sum()
        female = (pop[sex_col] == 'Female').sum()
        tot_s  = len(pop)
        print(f'  男性: {male} ({100*male/tot_s:.1f}%)  女性: {female} ({100*female/tot_s:.1f}%)')
    # BMI
    bmi = pop[bmi_col].dropna()
    print(f'  BMI: {bmi.mean():.1f}±{bmi.std():.1f}')
    # 吸烟（UKB 无直接字段，跳过）
    print(f'  吸烟史: 数据表中无吸烟字段（UKB字段 1239/1249 未纳入当前表）')


# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表6  眼底→心脏结构回归的可用人数')
print(SEP)

# 眼底人群中有 CMR 标注的人数
# 方法：fund_pop EID 与 cmr_pop 做 join，看 LVEF/LVEDV 非空
fund_join = fund_pop[['eid']].merge(
    cmr_pop[['eid','LV ejection fraction','LV end diastolic volume']],
    on='eid', how='left')
lvef_nn   = fund_join['LV ejection fraction'].notna().sum()
lvedv_nn  = fund_join['LV end diastolic volume'].notna().sum()
both_nn   = (fund_join['LV ejection fraction'].notna() & fund_join['LV end diastolic volume'].notna()).sum()
print(f'眼底人群中 LVEF 非空 (join CMR):            {lvef_nn}')
print(f'眼底人群中 LVEDV 非空:                      {lvedv_nn}')
print(f'眼底人群中 LVEF+LVEDV 同时非空:             {both_nn}')

# 与配对人群(stage2_fundus_dual)的重叠
overlap_eids = set(fund_join[fund_join['LV ejection fraction'].notna()]['eid']) & set(dual_pop['eid'])
print(f'上述人群与配对人群(stage2_fundus_dual)重叠:  {len(overlap_eids)}')


# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表7  CMR序列可用性')
print(SEP)

# instance=2 人群
has_sax_n  = has_sax.sum()
has_lax_n  = has_lax.sum()
print(f'CMR inst=2 有LAX (field 20208) 的人数:     {has_lax_n}')
print(f'CMR inst=2 有SAX (field 20209) 的人数:     {has_sax_n}')
print(f'CMR inst=2 同时有LAX+SAX:                  {both_seq}')

# T1 mapping (field 20213) —— stage1_cmr 未包含该字段
t1_cols = [c for c in cmr_all.columns if '20213' in c or 'T1' in c.lower() and 'mapping' in c.lower()]
if t1_cols:
    has_t1 = cmr_pop[t1_cols[0]].notna().sum()
    print(f'有T1 mapping (field 20213):              {has_t1}')
else:
    print(f'T1 mapping (field 20213): 当前stage1_cmr.csv未包含该字段，需从原始UKB数据提取')

# npy 文件 shape 检查（采样 200 个）
print(f'\nnpy 文件目录: {CMR_NPY_DIR}')
if os.path.exists(CMR_NPY_DIR):
    npy_files = [f for f in os.listdir(CMR_NPY_DIR) if f.endswith('.npy')]
    print(f'npy 文件总数: {len(npy_files)}')
    shape_counter = {}
    bad = []
    sample = npy_files[:200]
    for fn in sample:
        try:
            arr = np.load(os.path.join(CMR_NPY_DIR, fn), mmap_mode='r')
            sh = tuple(arr.shape)
            shape_counter[sh] = shape_counter.get(sh, 0) + 1
        except Exception as e:
            bad.append((fn, str(e)))
    print('shape 分布 (采样200个):')
    for sh, cnt in sorted(shape_counter.items(), key=lambda x: -x[1]):
        tag = ' ← 正常' if sh == (4, 224, 224) else ' ← 异常!'
        print(f'  {sh}: {cnt} 个{tag}')
    if bad:
        print(f'加载异常文件 ({len(bad)} 个): {bad[:5]}')
    # 估算全量 (4,224,224) 数量
    std_rate = shape_counter.get((4,224,224), 0) / len(sample)
    est_std  = int(std_rate * len(npy_files))
    print(f'估算全量标准shape(4,224,224)文件数: ~{est_std} / {len(npy_files)}')
else:
    print(f'目录不存在，跳过')

print()
print(SEP)
print('统计完成')
print(SEP)
