"""cohort_stats_v2.py — 按更正口径重新统计全表"""
import os
import numpy as np
import pandas as pd

PY_DATA    = 'contrastive_pretrain/preprocessed_data'
CMR_NPY_DIR = '/data/home/shujia/UKB/CMRI/preprocessed_lax_sax'

# ── 加载数据 ──────────────────────────────────────────────────────────────
cmr_all  = pd.read_csv(f'{PY_DATA}/stage1/stage1_cmr.csv',    low_memory=False)
fund_all = pd.read_csv(f'{PY_DATA}/stage1/stage1_fundus.csv', low_memory=False)
dual     = pd.read_csv(f'{PY_DATA}/stage2/fundus/stage2_fundus_dual.csv', low_memory=False)

# ── 三个基础人群 ───────────────────────────────────────────────────────────
# CMR人群：instance=2 唯一EID
cmr_pop  = cmr_all[cmr_all['instance'] == 2].drop_duplicates('eid').copy()
# 眼底人群：所有instance 唯一EID
fund_pop = fund_all.drop_duplicates('eid').copy()
# 配对人群：stage2_fundus_dual 唯一EID
dual_pop = dual.drop_duplicates('eid').copy()

SEP = '─' * 74

def pct(n, tot):
    return f'{n} / {tot} / {100*n/tot:.2f}%' if tot > 0 else f'{n} / {tot} / —'

def num_stat(series):
    s = series.dropna()
    if len(s) == 0:
        return '非空n=0'
    return (f'非空n={len(s)},  均值={s.mean():.2f}±{s.std():.2f},  '
            f'P25={s.quantile(0.25):.2f},  中位数={s.median():.2f},  '
            f'P75={s.quantile(0.75):.2f},  最小={s.min():.2f},  最大={s.max():.2f}')

# ── 要求1 修正：has_zip 过滤 "0"/"0.0"/"nan" ─────────────────────────────
def has_zip(x):
    if pd.isna(x):
        return False
    if isinstance(x, (int, float)) and x == 0:
        return False
    s = str(x).strip()
    return len(s) > 0 and s not in ('0', '0.0', 'nan')

COL_LAX = 'Long axis heart images - DICOM'
COL_SAX = 'Short axis heart images - DICOM'

# ════════════════════════════════════════════════════════════════════
print(SEP)
print('表1  各模态人群基础规模')
print(SEP)

fund_uid = fund_all['eid'].nunique()
cmr_uid  = cmr_all['eid'].nunique()

# Q1 更正：has_zip 过滤 + 唯一EID
cond_lax = cmr_pop[COL_LAX].map(has_zip)
cond_sax = cmr_pop[COL_SAX].map(has_zip)
both_seq = cmr_pop.loc[cond_lax & cond_sax, 'eid'].nunique()

# 眼底左右眼
left_col  = 'Fundus retinal eye image (left)'
right_col = 'Fundus retinal eye image (right)'
has_left  = fund_pop[left_col].map(has_zip).sum()
has_right = fund_pop[right_col].map(has_zip).sum()
has_both  = (fund_pop[left_col].map(has_zip) & fund_pop[right_col].map(has_zip)).sum()

print(f'眼底人群唯一EID数:                {fund_uid}    （已符合）')
print(f'CMR人群唯一EID数:                 {cmr_uid}    （已符合）')
print(f'CMR同时有LAX+SAX唯一EID数:        {both_seq}   【更正：has_zip过滤】')
print(f'眼底有左眼人数:                   {has_left}   【更正：has_zip过滤】')
print(f'眼底有右眼人数:                   {has_right}  【更正：has_zip过滤】')
print(f'眼底左右眼都有人数:               {has_both}   【更正：has_zip过滤】')

# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表2  配对人群规模')
print(SEP)

# Q2 更正：不限instance=2，包含所有成像visit，inner merge on [eid, instance]
cmr_inst  = cmr_all[['eid', 'instance']].drop_duplicates()
fund_inst = fund_all[['eid', 'instance']].drop_duplicates()
strict_merged = cmr_inst.merge(fund_inst, on=['eid', 'instance'], how='inner')
strict_n = strict_merged['eid'].nunique()

# 宽松配对：任意visit EID取交集
loose_eids = set(cmr_all['eid']) & set(fund_all['eid'])
loose_n = len(loose_eids)

print(f'严格配对唯一EID数（含所有instance）: {strict_n}  【更正：所有instance, merge on eid+instance】')
print(f'宽松配对唯一EID数:                  {loose_n}  （已符合）')

# Q3 更正：对每个EID，两两求差取绝对值，再取最小值
print('\n计算宽松配对时间间隔（两两最小绝对值）——可能需要约1分钟...')

# 预处理：每个EID的所有日期列表
cmr_date_series  = cmr_all[cmr_all['eid'].isin(loose_eids)].copy()
fund_date_series = fund_all[fund_all['eid'].isin(loose_eids)].copy()

cmr_date_series['dt']  = pd.to_datetime(cmr_date_series['Date of attending assessment centre'],  errors='coerce')
fund_date_series['dt'] = pd.to_datetime(fund_date_series['Date of attending assessment centre'], errors='coerce')

cmr_dates_by_eid  = cmr_date_series.dropna(subset=['dt']).groupby('eid')['dt'].apply(list).to_dict()
fund_dates_by_eid = fund_date_series.dropna(subset=['dt']).groupby('eid')['dt'].apply(list).to_dict()

valid_eids = set(cmr_dates_by_eid.keys()) & set(fund_dates_by_eid.keys())
gaps_years = []
for eid in valid_eids:
    fd = np.array([d.toordinal() for d in fund_dates_by_eid[eid]])
    cd = np.array([d.toordinal() for d in cmr_dates_by_eid[eid]])
    delta_days = np.abs(fd[:, None] - cd[None, :])   # 两两差绝对值矩阵
    gaps_years.append(delta_days.min() / 365.25)      # 取最近配对

gap = pd.Series(gaps_years)
le2  = (gap <= 2).sum()
y2_5 = ((gap > 2) & (gap <= 5)).sum()
gt5  = (gap > 5).sum()
tot  = len(gap)

print(f'有效配对n={tot}')
print(f'中位数={gap.median():.2f}年  P25={gap.quantile(0.25):.2f}  '
      f'P75={gap.quantile(0.75):.2f}  最小={gap.min():.2f}  最大={gap.max():.2f}')
print(f'间隔 ≤2年:  {pct(le2,  tot)}')
print(f'间隔 2-5年: {pct(y2_5, tot)}')
print(f'间隔 >5年:  {pct(gt5,  tot)}')

# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表3  分类任务标签统计')
print(SEP)

tasks_cls = [
    ('composite_ischemic_hd',      '缺血性心脏病(复合)'),
    ('prevalent_I21',              '心肌梗死 I21'),
    ('composite_cardiomyopathy_hf','心肌病/心衰(复合)'),
    ('composite_af_arrhythmia',    '房颤/心律失常(复合)'),
    ('prevalent_I25',              '慢性缺血性心脏病 I25'),
    ('prevalent_I50',              '心力衰竭 I50'),
    ('composite_hypertensive_hd',  '高血压性心脏病(复合)'),
]
pops3 = {
    'CMR人群(inst=2)': cmr_pop,
    '眼底人群':        fund_pop,
    '配对人群(2892)':  dual_pop,
}
fmt_hdr = f"{'任务':<30} {'CMR人群':^30} {'眼底人群':^30} {'配对人群':^30}"
print(fmt_hdr)
print('─' * 124)
for col, label in tasks_cls:
    row = f'{label:<30} '
    for pname, pop in pops3.items():
        if col not in pop.columns:
            row += f"{'无此列':^30} "
            continue
        s   = pop[col].dropna()
        pos = (s == 1).sum()
        row += f"{pct(pos, len(s)):<30} "
    print(row)
print('（表3已符合原口径，无需更正）')

# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表4  回归任务标签统计（含口径A和口径B）')
print(SEP)

# 口径A：眼底人群(唯一EID) left join CMR(inst=2去重)，key=eid
fund_joinA = fund_pop[['eid']].merge(
    cmr_pop[['eid', 'LV ejection fraction', 'LV end diastolic volume', 'LV end systolic volume']],
    on='eid', how='left')

# 口径B：眼底人群 按(eid, instance) left join CMR(inst=2), key=(eid, instance)
# 眼底人群中有instance=2的行（对应CMR imaging visit）
fund_inst2_rows = fund_all[fund_all['instance'].isin([2, 3])].drop_duplicates(['eid', 'instance'])
cmr_cols = ['eid', 'instance', 'LV ejection fraction', 'LV end diastolic volume', 'LV end systolic volume']
cmr_inst23 = cmr_all[cmr_all['instance'].isin([2, 3])].drop_duplicates(['eid', 'instance'])[cmr_cols]
fund_joinB = fund_inst2_rows[['eid', 'instance']].merge(cmr_inst23, on=['eid', 'instance'], how='inner')

reg_tasks = [
    ('LV ejection fraction',    'LVEF (%)'),
    ('LV end diastolic volume', 'LVEDV (mL)'),
    ('LV end systolic volume',  'LVESV (mL)'),
]
pops4 = [
    ('CMR人群(inst=2)',     cmr_pop,    None),
    ('眼底-口径A(跨visit)', fund_joinA, 'left join by eid'),
    ('眼底-口径B(同visit)', fund_joinB, 'inner join by eid+instance'),
    ('配对人群',            dual_pop,   None),
]
for col, label in reg_tasks:
    print(f'\n── {label} ──')
    for pname, pop, note in pops4:
        if col not in pop.columns:
            print(f'  {pname}: 无此列')
        else:
            uid = pop[pop[col].notna()]['eid'].nunique()
            print(f'  {pname} (唯一EID={uid}): {num_stat(pop[col])}')

# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表5  人口学基本特征')
print(SEP)

age_col = 'Age when attended assessment centre'
sex_col = 'Sex'
bmi_col = 'Body mass index (BMI)'
pops5 = [
    ('CMR人群(inst=2)', cmr_pop),
    ('眼底人群',        fund_pop),
    ('配对人群',        dual_pop),
]
for pname, pop in pops5:
    age  = pop[age_col].dropna()
    male = (pop[sex_col] == 'Male').sum()
    fem  = (pop[sex_col] == 'Female').sum()
    bmi  = pop[bmi_col].dropna()
    print(f'{pname} (n={len(pop)}):')
    print(f'  年龄: {age.mean():.1f}±{age.std():.1f}  中位数={age.median():.1f}'
          f' (P25={age.quantile(0.25):.1f}, P75={age.quantile(0.75):.1f})')
    print(f'  男性: {male} ({100*male/len(pop):.1f}%)  女性: {fem} ({100*fem/len(pop):.1f}%)')
    print(f'  BMI: {bmi.mean():.1f}±{bmi.std():.1f}')
print('（表5已符合口径，无需更正）')

# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表6  眼底→心脏结构回归可用人数（口径A / 口径B）')
print(SEP)

for col, label in [('LV ejection fraction','LVEF'),
                   ('LV end diastolic volume','LVEDV'),
                   ('LV end systolic volume','LVESV')]:
    nnA = fund_joinA[fund_joinA[col].notna()]['eid'].nunique()
    nnB = fund_joinB[fund_joinB[col].notna()]['eid'].nunique()
    print(f'{label}: 口径A(跨visit)={nnA}   口径B(同visit)={nnB}')

# 同时非空
bothA = fund_joinA[
    fund_joinA['LV ejection fraction'].notna() &
    fund_joinA['LV end diastolic volume'].notna()]['eid'].nunique()
bothB = fund_joinB[
    fund_joinB['LV ejection fraction'].notna() &
    fund_joinB['LV end diastolic volume'].notna()]['eid'].nunique()
print(f'LVEF+LVEDV同时非空: 口径A={bothA}   口径B={bothB}')

# 与严格配对人群（eid+instance）的重叠
strict_eids_set = set(strict_merged['eid'])
overlapA = fund_joinA[fund_joinA['LV ejection fraction'].notna() &
                      fund_joinA['eid'].isin(strict_eids_set)]['eid'].nunique()
overlapB = fund_joinB[fund_joinB['LV ejection fraction'].notna() &
                      fund_joinB['eid'].isin(strict_eids_set)]['eid'].nunique()
print(f'与严格配对人群重叠: 口径A={overlapA}  口径B={overlapB}')

# ════════════════════════════════════════════════════════════════════
print()
print(SEP)
print('表7  CMR序列可用性')
print(SEP)

lax_n   = cmr_pop.loc[cmr_pop[COL_LAX].map(has_zip), 'eid'].nunique()
sax_n   = cmr_pop.loc[cmr_pop[COL_SAX].map(has_zip), 'eid'].nunique()
both_n  = cmr_pop.loc[cmr_pop[COL_LAX].map(has_zip) & cmr_pop[COL_SAX].map(has_zip), 'eid'].nunique()
print(f'有LAX唯一EID数:       {lax_n}    【has_zip过滤】')
print(f'有SAX唯一EID数:       {sax_n}    【has_zip过滤】')
print(f'同时有LAX+SAX:        {both_n}   【has_zip过滤】')
print(f'T1 mapping (20213):  字段未纳入当前表，需从原始UKB数据提取')

# npy shape 检查（全量）
if os.path.exists(CMR_NPY_DIR):
    npy_files = [f for f in os.listdir(CMR_NPY_DIR) if f.endswith('.npy')]
    print(f'npy文件总数: {len(npy_files)}')
    shape_counter = {}
    bad = []
    for fn in npy_files[:300]:
        try:
            sh = tuple(np.load(os.path.join(CMR_NPY_DIR, fn), mmap_mode='r').shape)
            shape_counter[sh] = shape_counter.get(sh, 0) + 1
        except Exception as e:
            bad.append((fn, str(e)))
    for sh, cnt in sorted(shape_counter.items(), key=lambda x: -x[1]):
        tag = '正常' if sh == (4, 224, 224) else '异常!'
        print(f'  shape {sh}: {cnt}/300采样 [{tag}]')
    if bad:
        print(f'  加载异常: {bad[:3]}')
    std_rate = shape_counter.get((4, 224, 224), 0) / min(300, len(npy_files))
    print(f'  估算全量标准shape数: ~{int(std_rate * len(npy_files))} / {len(npy_files)}')

print()
print(SEP)
print('统计完成 v2')
print(SEP)
