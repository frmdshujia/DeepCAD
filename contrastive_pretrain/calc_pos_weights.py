"""
calc_pos_weights.py

从下采样后的 sampled CSV 计算 BCEWithLogitsLoss 的 pos_weight。
输出：
  pos_weights_cmr.json
  pos_weights_fundus.json

运行：
  /data/home/shujia/miniconda3/bin/python calc_pos_weights.py
"""

import pathlib
import json
import pandas as pd

HERE = pathlib.Path(__file__).parent

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]


def calc_pos_weights(csv_path, out_path):
    df = pd.read_csv(csv_path, low_memory=False)
    weights = {}
    print(f'\n{csv_path.name} 的 pos_weight（共 {len(df):,} 行）:')
    for col in CLS_COLS:
        if col not in df.columns:
            print(f'  {col}: 列不存在，跳过')
            continue
        n_pos = (df[col] == 1).sum()
        n_neg = (df[col] == 0).sum()
        pw = round(n_neg / n_pos, 2) if n_pos > 0 else 1.0
        weights[col] = pw
        print(f'  {col:<40} pos={n_pos:,}  neg={n_neg:,}  pos_weight={pw:.2f}')

    with open(out_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f'  -> 已保存 {out_path}')
    return weights


print('=' * 60)
print('计算 pos_weight')
print('=' * 60)

cmr_csv    = HERE / 'stage1_cmr_sampled.csv'
fundus_csv = HERE / 'stage1_fundus_sampled.csv'

if not cmr_csv.exists():
    raise FileNotFoundError(f'{cmr_csv} 不存在，请先运行 build_training_csv.py')
if not fundus_csv.exists():
    raise FileNotFoundError(f'{fundus_csv} 不存在，请先运行 build_training_csv.py')

calc_pos_weights(cmr_csv,    HERE / 'pos_weights_cmr.json')
calc_pos_weights(fundus_csv, HERE / 'pos_weights_fundus.json')

print('\n[DONE] calc_pos_weights.py 完成')
