"""
calc_norm_params.py

从 CMR 下采样训练集计算 LVEF / LVEDV / LVESV 的归一化参数。
必须只用训练集（不能包含验证/测试集），避免数据泄漏。

输出：
  norm_params.json

运行：
  /data/home/shujia/miniconda3/bin/python calc_norm_params.py
"""

import pathlib
import json
import pandas as pd

HERE = pathlib.Path(__file__).parent

REG_COLS = {
    'lvef':  'LV ejection fraction',
    'lvedv': 'LV end diastolic volume',
    'lvesv': 'LV end systolic volume',
}


def calc_norm_params(csv_path, out_path):
    df = pd.read_csv(csv_path, low_memory=False)
    print(f'读入 {csv_path.name}，共 {len(df):,} 行')

    norm_params = {}
    for key, col in REG_COLS.items():
        if col not in df.columns:
            print(f'  {col}: 列不存在，跳过')
            continue
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        norm_params[key] = {
            'mean': round(float(vals.mean()), 4),
            'std':  round(float(vals.std()),  4),
            'min':  round(float(vals.min()),  4),
            'max':  round(float(vals.max()),  4),
            'n':    int(vals.count()),
        }
        p = norm_params[key]
        print(f'  {key}: mean={p["mean"]:.2f}  std={p["std"]:.2f}  '
              f'min={p["min"]:.1f}  max={p["max"]:.1f}  n={p["n"]:,}')

    with open(out_path, 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f'  -> 已保存 {out_path}')
    return norm_params


print('=' * 60)
print('计算归一化参数（仅用 CMR 训练集）')
print('=' * 60)

train_csv = HERE / 'stage1_cmr_sampled_train.csv'
if not train_csv.exists():
    raise FileNotFoundError(f'{train_csv} 不存在，请先运行 build_training_csv.py')

calc_norm_params(train_csv, HERE / 'norm_params.json')

print('\n[DONE] calc_norm_params.py 完成')
