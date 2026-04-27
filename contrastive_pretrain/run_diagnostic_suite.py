#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动串联 EXPERIMENT_ROADMAP 中的低成本诊断，并写出 DIAGNOSTIC_REPORT.md。

用法：
  cd RETFound_MAE-main
  conda run -n retfound --no-capture-output python contrastive_pretrain/run_diagnostic_suite.py

依赖：output_dir/feature_cache/*.pt 已存在。
"""
from __future__ import print_function
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)

REPORT_PATH = os.path.join(ROOT, 'contrastive_pretrain', 'DIAGNOSTIC_REPORT.md')
OUT_JSON = os.path.join(ROOT, 'output_dir', 'diagnostic_suite_results.json')
FEATURE_CACHE = os.path.join(ROOT, 'output_dir', 'feature_cache')
CMR_CSV = os.path.join(ROOT, 'contrastive_pretrain', 'preprocessed_data', 'cmr_table.csv')
FUNDUS_CSV = os.path.join(ROOT, 'contrastive_pretrain', 'preprocessed_data', 'fundus_table.csv')
PC_COLS = 'M1_PC1,M1_PC2,M2_PC1,M2_PC2,M2_PC3,M3_PC1,M3_PC2,M4_PC1,M4_PC2,M5_PC1,M5_PC2,M6_PC1,M6_PC2,M6_PC3'
SIGMA = 6.5893
FINETUNE = os.path.join(ROOT, 'RETFound_cfp_weights.pth')


def run_cmd(cmd, env=None):
    t0 = time.time()
    p = subprocess.run(
        cmd, shell=True, cwd=ROOT,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        env=env or os.environ.copy(), universal_newlines=True)
    elapsed = time.time() - t0
    return p.returncode, p.stdout, elapsed


def section(title):
    return '\n## %s\n\n' % title


def baseline_embedding_metrics():
    """随机初始化的 proj+CMR encoder，在 val 上的 GT–pred 相关（上界对照：未训练）。"""
    import collections
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import collections.abc
    if 'torch._six' not in sys.modules:
        class _TorchSix:
            container_abcs = collections.abc
            inf = float('inf')
        sys.modules['torch._six'] = _TorchSix()
    import torch as T
    from contrastive_pretrain.models_contrast import ProjectionHead, CMREncoder
    from contrastive_pretrain.datasets_contrast import CMRBank
    from contrastive_pretrain.fast_contrast_train import run_full_eval_fast

    device = 'cuda:0' if T.cuda.is_available() else 'cpu'
    train_cache = T.load(os.path.join(FEATURE_CACHE, 'train_feats_full.pt'))
    val_cache = T.load(os.path.join(FEATURE_CACHE, 'val_feats_full.pt'))
    val_feats = val_cache['feats'].float()
    val_eids = [int(e) if hasattr(e, 'item') else e for e in val_cache['eids']]
    val_pc = val_cache['pc'].float()
    pc_cols = [c.strip() for c in PC_COLS.split(',')]
    cmr_val = CMRBank(CMR_CSV, pc_cols, split='val', device=device, max_rows=0)
    proj = ProjectionHead(1024, 512, 256).to(device)
    cmr_enc = CMREncoder(len(pc_cols), 128, 256).to(device)
    proj.eval()
    cmr_enc.eval()
    m = run_full_eval_fast(proj, cmr_enc, val_feats, val_eids, val_pc, cmr_val, device, SIGMA)
    return m


def main():
    results = {
        'started_at': datetime.now().isoformat(),
        'steps': [],
    }
    lines = ['# 自动诊断报告', '']
    lines.append('生成时间: %s' % results['started_at'])
    lines.append('')

    # --- 1) diagnostics_contrast: eid + sgt + sigma (no heavy grad/overfit) ---
    lines.append('## 1. 数据与 S_GT（diagnostics_contrast）\n')
    cmd = (
        'conda run -n retfound --no-capture-output python contrastive_pretrain/diagnostics_contrast.py '
        '--fundus_csv "%s" --cmr_csv "%s" --pc_cols "%s" --sigma %s --finetune "%s" '
        '--cmr_train_max_rows 0 --checks eid sgt sigma'
        % (FUNDUS_CSV, CMR_CSV, PC_COLS, SIGMA, FINETUNE)
    )
    code, out, elapsed = run_cmd(cmd)
    results['steps'].append({'name': 'diagnostics_eid_sgt_sigma', 'exit': code, 'sec': elapsed, 'log_tail': out[-4000:]})
    lines.append('```')
    lines.append(out[-8000:] if len(out) > 8000 else out)
    lines.append('```\n')

    # --- 2) 回归基线（冻结 CLS → PC）---
    lines.append('## 2. 冻结特征 MLP 回归 PC（Pearson）\n')
    cmd = 'conda run -n retfound --no-capture-output python contrastive_pretrain/regression_baseline.py'
    code, out, elapsed = run_cmd(cmd)
    results['steps'].append({'name': 'regression_baseline', 'exit': code, 'sec': elapsed})
    lines.append('```')
    lines.append(out[-6000:] if len(out) > 6000 else out)
    lines.append('```\n')

    # --- 3) 随机嵌入基线：GT–pred Spearman ---
    lines.append('## 3. 随机初始化头（未训练）验证集 gt_pred_spearman / Pearson\n')
    try:
        m = baseline_embedding_metrics()
        results['steps'].append({'name': 'random_init_metrics', 'metrics': m})
        lines.append('```json')
        lines.append(json.dumps(m, indent=2))
        lines.append('```\n')
    except Exception as e:
        results['steps'].append({'name': 'random_init_metrics', 'error': str(e)})
        lines.append('失败: %s\n' % e)

    # --- 4) 学习曲线：train_eid_frac ---
    lines.append('## 4. 学习曲线（fast_contrast_train，linear_proj，100 epoch）\n')
    for frac in (0.25, 0.5, 1.0):
        out_dir = os.path.join(ROOT, 'output_dir', 'diag_lc_%s' % str(frac).replace('.', 'p'))
        cmd = (
            'conda run -n retfound --no-capture-output python contrastive_pretrain/fast_contrast_train.py '
            '--train_feat output_dir/feature_cache/train_feats_full.pt '
            '--val_feat output_dir/feature_cache/val_feats_full.pt '
            '--cmr_csv contrastive_pretrain/preprocessed_data/cmr_table.csv '
            '--output_dir "%s" --epochs 100 --eval_freq 100 --batch_size 128 '
            '--lr 1e-3 --temperature 0.15 --cmr_sample_k 256 --train_eid_frac %s '
            '--linear_proj --sigma %s --gpu 0'
            % (out_dir, frac, SIGMA)
        )
        code, out, elapsed = run_cmd(cmd)
        last = {}
        log_path = os.path.join(out_dir, 'log.json')
        if os.path.isfile(log_path):
            with open(log_path) as f:
                log = json.load(f)
                if log:
                    last = log[-1]
        results['steps'].append({
            'name': 'learning_curve_frac_%s' % frac,
            'exit': code, 'sec': elapsed, 'last_metrics': last,
        })
        lines.append('### train_eid_frac = %s\n' % frac)
        lines.append('- 用时: %.1fs, exit=%s' % (elapsed, code))
        lines.append('- 末 epoch 指标:')
        lines.append('```json')
        lines.append(json.dumps(last, indent=2))
        lines.append('```\n')

    # --- 5) 温度对比 ---
    lines.append('## 5. 温度对比（linear_proj，train_eid_frac=1，120 epoch）\n')
    for temp in (0.07, 0.2):
        out_dir = os.path.join(ROOT, 'output_dir', 'diag_temp_%s' % str(temp).replace('.', 'p'))
        cmd = (
            'conda run -n retfound --no-capture-output python contrastive_pretrain/fast_contrast_train.py '
            '--train_feat output_dir/feature_cache/train_feats_full.pt '
            '--val_feat output_dir/feature_cache/val_feats_full.pt '
            '--cmr_csv contrastive_pretrain/preprocessed_data/cmr_table.csv '
            '--output_dir "%s" --epochs 120 --eval_freq 120 --batch_size 128 '
            '--lr 1e-3 --temperature %s --cmr_sample_k 256 --train_eid_frac 1.0 '
            '--linear_proj --sigma %s --gpu 0'
            % (out_dir, temp, SIGMA)
        )
        code, out, elapsed = run_cmd(cmd)
        last = {}
        log_path = os.path.join(out_dir, 'log.json')
        if os.path.isfile(log_path):
            with open(log_path) as f:
                log = json.load(f)
                if log:
                    last = log[-1]
        results['steps'].append({
            'name': 'temperature_%s' % temp, 'exit': code, 'sec': elapsed, 'last_metrics': last,
        })
        lines.append('### temperature = %s\n' % temp)
        lines.append('```json')
        lines.append(json.dumps(last, indent=2))
        lines.append('```\n')

    results['finished_at'] = datetime.now().isoformat()
    lines.append('## 6. 总结（自动归纳）\n')

    # Auto summary bullets
    lc_gt = []
    for s in results['steps']:
        if 'learning_curve' in s.get('name', '') and s.get('last_metrics'):
            lm = s['last_metrics']
            lc_gt.append((s['name'], lm.get('gt_pred_spearman'), lm.get('gt_pred_pearson'), lm.get('paired_cosine'), lm.get('R@5')))

    lines.append('### 学习曲线末轮 gt_pred_spearman / Pearson（越高越好）\n')
    for name, sp, pe, pc, r5 in lc_gt:
        lines.append('- **%s**: Spearman=%s, Pearson=%s, paired_cosine=%s, R@5=%s' % (name, sp, pe, pc, r5))

    lines.append('\n### 解读提示\n')
    lines.append('- **gt_pred_spearman**：跨人「fundus·CMR 余弦」与 PC 高斯 GT 的秩相关；比 R@k 更贴近 soft 目标。')
    lines.append('- 若随 **train_eid_frac** 增大而 Spearman **单调升**，扩大样本大概率有效。')
    lines.append('- 若随机初始化 Spearman 已接近训练后，说明 **嵌入空间尚未学到表型结构** 或信号极弱。')
    lines.append('- 冻结特征回归 val Pearson≈0 且 gt_pred 仍低 → **瓶颈多在表征/信号**，非仅调参。')
    lines.append('')

    with open(REPORT_PATH, 'w') as f:
        f.write('\n'.join(lines))
    with open(OUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)

    print('Wrote', REPORT_PATH)
    print('Wrote', OUT_JSON)


if __name__ == '__main__':
    main()
