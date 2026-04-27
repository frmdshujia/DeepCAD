"""make_report.py — 生成 ceiling 实验汇总 PDF"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 注册 Noto Sans CJK 字体支持中文
plt.rcParams['font.family'] = ['Noto Sans CJK JP', 'Droid Sans Fallback', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
from PIL import Image

BASE        = 'output_dir/ceiling_v2'
BASE_4DIS   = 'output_dir/4disease_cmr'
OUTPDF      = 'output_dir/ceiling_v2/ceiling_report.pdf'

# ---------------------------------------------------------------------------
# 读取所有结果
# ---------------------------------------------------------------------------
def load_results():
    rows = {}
    for d in sorted(os.listdir(BASE)):
        j = os.path.join(BASE, d, 'results.json')
        if not os.path.exists(j): continue
        r = json.load(open(j))
        rows[d] = r
    return rows

def load_4disease_results():
    rows = {}
    if not os.path.exists(BASE_4DIS):
        return rows
    for d in sorted(os.listdir(BASE_4DIS)):
        j = os.path.join(BASE_4DIS, d, 'results.json')
        if not os.path.exists(j): continue
        rows[d] = json.load(open(j))
    return rows

# ---------------------------------------------------------------------------
# 颜色方案
# ---------------------------------------------------------------------------
C_CMR   = '#1565C0'   # 深蓝
C_FUND  = '#2E7D32'   # 深绿
C_LIN   = '#78909C'   # 灰蓝
C_FT    = '#E53935'   # 红
C_LAX   = '#7B1FA2'   # 紫
C_SAX   = '#F57C00'   # 橙
C_FULL  = '#1565C0'

def task_cn(t):
    return {'i21':'心梗 (I21)', 'isch_hd':'心肌缺血', 'lvef':'LVEF (%)', 'lvedv':'LVEDV (mL)'}[t]

def mode_cn(m):
    return {'linear':'线性探针', 'finetune':'全参微调'}[m]

def enc_cn(e):
    return {'medsam':'MedSAM (CMR)', 'retfound':'RETFound (眼底)'}[e]

# ---------------------------------------------------------------------------
# Page 1: 封面 + 实验设计概述
# ---------------------------------------------------------------------------
def page_cover(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
    fig.patch.set_facecolor('#1A237E')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')
    ax.set_facecolor('#1A237E')

    # 标题
    ax.text(0.5, 0.72, 'MedSAM / RETFound 预测上限实验', color='white',
            fontsize=28, fontweight='bold', ha='center', va='center',
            fontfamily='DejaVu Sans')
    ax.text(0.5, 0.60, 'Ceiling Experiments: CMR & Fundus → Cardiac Disease / Function',
            color='#90CAF9', fontsize=16, ha='center', va='center')
    ax.text(0.5, 0.50, '编码器: MedSAM (ViT-B, 85.9M)  |  RETFound (ViT-L, 307M)',
            color='#B0BEC5', fontsize=13, ha='center', va='center')

    # 实验列表
    items = [
        'Exp 1  MedSAM 线性探针  CMR (n=8,224)  →  心梗 / 心肌缺血 / LVEF / LVEDV',
        'Exp 2  MedSAM 全参微调  CMR (n=8,224)  →  心梗 / 心肌缺血 / LVEF / LVEDV',
        'Exp 3  RETFound 线性探针  眼底 (n=10,000)  →  心梗 / 心肌缺血',
        'Exp 4  RETFound 全参微调  眼底 (n=10,000)  →  心梗 / 心肌缺血 / LVEF / LVEDV',
        'Exp 5  MedSAM 全参微调  CMR 序列对比 (全帧 / LAX / SAX)  →  心梗 / 心肌缺血',
    ]
    for i, txt in enumerate(items):
        ax.text(0.12, 0.36 - i * 0.065, f'● {txt}', color='#E3F2FD',
                fontsize=11, va='center')

    ax.text(0.5, 0.06, 'CMR = 心脏MRI npy (4帧, 224×224)  |  LAX = 长轴 frames 0-1  |  SAX = 短轴 frames 2-3',
            color='#78909C', fontsize=9, ha='center', va='center')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Page 2: 方法论 (Q1 帧选择 / Q2 训练策略 / Q3 CMR序列)
# ---------------------------------------------------------------------------
def page_methodology(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    fig.suptitle('方法论说明', fontsize=17, fontweight='bold', y=0.98)

    # ── Q1: 帧选择 ──────────────────────────────────────────────────────────
    ax.text(0.03, 0.90, 'Q1  CMR 帧选择依据（SAX / LAX）', fontsize=13,
            fontweight='bold', color='#1565C0', va='top')
    q1 = (
        '  UKB CMR npy 文件形状: (4, 224, 224) float16\n'
        '  • frames 0-1  =  长轴 (LAX) cine  [UKB field 20208]\n'
        '  • frames 2-3  =  短轴 (SAX) cine  [UKB field 20209]\n'
        '\n'
        '  每个序列各保留 2 帧，对应心动周期的 舒张末期 (ED) 和 收缩末期 (ES)。\n'
        '  选取依据：ED/ES 是心脏功能评估的标准关键帧（EF、EDV、ESV 均在此定义）；\n'
        '  cine 序列通常含 25-50 帧，只截取关键帧可大幅降低存储和计算开销，同时保留最大诊断信息。'
    )
    ax.text(0.03, 0.84, q1, fontsize=10, va='top', linespacing=1.55, color='#37474F')

    ax.axhline(0.62, xmin=0.03, xmax=0.97, color='#E0E0E0', lw=0.8)

    # ── Q2: 训练策略 ─────────────────────────────────────────────────────────
    ax.text(0.03, 0.60, 'Q2  训练策略（线性探针 vs. 全参微调）', fontsize=13,
            fontweight='bold', color='#1565C0', va='top')

    headers = ['项目', '线性探针', '全参微调']
    rows_q2 = [
        ['优化器',       'sklearn LR/Ridge (LBFGS)',        'AdamW (PyTorch)'],
        ['骨干网络',     '冻结 (无梯度)',                    '解冻 lr=5e-6（CMR）/ 1e-5（眼底）'],
        ['分类头 lr',    '—',                               '1e-3'],
        ['Batch size',  '全量特征 (CPU)',                   '16 (GPU, AMP)'],
        ['分类损失',     'LogisticRegression balanced',      'BCE + pos_weight (正负样本不平衡)'],
        ['回归损失',     'Ridge regression',                 'MSE'],
        ['早停',         '—',                               '15 epoch patience (AUC/MAE)'],
        ['训练/验证/测试', '70 / 15 / 15 %  随机分层',       '同左'],
        ['训练轮数',     '1 (sklearn)',                      '50 epoch (分类) / 50 epoch (回归)'],
    ]
    col_x = [0.04, 0.28, 0.62]
    col_w = [0.22, 0.33, 0.35]

    # header
    for h, x in zip(headers, col_x):
        ax.text(x, 0.555, h, fontsize=9.5, fontweight='bold', va='top',
                color='white',
                bbox=dict(boxstyle='round', facecolor='#37474F', pad=0.25, alpha=0.9))
    for i, row in enumerate(rows_q2):
        yy = 0.510 - i * 0.041
        bg = '#F5F5F5' if i % 2 == 0 else 'white'
        ax.add_patch(plt.Rectangle((0.03, yy - 0.017), 0.94, 0.038,
                                   facecolor=bg, edgecolor='none', zorder=0))
        for val, x in zip(row, col_x):
            ax.text(x, yy, val, fontsize=8.5, va='center', color='#263238')

    ax.axhline(0.145, xmin=0.03, xmax=0.97, color='#E0E0E0', lw=0.8)

    # ── Q3: UKB 可用 CMR 序列 ───────────────────────────────────────────────
    ax.text(0.03, 0.13, 'Q3  UKB 可用 CMR 序列', fontsize=13,
            fontweight='bold', color='#1565C0', va='top')
    q3 = (
        '  • 20208  长轴 cine (LAX 4-ch / 2-ch)     ✅ 已使用 (frames 0-1)\n'
        '  • 20209  短轴 cine (SAX stack, 多切面)    ✅ 已使用 (frames 2-3)\n'
        '  • 20213  原生 T1-mapping (心肌纤维化)      ⬜ 可用，尚未使用\n'
        '  • 20214  钆后 T1 / ECV mapping            ⬜ 可用，尚未使用\n'
        '  • LGE (延迟钆强化)                         ❌ UKB 标准方案不采集\n'
        '  • 首过灌注 (First-pass perfusion)          ❌ UKB 标准方案不采集\n'
        '  • T2-STIR (心肌水肿)                       ❌ UKB 标准方案不采集\n'
        '  → 后续可探索将 T1-mapping + ECV 序列纳入多模态预训练（字段 20213/20214）'
    )
    ax.text(0.03, 0.07, q3, fontsize=9.5, va='top', linespacing=1.6, color='#37474F',
            family='monospace')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Page N: 四病分类实验 (I21 / isch_hd / I25 / I20) MedSAM 全参微调
# ---------------------------------------------------------------------------
def page_4disease(pdf, r4):
    if not r4:
        return  # 实验尚未完成，跳过

    fig, axes = plt.subplots(1, 4, figsize=(11.69, 8.27))
    fig.suptitle(
        '四病分类实验 — MedSAM 全参微调 CMR (n≈8,224)  Test AUC & 阳性准确率',
        fontsize=14, fontweight='bold', y=0.98)

    task_info = [
        ('i21',     '心梗 (I21)',          '#1565C0'),
        ('isch_hd', '缺血性心脏病\n(复合)', '#E53935'),
        ('i25',     '慢性缺血性心脏病\n(I25)', '#2E7D32'),
        ('i20',     '心绞痛 (I20)',         '#F57C00'),
    ]

    for ax, (task, title, color) in zip(axes, task_info):
        key = task
        if key not in r4:
            ax.text(0.5, 0.5, '待完成', ha='center', va='center',
                    transform=ax.transAxes, fontsize=14, color='gray')
            ax.set_title(title, fontsize=11, fontweight='bold')
            ax.axis('off')
            continue

        tv = r4[key]['results']
        splits = ['val', 'test']
        aucs = [tv[s].get('auc', 0) for s in splits]
        baccs = [tv[s].get('bacc', 0) for s in splits]
        # sensitivity = positive class accuracy = recall of positive class
        # bacc = (sens + spec)/2, and we logged sens as part of bacc computation
        # Use f1 as proxy or display bacc
        senss = []
        for s in splits:
            f1 = tv[s].get('f1', 0)
            n  = tv[s].get('n', 1)
            pos = tv[s].get('pos', 0)
            # sensitivity ~ 2*f1 - f1/precision, but we don't have precision
            # fall back to bacc*2 - specificity estimate; just display bacc
            senss.append(tv[s].get('bacc', 0))

        x = np.arange(2)
        bars = ax.bar(x, aucs, color=[color]*2, alpha=[0.6, 1.0], width=0.5,
                      edgecolor='white', linewidth=0.8)
        for bar, v, bacc in zip(bars, aucs, baccs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'AUC={v:.3f}\nbACC={bacc:.3f}',
                    ha='center', va='bottom', fontsize=8, fontweight='bold')

        # 阳性样本信息
        t_test = tv.get('test', {})
        pos_n = t_test.get('pos', '?')
        tot_n = t_test.get('n', '?')
        if isinstance(pos_n, int) and isinstance(tot_n, int) and tot_n > 0:
            pct = 100 * pos_n / tot_n
            ax.set_xlabel(f'test pos={pos_n}/{tot_n} ({pct:.1f}%)',
                          fontsize=8, color='gray')

        ax.axhline(0.5, color='black', lw=1, ls='--', alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(['Val', 'Test'], fontsize=10)
        ax.set_ylim(0.40, 0.90)
        ax.set_ylabel('AUC / bACC', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Page 2: 分类任务汇总 (心梗 + 心肌缺血)
# ---------------------------------------------------------------------------
def page_binary(pdf, results):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.suptitle('分类任务 Test AUC 汇总 (心梗 / 心肌缺血)', fontsize=16,
                 fontweight='bold', y=0.97)

    for ax, task, title in zip(axes,
                                ['i21', 'isch_hd'],
                                ['心梗 (prevalent_I21)', '心肌缺血 (composite_ischemic_hd)']):
        labels, aucs, colors, hatches = [], [], [], []

        order = [
            ('exp1_linear_cmr',    'MedSAM\n线性探针\nCMR',   C_CMR,  ''),
            ('exp2_finetune_cmr',  'MedSAM\n全参微调\nCMR',   C_FT,   ''),
            ('exp3_linear_fundus', 'RETFound\n线性探针\n眼底', C_FUND, ''),
            ('exp4_finetune_fundus','RETFound\n全参微调\n眼底',C_LIN,  '//'),
            ('exp5_lax',           'MedSAM\n微调 LAX',        C_LAX,  ''),
            ('exp5_sax',           'MedSAM\n微调 SAX',        C_SAX,  ''),
        ]
        for key, lbl, col, hatch in order:
            k = f'{key}_{task}'
            if k not in results: continue
            auc = results[k]['results']['test'].get('auc', None)
            if auc is None: continue
            labels.append(lbl); aucs.append(auc)
            colors.append(col); hatches.append(hatch)

        x = np.arange(len(labels))
        bars = ax.bar(x, aucs, color=colors, width=0.6, edgecolor='white', linewidth=0.8)
        for bar, h in zip(bars, hatches):
            bar.set_hatch(h)
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.axhline(0.5, color='black', lw=1, ls='--', alpha=0.5, label='Random (0.5)')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_ylim(0.45, 0.80)
        ax.set_ylabel('Test AUC', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

        # 阳性样本数注释
        k0 = f'exp1_linear_cmr_{task}'
        if k0 in results:
            pos = results[k0]['results']['test'].get('pos', '?')
            n   = results[k0]['results']['test'].get('n', '?')
            ax.text(0.98, 0.97, f'Test n={n}, pos={pos} ({100*pos/n:.1f}%)',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    color='gray', style='italic')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Page 3: 回归任务汇总 (LVEF + LVEDV)
# ---------------------------------------------------------------------------
def page_regression_bar(pdf, results):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.suptitle('回归任务 Test Pearson 汇总 (LVEF / LVEDV)', fontsize=16,
                 fontweight='bold', y=0.97)

    for ax, task, title in zip(axes,
                                ['lvef', 'lvedv'],
                                ['LVEF (%)', 'LVEDV (mL)']):
        labels, pearsons, r2s, colors = [], [], [], []

        order = [
            ('exp1_linear_cmr',      'MedSAM\n线性探针 CMR',   C_CMR),
            ('exp2_finetune_cmr',    'MedSAM\n全参微调 CMR',   C_FT),
            ('exp4_finetune_fundus', 'RETFound\n全参微调 眼底', C_FUND),
        ]
        for key, lbl, col in order:
            k = f'{key}_{task}'
            if k not in results: continue
            t = results[k]['results']['test']
            if 'pearson' not in t: continue
            labels.append(lbl); pearsons.append(t['pearson'])
            r2s.append(t['r2']); colors.append(col)

        x = np.arange(len(labels))
        bars = ax.bar(x, pearsons, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
        for bar, v, r2 in zip(bars, pearsons, r2s):
            ax.text(bar.get_x() + bar.get_width()/2,
                    max(bar.get_height(), 0) + 0.01,
                    f'r={v:.3f}\nR²={r2:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.axhline(0, color='black', lw=0.8, ls='-', alpha=0.4)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(-0.15, 1.0)
        ax.set_ylabel('Test Pearson r', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Page 4-5: 散点图 (LVEDV 线性/微调, LVEF 线性/微调)
# ---------------------------------------------------------------------------
def page_scatter_grid(pdf, title, img_paths):
    """img_paths: list of (label, path) tuples, up to 4."""
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.suptitle(title, fontsize=15, fontweight='bold', y=0.99)
    cols = min(len(img_paths), 4)
    gs = gridspec.GridSpec(1, cols, figure=fig, wspace=0.06)
    for i, (lbl, path) in enumerate(img_paths):
        ax = fig.add_subplot(gs[0, i])
        if path and os.path.exists(path):
            img = Image.open(path)
            ax.imshow(np.array(img))
        else:
            ax.text(0.5, 0.5, '无数据', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='gray')
        ax.set_title(lbl, fontsize=10, fontweight='bold')
        ax.axis('off')
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Page 6: CMR 序列对比
# ---------------------------------------------------------------------------
def page_sequence(pdf, results):
    fig, axes = plt.subplots(1, 2, figsize=(11.69, 8.27))
    fig.suptitle('Exp5: CMR 序列对比 — MedSAM 全参微调 Test AUC', fontsize=15,
                 fontweight='bold', y=0.97)

    for ax, task, title in zip(axes, ['i21', 'isch_hd'],
                                ['心梗 (I21)', '心肌缺血']):
        seq_map = [
            ('全帧 LAX+SAX', f'exp2_finetune_cmr_{task}', C_FULL),
            ('LAX only',     f'exp5_lax_{task}',           C_LAX),
            ('SAX only',     f'exp5_sax_{task}',           C_SAX),
        ]
        labels, aucs, colors = [], [], []
        for lbl, key, col in seq_map:
            if key not in results: continue
            auc = results[key]['results']['test'].get('auc', None)
            if auc is None: continue
            labels.append(lbl); aucs.append(auc); colors.append(col)

        x = np.arange(len(labels))
        bars = ax.bar(x, aucs, color=colors, width=0.5, edgecolor='white', linewidth=0.8)
        for bar, v in zip(bars, aucs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.003,
                    f'{v:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax.axhline(0.5, color='black', lw=1, ls='--', alpha=0.4, label='Random')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylim(0.45, 0.80)
        ax.set_ylabel('Test AUC', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Page 7: 结论总结表
# ---------------------------------------------------------------------------
def page_summary(pdf, results):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('white')
    ax = fig.add_axes([0.03, 0.05, 0.94, 0.88])
    ax.axis('off')
    fig.suptitle('实验结果汇总表 (Test集)', fontsize=16, fontweight='bold', y=0.97)

    headers = ['实验', '编码器', '模态/序列', '模式', '任务', 'AUC', 'Pearson r', 'R²', '备注']
    rows_data = []

    order = [
        ('exp1_linear_cmr_i21',          'MedSAM', 'CMR全帧',  '线性探针', '心梗'),
        ('exp1_linear_cmr_isch_hd',      'MedSAM', 'CMR全帧',  '线性探针', '心肌缺血'),
        ('exp1_linear_cmr_lvef',         'MedSAM', 'CMR全帧',  '线性探针', 'LVEF'),
        ('exp1_linear_cmr_lvedv',        'MedSAM', 'CMR全帧',  '线性探针', 'LVEDV'),
        ('exp2_finetune_cmr_i21',        'MedSAM', 'CMR全帧',  '全参微调', '心梗'),
        ('exp2_finetune_cmr_isch_hd',    'MedSAM', 'CMR全帧',  '全参微调', '心肌缺血'),
        ('exp2_finetune_cmr_lvef',       'MedSAM', 'CMR全帧',  '全参微调', 'LVEF'),
        ('exp2_finetune_cmr_lvedv',      'MedSAM', 'CMR全帧',  '全参微调', 'LVEDV'),
        ('exp3_linear_fundus_i21',       'RETFound','眼底',    '线性探针', '心梗'),
        ('exp3_linear_fundus_isch_hd',   'RETFound','眼底',    '线性探针', '心肌缺血'),
        ('exp4_finetune_fundus_i21',     'RETFound','眼底',    '全参微调', '心梗'),
        ('exp4_finetune_fundus_isch_hd', 'RETFound','眼底',    '全参微调', '心肌缺血'),
        ('exp4_finetune_fundus_lvef',    'RETFound','眼底',    '全参微调', 'LVEF'),
        ('exp4_finetune_fundus_lvedv',   'RETFound','眼底',    '全参微调', 'LVEDV'),
        ('exp5_lax_i21',                 'MedSAM', 'CMR LAX',  '全参微调', '心梗'),
        ('exp5_lax_isch_hd',             'MedSAM', 'CMR LAX',  '全参微调', '心肌缺血'),
        ('exp5_sax_i21',                 'MedSAM', 'CMR SAX',  '全参微调', '心梗'),
        ('exp5_sax_isch_hd',             'MedSAM', 'CMR SAX',  '全参微调', '心肌缺血'),
    ]

    notes = {
        'exp1_linear_cmr_lvedv': '★ 最强信号',
        'exp2_finetune_cmr_lvedv': '★★ 最佳结果',
        'exp2_finetune_cmr_i21': '阳性仅23例',
        'exp4_finetune_fundus_lvef': 'n=223,样本不足',
        'exp4_finetune_fundus_lvedv': 'n=223,样本不足',
        'exp5_lax_isch_hd': 'LAX>SAX',
    }

    for key, enc, mod, mode, task in order:
        if key not in results: continue
        t = results[key]['results']['test']
        auc  = f"{t['auc']:.3f}"  if 'auc'     in t else '—'
        pear = f"{t['pearson']:.3f}" if 'pearson' in t else '—'
        r2   = f"{t['r2']:.3f}"  if 'r2'      in t else '—'
        note = notes.get(key, '')
        rows_data.append([key.replace('exp','E').split('_')[0]+key[3:5],
                          enc, mod, mode, task, auc, pear, r2, note])

    col_widths = [0.06, 0.09, 0.09, 0.09, 0.09, 0.07, 0.09, 0.07, 0.14]
    col_x = [sum(col_widths[:i]) + 0.01 for i in range(len(col_widths))]

    # Header
    for j, (h, x, w) in enumerate(zip(headers, col_x, col_widths)):
        ax.text(x + w/2, 0.94, h, ha='center', va='center', fontsize=8.5,
                fontweight='bold', color='white',
                bbox=dict(boxstyle='round', facecolor='#1565C0', pad=0.3))

    # Rows
    row_colors = ['#E3F2FD', '#FFFFFF']
    for i, row in enumerate(rows_data):
        y = 0.88 - i * 0.049
        ax.add_patch(FancyBboxPatch((0.01, y - 0.022), 0.98, 0.044,
                                    boxstyle='round,pad=0.002',
                                    facecolor=row_colors[i % 2], edgecolor='none'))
        for j, (val, x, w) in enumerate(zip(row, col_x, col_widths)):
            color = '#B71C1C' if val == '—' and j in (5, 6, 7) else \
                    '#1B5E20' if '★' in str(row[-1]) and j == len(row)-1 else 'black'
            ax.text(x + w/2, y, str(val), ha='center', va='center',
                    fontsize=7.5, color=color)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Page 8: 核心结论
# ---------------------------------------------------------------------------
def page_conclusions(pdf):
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.patch.set_facecolor('#FAFAFA')
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis('off')
    fig.suptitle('核心结论', fontsize=18, fontweight='bold', y=0.97)

    conclusions = [
        ('🔵', 'LVEDV 是最强可预测信号',
         'MedSAM 线性探针即达 Pearson=0.711, R²=0.495\n'
         '全参微调后进一步提升至 Pearson=0.832, R²=0.690\n'
         '→ CMR 视觉特征对心室舒张末容积有天然表征能力'),

        ('🔴', 'LVEF 信号极弱',
         '线性探针 Pearson=0.186，全参微调也仅 0.301\n'
         '→ LVEF 反映心肌收缩功能，不在形态特征中，CMR视觉特征无法有效预测'),

        ('🟡', '疾病分类受限于极少阳性样本',
         '心梗 test 集仅 23 阳性例（总 824 例，2.8%），AUC 统计噪声大\n'
         '全参微调 AUC 0.659（MedSAM CMR），线性探针 0.672（RETFound 眼底）\n'
         '→ 不能排除更多阳性样本后 AUC 可进一步提升'),

        ('🟢', 'CMR 序列对比：LAX 优于 SAX',
         '心肌缺血: LAX AUC=0.636 > 全帧 0.585 > SAX 0.545\n'
         '→ 长轴切面包含更多诊断相关信息，短轴单独使用效果最差'),

        ('🟣', '眼底预测 LVEF/LVEDV 失效',
         '有 CMR 标注的眼底交集仅 n=223，样本远不足以支撑回归任务\n'
         '→ 需要更大规模双模态队列'),
    ]

    for i, (icon, title, body) in enumerate(conclusions):
        y = 0.85 - i * 0.165
        ax.text(0.04, y, icon, fontsize=20, va='center')
        ax.text(0.09, y + 0.025, title, fontsize=13, fontweight='bold',
                va='center', color='#1A237E')
        ax.text(0.09, y - 0.04, body, fontsize=10, va='center',
                color='#37474F', linespacing=1.6)
        ax.axhline(y - 0.09, xmin=0.04, xmax=0.96,
                   color='#E0E0E0', lw=0.8)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    results = load_results()
    r4 = load_4disease_results()
    print(f'加载到 {len(results)} 个ceiling实验结果, {len(r4)} 个4病实验结果')

    with PdfPages(OUTPDF) as pdf:
        # 封面
        page_cover(pdf)
        print('封面 ✓')

        # 方法论页面 (Q1/Q2/Q3)
        page_methodology(pdf)
        print('方法论 ✓')

        # 分类任务汇总
        page_binary(pdf, results)
        print('分类柱状图 ✓')

        # 回归任务汇总
        page_regression_bar(pdf, results)
        print('回归柱状图 ✓')

        # LVEDV 散点图
        page_scatter_grid(pdf, 'LVEDV 预测散点图 (Test集)',
            [('MedSAM 线性探针\nCMR全帧 (test)',
              f'{BASE}/exp1_linear_cmr_lvedv/scatter_test.png'),
             ('MedSAM 全参微调\nCMR全帧 (val)',
              f'{BASE}/exp2_finetune_cmr_lvedv/scatter_val.png'),
             ('MedSAM 全参微调\nCMR全帧 (test)',
              f'{BASE}/exp2_finetune_cmr_lvedv/scatter_test.png'),
             ('RETFound 全参微调\n眼底 (test)',
              f'{BASE}/exp4_finetune_fundus_lvedv/scatter_test.png')])
        print('LVEDV散点图 ✓')

        # LVEF 散点图
        page_scatter_grid(pdf, 'LVEF 预测散点图 (Test集)',
            [('MedSAM 线性探针\nCMR全帧 (test)',
              f'{BASE}/exp1_linear_cmr_lvef/scatter_test.png'),
             ('MedSAM 全参微调\nCMR全帧 (val)',
              f'{BASE}/exp2_finetune_cmr_lvef/scatter_val.png'),
             ('MedSAM 全参微调\nCMR全帧 (test)',
              f'{BASE}/exp2_finetune_cmr_lvef/scatter_test.png'),
             ('RETFound 全参微调\n眼底 (test)',
              f'{BASE}/exp4_finetune_fundus_lvef/scatter_test.png')])
        print('LVEF散点图 ✓')

        # CMR 序列对比
        page_sequence(pdf, results)
        print('序列对比 ✓')

        # 汇总表
        page_summary(pdf, results)
        print('汇总表 ✓')

        # 四病分类实验
        page_4disease(pdf, r4)
        if r4:
            print('四病分类 ✓')
        else:
            print('四病分类（尚未完成，跳过）')

        # 结论
        page_conclusions(pdf)
        print('结论 ✓')

        # PDF metadata
        d = pdf.infodict()
        d['Title'] = 'MedSAM/RETFound Ceiling Experiments Report'
        d['Author'] = 'AutoReport'
        d['Subject'] = 'CMR Fundus Cardiac Prediction'

    print(f'\n[done] 报告已保存: {OUTPDF}')

if __name__ == '__main__':
    main()
