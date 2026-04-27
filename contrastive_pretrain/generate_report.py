"""
generate_report.py
Generate training plan PDF report + sample path CSV files (English version)
"""
import pickle, pathlib, pandas as pd, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyBboxPatch
import matplotlib.gridspec as gridspec
import numpy as np

HERE  = pathlib.Path(__file__).parent
ROOT  = HERE.parent
OUT   = HERE / 'task_reports'
OUT.mkdir(exist_ok=True)

# ── Load sample data ──────────────────────────────────────────────────────────
with open('/tmp/task_samples.pkl', 'rb') as f:
    S = pickle.load(f)

# ── Write sample path CSV files ───────────────────────────────────────────────
def write_csv(rows, path):
    pd.DataFrame(rows).to_csv(path, index=False)
    print(f'  Written: {path}  ({len(rows):,} rows)')

# Task 1: CMR single-tower
write_csv(S['train']['cmr'],  OUT / 'task1_cmr_train.csv')
write_csv(S['val']['cmr'],    OUT / 'task1_cmr_val.csv')
write_csv(S['test']['cmr'],   OUT / 'task1_cmr_test.csv')

# Task 2: Fundus single-tower
write_csv(S['train']['fundus'], OUT / 'task2_fundus_train.csv')
write_csv(S['val']['fundus'],   OUT / 'task2_fundus_val.csv')
write_csv(S['test']['fundus'],  OUT / 'task2_fundus_test.csv')

# Task 3&4: Paired (CrossAttn + Late-fusion)
def paired_rows(split):
    pe = S[split]['paired_eids']
    cmr_map   = {r['eid']: r for r in S[split]['cmr']}
    fund_map  = {r['eid']: r for r in S[split]['fundus']}
    rows = []
    for eid in pe:
        if eid in cmr_map and eid in fund_map:
            rows.append({'eid': eid,
                         'cmr_path': cmr_map[eid]['path'],
                         'fundus_path': fund_map[eid]['path']})
    return rows

write_csv(paired_rows('train'), OUT / 'task34_paired_train.csv')
write_csv(paired_rows('val'),   OUT / 'task34_paired_val.csv')
write_csv(paired_rows('test'),  OUT / 'task34_paired_test.csv')

print()

# ── PDF helpers ───────────────────────────────────────────────────────────────
BLUE   = '#1a5276'
LBLUE  = '#d6eaf8'
GREEN  = '#1e8449'
LGREEN = '#d5f5e3'
ORANGE = '#d35400'
LORANGE= '#fdebd0'
PURPLE = '#6c3483'
LPURPLE= '#e8daef'
GRAY   = '#566573'
LGRAY  = '#f2f3f4'
WHITE  = '#ffffff'

def add_page(pdf, fig_size=(11.69, 8.27)):
    fig = plt.figure(figsize=fig_size)
    return fig

def header(ax, title, color=BLUE):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.add_patch(FancyBboxPatch((0,0.7),(1,0.3),
        boxstyle='round,pad=0.02', fc=color, ec='none'))
    ax.text(0.5,0.85, title, ha='center', va='center',
            fontsize=16, fontweight='bold', color='white',
            fontfamily='DejaVu Sans')

def section_box(ax, title, body_lines, x=0.02, y=0.95, w=0.96,
                title_color=BLUE, bg_color=LBLUE, title_fs=11, body_fs=9):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    line_h = 0.055
    total_h = line_h * (1 + len(body_lines)) + 0.04
    ax.add_patch(FancyBboxPatch((x, y-line_h-0.01), w, line_h+0.01,
        boxstyle='round,pad=0.01', fc=title_color, ec='none'))
    ax.text(x+0.01, y-line_h/2-0.005, title,
            va='center', fontsize=title_fs, fontweight='bold',
            color='white', fontfamily='DejaVu Sans')
    ax.add_patch(FancyBboxPatch((x, y-total_h), w, total_h-line_h-0.01,
        boxstyle='round,pad=0.01', fc=bg_color, ec='none'))
    for i, line in enumerate(body_lines):
        ax.text(x+0.015, y - line_h - 0.025 - i*line_h,
                line, va='center', fontsize=body_fs,
                fontfamily='DejaVu Sans', color='#1a1a1a')
    return y - total_h - 0.02

def kv_table(ax, rows, x=0.02, y=0.95, w=0.96, col_split=0.38,
             header_color=BLUE, alt_color=LGRAY, fs=8.5):
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    lh = 0.052
    ax.add_patch(FancyBboxPatch((x,y-lh), w, lh,
        boxstyle='round,pad=0.005', fc=header_color, ec='none'))
    ax.text(x+0.01, y-lh/2, rows[0][0], va='center', fontsize=fs,
            fontweight='bold', color='white')
    ax.text(x+col_split, y-lh/2, rows[0][1], va='center', fontsize=fs,
            fontweight='bold', color='white')
    for i, (k, v) in enumerate(rows[1:]):
        bg = alt_color if i%2==0 else WHITE
        ax.add_patch(FancyBboxPatch((x, y-(i+2)*lh), w, lh,
            boxstyle='round,pad=0.002', fc=bg, ec='#cccccc', lw=0.3))
        ax.text(x+0.01, y-(i+1.5)*lh, k, va='center', fontsize=fs,
                fontfamily='DejaVu Sans', color=GRAY, fontweight='bold')
        ax.text(x+col_split, y-(i+1.5)*lh, v, va='center', fontsize=fs,
                fontfamily='DejaVu Sans', color='#1a1a1a')
    return y - (len(rows)+0.5)*lh

# ── PDF Pages ─────────────────────────────────────────────────────────────────
pdf_path = HERE / 'task_reports' / 'training_plan.pdf'
print(f'Generating PDF: {pdf_path}')

with PdfPages(pdf_path) as pdf:

    # ────────────────────────────── PAGE 1: COVER ─────────────────────────────
    fig = plt.figure(figsize=(11.69, 8.27))
    ax  = fig.add_axes([0,0,1,1])
    ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')
    ax.add_patch(plt.Rectangle((0,0),1,1, fc='#1a2636'))
    ax.add_patch(plt.Rectangle((0,0.55),1,0.45, fc=BLUE, alpha=0.9))
    ax.text(0.5, 0.78,
            'Dual-Tower Cross-Modal Cardiovascular Prediction',
            ha='center', va='center', fontsize=20, fontweight='bold',
            color='white', fontfamily='DejaVu Sans')
    ax.text(0.5, 0.65,
            'RETFound ViT-L (Fundus)  x  MedSAM ViT-B (CMR)  |  UK Biobank',
            ha='center', va='center', fontsize=13,
            color='#aed6f1', fontfamily='DejaVu Sans')
    ax.text(0.5, 0.43,
            'Experiment Task Planning Report',
            ha='center', va='center', fontsize=18, fontweight='bold',
            color='white', fontfamily='DejaVu Sans')

    stats = [
        ('CMR Train Samples\n(npy confirmed)', f'{len(S["train"]["cmr"]):,}'),
        ('Fundus Train Samples\n(PNG confirmed)', f'{len(S["train"]["fundus"]):,}'),
        ('Strict Paired Train\n(same-downsample)', f'{len(S["train"]["paired_eids"]):,}'),
        ('Max Paired Train\n(all verified pairs)', '6,725'),
    ]
    for i, (lbl, val) in enumerate(stats):
        bx = 0.06 + i*0.235
        ax.add_patch(FancyBboxPatch((bx,0.14),0.21,0.18,
            boxstyle='round,pad=0.02', fc='#2e4057', ec='#5dade2', lw=1.5))
        ax.text(bx+0.105, 0.26, val, ha='center', va='center',
                fontsize=18, fontweight='bold', color='#5dade2')
        ax.text(bx+0.105, 0.185, lbl, ha='center', va='center',
                fontsize=8, color='#aed6f1', multialignment='center')

    ax.text(0.5, 0.04, 'UK Biobank  |  RETFound ViT-L x MedSAM ViT-B  |  2026',
            ha='center', va='center', fontsize=9, color='#7f8c8d')
    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ─────────────────────── PAGE 2: Architecture Overview ────────────────────
    fig, axes = plt.subplots(1,1, figsize=(11.69,8.27))
    ax = axes; ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

    ax.add_patch(FancyBboxPatch((0.01,0.93),0.98,0.06,
        boxstyle='round,pad=0.01', fc=BLUE, ec='none'))
    ax.text(0.5,0.96,'Framework Architecture Overview', ha='center', va='center',
            fontsize=14, fontweight='bold', color='white')

    for tx, label, dim, color in [
        (0.08, 'Fundus\n(Retinal Photo)', 'ViT-L\n1024-dim', '#1a5276'),
        (0.74, 'CMR\n(Cardiac MRI x4)', 'ViT-B\n768-dim',   '#145a32'),
    ]:
        ax.add_patch(FancyBboxPatch((tx,0.65),0.18,0.22,
            boxstyle='round,pad=0.02', fc=color, ec='none', alpha=0.85))
        ax.text(tx+0.09,0.80, label, ha='center', va='center',
                fontsize=11, fontweight='bold', color='white', multialignment='center')
        ax.text(tx+0.09,0.69, dim, ha='center', va='center',
                fontsize=9, color='#aed6f1', multialignment='center')

    ax.add_patch(FancyBboxPatch((0.36,0.60),0.28,0.22,
        boxstyle='round,pad=0.02', fc=PURPLE, ec='none', alpha=0.9))
    ax.text(0.50,0.75,'CrossModalGating', ha='center', va='center',
            fontsize=11, fontweight='bold', color='white')
    ax.text(0.50,0.67,'Bidirectional Cross-Attn + LoRA', ha='center', va='center',
            fontsize=8.5, color='#d7bde2')

    for (x1,y1,dx,dy) in [
        (0.26,0.76, 0.10,0), (0.64,0.76, 0.10,0),
        (0.50,0.60, 0,  -0.10),
    ]:
        ax.annotate('', xy=(x1+dx,y1+dy), xytext=(x1,y1),
                    arrowprops=dict(arrowstyle='->', color='#aaaaaa', lw=1.5))

    for tx, lbl in [(0.1,'Cls x4\nReg x3'), (0.74,'Cls x4\nReg x3'), (0.38,'Fused Cls x4\nFused Reg x3')]:
        ax.add_patch(FancyBboxPatch((tx,0.38),0.18,0.12,
            boxstyle='round,pad=0.01', fc=ORANGE, ec='none', alpha=0.8))
        ax.text(tx+0.09,0.44, lbl, ha='center', va='center',
                fontsize=8, color='white', multialignment='center')

    stage_info = [
        ('Stage 1', 'Single-tower multi-task fine-tuning\n(Exp A + Exp B)', '#1a5276'),
        ('Stage 2', 'CrossModalGating specialized training\n(frozen backbones, train interaction layers only)', '#6c3483'),
        ('Stage 3', 'Joint full fine-tuning\n(layered LR, all layers unfrozen)', '#145a32'),
    ]
    for i,(s,t,c) in enumerate(stage_info):
        bx = 0.04 + i*0.32
        ax.add_patch(FancyBboxPatch((bx,0.04),0.29,0.28,
            boxstyle='round,pad=0.02', fc=c, ec='none', alpha=0.15))
        ax.add_patch(FancyBboxPatch((bx,0.27),0.29,0.05,
            boxstyle='round,pad=0.01', fc=c, ec='none'))
        ax.text(bx+0.145,0.295, s, ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        ax.text(bx+0.145,0.17, t, ha='center', va='center',
                fontsize=8.5, color='#1a1a1a', multialignment='center')

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ─────────────────────── PAGE 3: Dataset Summary ──────────────────────────
    fig, ax = plt.subplots(figsize=(11.69, 8.27))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

    ax.add_patch(FancyBboxPatch((0.01, 0.93), 0.98, 0.06,
        boxstyle='round,pad=0.01', fc=BLUE, ec='none'))
    ax.text(0.5, 0.96, 'Dataset Summary & Cohort Design',
            ha='center', va='center', fontsize=14, fontweight='bold', color='white')

    # ── Left: three cohorts overview ──
    cohorts = [
        ('CMR Single-Tower Cohort', GREEN,
         ['Source: stage1_cmr_sampled.csv  (after smart_downsample neg_ratio=3)',
          'Total: 21,599 EIDs  |  Train: 14,472  Val: 1,801  Test: 1,807',
          'Files: npy verified on disk for all rows',
          'Labels: 4 cls (pos-rate 4.6–29%)  +  3 LV regression targets',
          'CSV: contrastive_pretrain/task_reports/task1_cmr_*.csv']),
        ('Fundus Single-Tower Cohort', BLUE,
         ['Source: stage1_fundus_sampled.csv  (after smart_downsample neg_ratio=3)',
          'Total: 19,122 EIDs  |  Train: 15,276  Val: 1,907  Test: 1,910',
          'Files: PNG verified on disk for all rows',
          'Labels: 4 cls  +  LV reg only for ~13,859 cross-modal EIDs (reg_mask otherwise)',
          'CSV: contrastive_pretrain/task_reports/task2_fundus_*.csv']),
        ('Strict Paired Cohort  (primary for Exp C/D)', PURPLE,
         ['Source: intersection of CMR-sampled & Fundus-sampled CSVs',
          'Total: 2,779 EIDs  |  Train: 2,694  Val: 44  Test: 41',
          'Limitation: smart_downsample independently pruned cross-modal negatives',
          'Pos-rate elevated: any-pos=68.6%  (cardiomyopathy only 5.5%)',
          'CSV: contrastive_pretrain/task_reports/task34_paired_*.csv']),
    ]

    y = 0.89
    for title, color, lines in cohorts:
        h = 0.045 + 0.038 * len(lines)
        ax.add_patch(FancyBboxPatch((0.01, y - h), 0.58, h,
            boxstyle='round,pad=0.01', fc=color, ec='none', alpha=0.12))
        ax.add_patch(FancyBboxPatch((0.01, y - 0.042), 0.58, 0.042,
            boxstyle='round,pad=0.005', fc=color, ec='none'))
        ax.text(0.02, y - 0.021, title, va='center', fontsize=9.5,
                fontweight='bold', color='white')
        for i, line in enumerate(lines):
            ax.text(0.025, y - 0.052 - i * 0.038, line, va='center',
                    fontsize=8, color='#1a1a1a')
        y -= h + 0.015

    # ── Right: Max Paired Cohort (backup) ──
    bx, bw = 0.62, 0.37
    ax.add_patch(FancyBboxPatch((bx, 0.06), bw, 0.82,
        boxstyle='round,pad=0.015', fc='#fef9e7', ec=ORANGE, lw=2))
    ax.add_patch(FancyBboxPatch((bx, 0.80), bw, 0.08,
        boxstyle='round,pad=0.01', fc=ORANGE, ec='none'))
    ax.text(bx + bw/2, 0.84,
            'MAX PAIRED COHORT\n(Backup Dataset)',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white',
            multialignment='center')

    max_info = [
        ('Total EIDs',   '9,493'),
        ('Train',        '6,725'),
        ('Val',          '1,379'),
        ('Test',         '1,389'),
        ('vs Strict Paired', '3.4x larger'),
        ('', ''),
        ('Construction', 'All EIDs in fundus_table_extended'),
        ('',             'with verified CMR npy + PNG'),
        ('Cross-visit OK?', 'Yes (pair_type=cross_inst allowed)'),
        ('Any-positive rate', '15.1%'),
        ('Ischemic HD pos',   '796  (8.4%)'),
        ('MI (I21) pos',      '320  (3.4%)'),
        ('Cardiomyopathy pos','121  (1.3%)'),
        ('AF pos',            '728  (7.7%)'),
        ('', ''),
        ('CSV files',    'task_reports/paired_max_*.csv'),
        ('Status',       'Ready -- use if strict paired'),
        ('',             'cohort underperforms'),
    ]

    iy = 0.77
    for k, v in max_info:
        if k == '' and v == '':
            iy -= 0.01
            continue
        ax.text(bx + 0.01, iy, k, va='center', fontsize=7.5,
                fontweight='bold', color=ORANGE)
        ax.text(bx + 0.18, iy, v, va='center', fontsize=7.5, color='#1a1a1a')
        iy -= 0.033

    ax.text(bx + bw/2, 0.10,
            'Why max cohort exists:\nstrict paired was pruned by independent\nneg_ratio=3 downsampling on each modality.',
            ha='center', va='center', fontsize=7.5, color=GRAY,
            multialignment='center',
            bbox=dict(boxstyle='round,pad=0.3', fc='#fdebd0', ec=ORANGE, lw=0.8))

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ────────────────── TASK PAGES ────────────────────────────────────────────
    task_defs = [
        dict(
            num=1, name='Exp A  --  CMR Single-Tower Multi-Task Fine-Tuning',
            color=GREEN, lcolor=LGREEN,
            modality='CMR (Cardiac MRI)',
            encoder='MedSAM ViT-B  ->  768-dim pooled embedding',
            arch='CMREncoder -> TaskHead (4 classification + 3 regression)',
            data_train=len(S['train']['cmr']),
            data_val=len(S['val']['cmr']),
            data_test=len(S['test']['cmr']),
            data_note='Source: stage1_cmr_sampled_*.csv  |  all .npy files verified on disk',
            csv_files='task1_cmr_train/val/test.csv',
            npy_path='/data/home/shujia/UKB/CMRI/preprocessed_lax_sax/{eid}.npy',
            targets='Cls: ischemic HD / MI / cardiomyopathy-HF / AF\nReg: LVEF / LVEDV / LVESV',
            lr='1e-4', bs='4/GPU x 4 GPUs = 16', epochs='15',
            optimizer='AdamW  weight_decay=0.05',
            scheduler='CosineAnnealingLR  eta_min=1e-6',
            amp='FP16 mixed precision + GradScaler',
            grad_ckpt='Off (CMR model only 86M params, memory sufficient)',
            patience='5 epochs (early stopping on mean_AUC)',
            freeze='None -- full parameter fine-tuning',
            cmd='EXP=A GPUS=4 BATCH=4 bash contrastive_pretrain/run_stage1_mini.sh',
            purpose='Establish CMR single-tower AUC baseline; verify MedSAM encoder ability to predict structural heart disease',
        ),
        dict(
            num=2, name='Exp B  --  Fundus Single-Tower Multi-Task Fine-Tuning',
            color=BLUE, lcolor=LBLUE,
            modality='Fundus (Retinal Photography)',
            encoder='RETFound ViT-L  ->  1024-dim global-pool embedding',
            arch='FundusEncoder -> TaskHead (4 cls + 3 reg; reg_mask applied when LV labels absent)',
            data_train=len(S['train']['fundus']),
            data_val=len(S['val']['fundus']),
            data_test=len(S['test']['fundus']),
            data_note='Source: stage1_fundus_sampled_*.csv  |  all PNG images verified on disk',
            csv_files='task2_fundus_train/val/test.csv',
            npy_path='/data/home/home6/fundus_data/UKB/fundus_images/{eid}_21016_{inst}_*.png',
            targets='Cls: ischemic HD / MI / cardiomyopathy-HF / AF\nReg: LVEF/LVEDV/LVESV (only ~13,859 paired EIDs have LV labels)',
            lr='1e-4', bs='4/GPU x 4 GPUs = 16', epochs='15',
            optimizer='AdamW  weight_decay=0.05',
            scheduler='CosineAnnealingLR  eta_min=1e-6',
            amp='FP16 + GradScaler',
            grad_ckpt='ON (ViT-L 307M params; grad_ckpt reduces peak mem from ~10GB to ~3GB/GPU)',
            patience='5 epochs (on mean_AUC)',
            freeze='Full fine-tuning (freeze_fundus_blocks=0)',
            cmd='EXP=B GPUS=4 BATCH=4 bash contrastive_pretrain/run_stage1_mini.sh',
            purpose='Establish fundus single-tower AUC baseline; verify RETFound can predict CVD from retinal images',
        ),
        dict(
            num=3, name='Exp C  --  Late Fusion Baseline',
            color=ORANGE, lcolor=LORANGE,
            modality='CMR + Fundus (paired cohort)',
            encoder='MedSAM ViT-B (768d) + RETFound ViT-L (1024d) -> Concat -> 1792d',
            arch='Concat([z_cmr, z_fundus]) -> shared TaskHead (no CrossModalGating)',
            data_train=len(S['train']['paired_eids']),
            data_val=len(S['val']['paired_eids']),
            data_test=len(S['test']['paired_eids']),
            data_note='Source: task34_paired_*.csv  |  both CMR npy and fundus PNG verified',
            csv_files='task34_paired_train/val/test.csv',
            npy_path='CMR: preprocessed_lax_sax/{eid}.npy\nFundus: fundus_images/{eid}_2101x_{inst}_*.png',
            targets='Cls: 4 tasks; Reg: LVEF/LVEDV/LVESV (all paired EIDs have LV labels from CMR row)',
            lr='1e-4', bs='2/GPU x 4 GPUs = 8', epochs='15',
            optimizer='AdamW  weight_decay=0.05',
            scheduler='CosineAnnealingLR',
            amp='FP16 + GradScaler',
            grad_ckpt='ON (both encoders active simultaneously)',
            patience='5 epochs',
            freeze='Full fine-tuning',
            cmd='EXP=C GPUS=4 BATCH=2 bash contrastive_pretrain/run_stage1_mini.sh\n(Exp C mode needs implementing: concat fusion, no CrossAttn)',
            purpose='Ablation control for Exp D: dual-modal input WITHOUT cross-modal interaction -- quantifies CrossModalGating contribution',
        ),
        dict(
            num=4, name='Exp D  --  Dual-Tower + CrossModalGating',
            color=PURPLE, lcolor=LPURPLE,
            modality='CMR + Fundus (paired cohort)',
            encoder='MedSAM ViT-B (768d) + RETFound ViT-L (1024d) -> CrossModalGating',
            arch='Bidirectional cross-attn + Sigmoid gate + LoRA backflow -> per-tower TaskHead + align loss',
            data_train=len(S['train']['paired_eids']),
            data_val=len(S['val']['paired_eids']),
            data_test=len(S['test']['paired_eids']),
            data_note='Source: task34_paired_*.csv  (same as Exp C)',
            csv_files='task34_paired_train/val/test.csv',
            npy_path='CMR: preprocessed_lax_sax/{eid}.npy\nFundus: fundus_images/{eid}_2101x_{inst}_*.png',
            targets='Same as Exp C + align_loss (cosine alignment of z_fundus and z_cmr)',
            lr='1e-4', bs='2/GPU x 4 GPUs = 8', epochs='15',
            optimizer='AdamW  weight_decay=0.05',
            scheduler='CosineAnnealingLR',
            amp='FP16 + GradScaler',
            grad_ckpt='ON',
            patience='5 epochs',
            freeze='Full fine-tuning',
            cmd='EXP=D GPUS=4 BATCH=2 bash contrastive_pretrain/run_stage1_mini.sh',
            purpose='Core experiment: verify CrossModalGating bidirectional knowledge injection improves AUC over Exp C (late fusion) and Exp A/B (single-tower)',
        ),
        dict(
            num=5, name='Stage 2  --  CrossModalGating Specialized Training',
            color='#117a65', lcolor='#d1f2eb',
            modality='CMR + Fundus (paired cohort)',
            encoder='Load Stage 1 checkpoints; freeze both backbones; train only CrossModalGating + TaskHead',
            arch='Load Exp A / Exp B checkpoints -> freeze backbone -> train cross-modal layers',
            data_train=len(S['train']['paired_eids']),
            data_val=len(S['val']['paired_eids']),
            data_test=len(S['test']['paired_eids']),
            data_note='Same as Exp D  |  depends on Stage 1 (Exp A + Exp B) checkpoints',
            csv_files='task34_paired_train/val/test.csv',
            npy_path='Same as Exp D',
            targets='Same as Exp D',
            lr='5e-4 (CrossModal layers) / backbone frozen', bs='4/GPU x 4 GPUs = 16', epochs='20',
            optimizer='AdamW  weight_decay=0.01',
            scheduler='CosineAnnealingLR',
            amp='FP16',
            grad_ckpt='Not needed (backbone frozen, no activation backprop)',
            patience='5 epochs',
            freeze='Both backbones frozen; only CrossModalGating + TaskHead trainable',
            cmd='(Stage 2 training script to be implemented, depends on Exp A/B checkpoints)',
            purpose='Optimize cross-modal interaction layers without multi-task loss interference; provides strong initialization for Stage 3',
        ),
        dict(
            num=6, name='Stage 3  --  Joint Full Fine-Tuning',
            color='#6e2f1a', lcolor='#fce4d6',
            modality='CMR + Fundus (paired cohort)',
            encoder='Load Stage 2 checkpoint; unfreeze all layers; layered learning rates',
            arch='Layered LR: backbone 1e-5 / CrossModal 5e-5 / TaskHead 1e-4',
            data_train=len(S['train']['paired_eids']),
            data_val=len(S['val']['paired_eids']),
            data_test=len(S['test']['paired_eids']),
            data_note='Same as Exp D  |  depends on Stage 2 checkpoint',
            csv_files='task34_paired_train/val/test.csv',
            npy_path='Same as Exp D',
            targets='Same as Exp D',
            lr='backbone=1e-5 / CrossModal=5e-5 / Head=1e-4', bs='2/GPU x 4 GPUs = 8', epochs='30',
            optimizer='AdamW  weight_decay=0.05, per-param-group',
            scheduler='CosineAnnealingWarmRestarts',
            amp='FP16 + GradScaler',
            grad_ckpt='ON',
            patience='7 epochs',
            freeze='All unfrozen, layered learning rates',
            cmd='(Stage 3 training script to be implemented, depends on Stage 2 checkpoint)',
            purpose='Joint fine-tune all parameters to produce final model; full ablation comparison against Exp A/B/C/D',
        ),
    ]

    for t in task_defs:
        fig = plt.figure(figsize=(11.69,8.27))
        ax  = fig.add_axes([0.02,0.02,0.96,0.96])
        ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis('off')

        # Header
        ax.add_patch(FancyBboxPatch((0,0.92),1,0.08,
            boxstyle='round,pad=0.01', fc=t['color'], ec='none'))
        ax.text(0.5,0.96, f"Task {t['num']}: {t['name']}",
                ha='center', va='center', fontsize=12,
                fontweight='bold', color='white')

        # Purpose
        ax.add_patch(FancyBboxPatch((0,0.85),1,0.065,
            boxstyle='round,pad=0.005', fc=t['lcolor'], ec='none'))
        ax.text(0.01,0.883, f"Objective: {t['purpose']}",
                va='center', fontsize=8.5, color='#1a1a1a',
                wrap=True, fontfamily='DejaVu Sans')

        # Left column: Data
        lw, rw = 0.48, 0.48
        rx = 0.52

        data_rows = [
            ('Split', 'Sample Count (files verified on disk)'),
            ('Train', f"{t['data_train']:,} samples"),
            ('Val',   f"{t['data_val']:,} samples"),
            ('Test',  f"{t['data_test']:,} samples"),
            ('Total', f"{t['data_train']+t['data_val']+t['data_test']:,} samples"),
        ]
        y = 0.82
        ax.text(0.01, y+0.005, 'Data Configuration', fontsize=10,
                fontweight='bold', color=t['color'], va='center')
        y -= 0.03
        lh = 0.053
        for i,(k,v) in enumerate(data_rows):
            bg = t['lcolor'] if i%2==0 else WHITE
            ax.add_patch(FancyBboxPatch((0.01,y-lh),lw,lh,
                boxstyle='round,pad=0.002', fc=bg, ec='#cccccc', lw=0.3))
            fw = 'bold' if i==0 or i==len(data_rows)-1 else 'normal'
            ax.text(0.02,  y-lh/2, k, va='center', fontsize=8.5,
                    fontweight=fw, color=t['color'] if i==0 else GRAY)
            ax.text(0.28, y-lh/2, v, va='center', fontsize=8.5,
                    fontweight=fw, color='#1a1a1a')
            y -= lh

        y -= 0.01
        ax.text(0.01, y, f"  Modality: {t['modality']}", fontsize=8,
                color='#2c3e50', va='center')
        y -= 0.04
        for line in t['data_note'].split('\n'):
            ax.text(0.01, y, f"  {line}", fontsize=7.5, color=GRAY, va='center')
            y -= 0.035
        y -= 0.01
        ax.text(0.01, y, f"  CSV path: contrastive_pretrain/task_reports/{t['csv_files']}",
                fontsize=7.5, color=GRAY, va='center')
        y -= 0.04
        for line in t['npy_path'].split('\n'):
            ax.text(0.01, y, f"  File path: {line}", fontsize=7.5, color=GRAY, va='center')
            y -= 0.035

        # Right column: Training config
        ax.text(rx, 0.82+0.005, 'Training Strategy & Hyperparameters', fontsize=10,
                fontweight='bold', color=t['color'], va='center')
        cfg_rows = [
            ('Encoder',      t['encoder']),
            ('Architecture', t['arch']),
            ('Targets',      t['targets'].replace('\n', ' / ')),
            ('Learning Rate',t['lr']),
            ('Batch Size',   t['bs']),
            ('Epochs',       t['epochs']),
            ('Optimizer',    t['optimizer']),
            ('LR Schedule',  t['scheduler']),
            ('Mixed Prec.',  t['amp']),
            ('Grad Ckpt',    t['grad_ckpt']),
            ('Early Stop',   t['patience']),
            ('Freeze',       t['freeze']),
        ]
        cy = 0.79
        for k, v in cfg_rows:
            v_lines = textwrap.wrap(v, width=52)
            h = max(0.042, 0.042 * len(v_lines))
            ax.add_patch(FancyBboxPatch((rx, cy-h), rw, h,
                boxstyle='round,pad=0.002', fc=LGRAY, ec='#dddddd', lw=0.3))
            ax.text(rx+0.01, cy-h/2, k, va='center', fontsize=7.5,
                    fontweight='bold', color=t['color'])
            ax.text(rx+0.17, cy-h/2, v_lines[0], va='center', fontsize=7.5,
                    color='#1a1a1a')
            for li, vl in enumerate(v_lines[1:], 1):
                ax.text(rx+0.17, cy-h/2 + (0.5-li)*0.03, vl,
                        va='center', fontsize=7.5, color='#1a1a1a')
            cy -= h + 0.003

        # Command box
        cy -= 0.005
        ax.add_patch(FancyBboxPatch((rx, cy-0.075), rw, 0.075,
            boxstyle='round,pad=0.01', fc='#1a2636', ec='none'))
        ax.text(rx+0.01, cy-0.01, 'Launch Command:', fontsize=8,
                color='#aed6f1', fontweight='bold', va='center')
        for li, line in enumerate(t['cmd'].split('\n')):
            ax.text(rx+0.01, cy-0.033-li*0.022, line,
                    fontsize=7, color='#5dade2', va='center',
                    fontfamily='monospace')

        pdf.savefig(fig, bbox_inches='tight'); plt.close()

    # ─────────────────────── PAGE: Timeline ──────────────────────────────────
    fig, ax = plt.subplots(figsize=(11.69,8.27))
    ax.set_xlim(0,10); ax.set_ylim(0,6); ax.axis('off')
    fig.patch.set_facecolor('#f8f9fa')

    ax.text(5, 5.6, 'Experiment Execution Timeline & Parallel Strategy',
            ha='center', va='center', fontsize=14,
            fontweight='bold', color=BLUE)

    timeline = [
        # (label, t_start, t_end, row, color, note)
        ('CMR npy preprocessing\n(COMPLETE)',    0, 1.5, 4.5, '#95a5a6', 'Done\n24,499 npy'),
        ('Exp B Fundus Single-Tower\n(ready now)', 0, 3, 3.5, BLUE,    '15 epochs\n~6-8h / 4 GPUs'),
        ('Exp A CMR Single-Tower\n(ready now)',    0, 3, 2.5, GREEN,   '15 epochs\n~4-5h / 4 GPUs'),
        ('Exp C Late Fusion',                      3, 5, 4.5, ORANGE,  '15 epochs\npaired data'),
        ('Exp D Dual-Tower+CrossAttn',             3, 5, 3.5, PURPLE,  '15 epochs\ncore experiment'),
        ('Stage 2 CrossModal Training',            5, 7, 2.5, '#117a65','depends on\nA+B ckpt'),
        ('Stage 3 Joint Fine-Tuning',              7, 9, 2.5, '#6e2f1a','depends on\nStage2 ckpt'),
        ('Linear Probe / Evaluation',              9, 10, 2.5, GRAY,   'final results'),
    ]

    for label, t0, t1, row, color, note in timeline:
        ax.add_patch(FancyBboxPatch((t0*0.9+0.2, row-0.3), (t1-t0)*0.9, 0.6,
            boxstyle='round,pad=0.05', fc=color, ec='white', lw=1.5, alpha=0.85))
        ax.text((t0+t1)*0.45+0.2, row, label, ha='center', va='center',
                fontsize=7.5, color='white', fontweight='bold', multialignment='center')
        ax.text((t1)*0.9+0.25, row, note, va='center',
                fontsize=6.5, color=color, multialignment='left')

    ax.add_patch(FancyBboxPatch((0.2,1.2),4*0.9,0.5,
        boxstyle='round,pad=0.05', fc='none', ec=BLUE, lw=1.5, ls='--'))
    ax.text(2.2,1.45,'GPU 1+3 available for Exp A+B (CMR preprocessing uses CPU only)',
            ha='center', va='center', fontsize=8, color=BLUE)

    ax.text(5, 0.5, 'Timeline (relative units; each grid ~4-8 hours)',
            ha='center', va='center', fontsize=9, color=GRAY)

    pdf.savefig(fig, bbox_inches='tight'); plt.close()

print(f'\nPDF saved: {pdf_path}')
print(f'CSV files saved in: {OUT}')
print('Files:')
for f in sorted(OUT.iterdir()):
    print(f'  {f.name}  ({f.stat().st_size//1024} KB)')
