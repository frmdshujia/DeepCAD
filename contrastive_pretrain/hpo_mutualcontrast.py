"""
hpo_mutualcontrast.py

Automated hyperparameter sweep for Mutual Contrastive training.

Phase 1 (pilot): Run each config for --pilot_epochs epochs on 4 GPUs.
                  Log val mean_AUC_f after each epoch.
Phase 2 (full):  Re-run the best config for --full_epochs epochs.

Grid searched:
  lr   : [5e-5, 1e-4, 3e-4]
  lam  : [0.1, 0.3]
  (gamma=1.0, temperature=0.07 fixed)

Usage:
  tmux new-session -d -s hpo \\
    "cd /data/home/shujia/CHD/model_train/RETFound_MAE-main && \\
     /data/home/shujia/miniconda3/envs/modeltrain/bin/python \\
     contrastive_pretrain/hpo_mutualcontrast.py 2>&1 | tee /tmp/hpo.log"
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import time

HERE   = pathlib.Path(__file__).parent
ROOT   = HERE.parent
PYTHON    = '/data/home/shujia/miniconda3/envs/modeltrain/bin/python'
TORCHRUN  = '/data/home/shujia/miniconda3/envs/modeltrain/bin/torchrun'

PILOT_EPOCHS = 5
FULL_EPOCHS  = 30
GPUS         = 4
BATCH        = 8       # per-GPU batch size
MASTER_PORT  = 29502   # avoid collision with other runs

GRID = [
    {'lr': 5e-5,  'lam': 0.1},
    {'lr': 5e-5,  'lam': 0.3},
    {'lr': 1e-4,  'lam': 0.1},
    {'lr': 1e-4,  'lam': 0.3},
    {'lr': 3e-4,  'lam': 0.1},
    {'lr': 3e-4,  'lam': 0.3},
]

OUT_ROOT = HERE / 'hpo_results'
OUT_ROOT.mkdir(exist_ok=True)


def run_config(cfg: dict, epochs: int, tag: str, out_dir: pathlib.Path) -> float:
    """
    Launch torchrun for one config. Returns best val mean_AUC_f achieved.
    Reads it from the saved checkpoint JSON sidecar.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = OUT_ROOT / f'{tag}.log'

    cmd = [
        TORCHRUN,
        f'--nproc_per_node={GPUS}',
        f'--master_port={MASTER_PORT}',
        'contrastive_pretrain/train_mutualcontrast.py',
        '--epochs',       str(epochs),
        '--batch_size',   str(BATCH),
        '--lr',           str(cfg['lr']),
        '--lam',          str(cfg['lam']),
        '--gamma',        '1.0',
        '--temperature',  '0.07',
        '--patience',     str(epochs),   # no early stopping during pilot
        '--grad_ckpt',
        '--out_dir',      str(out_dir),
    ]

    print(f'\n{"="*60}')
    print(f'[HPO] Running {tag}')
    print(f'  lr={cfg["lr"]}  lam={cfg["lam"]}  epochs={epochs}')
    print(f'  log → {log_path}')
    print(f'{"="*60}', flush=True)

    t0 = time.time()
    with open(log_path, 'w') as flog:
        proc = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=flog,
            stderr=subprocess.STDOUT,
        )

    elapsed = time.time() - t0
    print(f'[HPO] {tag} finished in {elapsed/60:.1f} min  '
          f'(returncode={proc.returncode})', flush=True)

    # read best AUC from checkpoint
    ckpt_path = out_dir / 'mutualcontrast_best_full.pth'
    if not ckpt_path.exists():
        print(f'[HPO] WARNING: no checkpoint found for {tag}', flush=True)
        return 0.0

    import torch
    ckpt = torch.load(ckpt_path, map_location='cpu')
    best_auc = ckpt.get('metrics', {}).get('mean_auc_f', 0.0)
    print(f'[HPO] {tag}: best val mean_AUC_f = {best_auc:.4f}', flush=True)

    # save a compact summary JSON
    summary = {
        'tag': tag, 'config': cfg, 'epochs': epochs,
        'best_auc_f': best_auc,
        'epoch': ckpt.get('epoch', -1),
        'metrics': ckpt.get('metrics', {}),
        'elapsed_min': round(elapsed / 60, 1),
    }
    with open(OUT_ROOT / f'{tag}_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    return best_auc


def _plot_curves(out_dir: pathlib.Path):
    """Read metrics_history.json and save loss_curves.png."""
    hist_path = out_dir / 'metrics_history.json'
    if not hist_path.exists():
        print(f'[HPO] No history file found at {hist_path}, skipping plot.', flush=True)
        return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        with open(hist_path) as f:
            history = json.load(f)

        epochs      = [r['epoch'] for r in history]
        train_total = [r.get('train_total', float('nan')) for r in history]
        train_task_f= [r.get('train_task_f', float('nan')) for r in history]
        train_task_c= [r.get('train_task_c', float('nan')) for r in history]
        train_align = [r.get('train_align', float('nan')) for r in history]
        auc_f       = [r.get('mean_auc_f', float('nan')) for r in history]
        auc_c       = [r.get('mean_auc_c', float('nan')) for r in history]
        val_align   = [r.get('val_align', float('nan')) for r in history]

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        # panel 1: train losses
        ax = axes[0]
        ax.plot(epochs, train_total,  label='L_total',  linewidth=2)
        ax.plot(epochs, train_task_f, label='L_task_f', linestyle='--')
        ax.plot(epochs, train_task_c, label='L_task_c', linestyle='--')
        ax.plot(epochs, train_align,  label='L_align',  linestyle=':')
        ax.set_xlabel('Epoch'); ax.set_ylabel('Loss')
        ax.set_title('Train Loss Curves')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # panel 2: val AUC
        ax = axes[1]
        ax.plot(epochs, auc_f, label='mean_AUC_f (fundus)', linewidth=2, color='steelblue')
        ax.plot(epochs, auc_c, label='mean_AUC_c (CMR)',    linewidth=2, color='orange',
                linestyle='--')
        ax.set_xlabel('Epoch'); ax.set_ylabel('AUC')
        ax.set_title('Val Mean AUC (4 diseases)')
        ax.set_ylim(0.4, 1.0); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # panel 3: val align loss
        ax = axes[2]
        ax.plot(epochs, val_align, color='purple', linewidth=2)
        ax.set_xlabel('Epoch'); ax.set_ylabel('InfoNCE Loss')
        ax.set_title('Val Contrastive Alignment Loss')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = out_dir / 'loss_curves.png'
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f'[HPO] Loss curves saved → {save_path}', flush=True)
    except Exception as e:
        print(f'[HPO] Plot failed (non-fatal): {e}', flush=True)


def main():
    print('=' * 60)
    print('Mutual Contrastive HPO Sweep')
    print(f'  Grid: {len(GRID)} configs × {PILOT_EPOCHS} pilot epochs')
    print(f'  GPUs: {GPUS}   batch/GPU: {BATCH}')
    print(f'  Results dir: {OUT_ROOT}')
    print('=' * 60, flush=True)

    # ── Phase 1: pilot sweep ───────────────────────────────────────────────
    results = []
    for i, cfg in enumerate(GRID):
        tag     = f'pilot_lr{cfg["lr"]:.0e}_lam{cfg["lam"]}'
        out_dir = OUT_ROOT / tag
        auc     = run_config(cfg, PILOT_EPOCHS, tag, out_dir)
        results.append({'cfg': cfg, 'tag': tag, 'auc': auc, 'out_dir': out_dir})

    # ── summary table ─────────────────────────────────────────────────────
    results.sort(key=lambda x: x['auc'], reverse=True)
    print('\n' + '=' * 60)
    print('PILOT SWEEP RESULTS (sorted by val mean_AUC_f):')
    print(f'  {"rank":>4}  {"lr":>8}  {"lam":>5}  {"AUC_f":>7}')
    for rank, r in enumerate(results, 1):
        print(f'  {rank:>4}  {r["cfg"]["lr"]:>8.0e}  '
              f'{r["cfg"]["lam"]:>5.1f}  {r["auc"]:>7.4f}')
    print('=' * 60, flush=True)

    best = results[0]
    print(f'\n[HPO] Best config: lr={best["cfg"]["lr"]:.0e}  '
          f'lam={best["cfg"]["lam"]}  → AUC_f={best["auc"]:.4f}')

    # save overall summary
    with open(OUT_ROOT / 'sweep_summary.json', 'w') as f:
        json.dump([{'cfg': r['cfg'], 'tag': r['tag'], 'auc': r['auc']}
                   for r in results], f, indent=2)

    # ── Phase 2: full training with best config ────────────────────────────
    print(f'\n[HPO] Starting full {FULL_EPOCHS}-epoch training with best config ...', flush=True)
    full_out = HERE / 'checkpoints_mutualcontrast_best'
    run_config(best['cfg'], FULL_EPOCHS, 'full_best', full_out)

    # ── plot loss curves ───────────────────────────────────────────────────
    _plot_curves(full_out)

    print('\n[HPO] All done.')
    print(f'  Final checkpoint : {full_out}/mutualcontrast_best_full.pth')
    print(f'  Deploy checkpoint: {full_out}/mutualcontrast_best_fundus.pt')
    print(f'  Loss curves PNG  : {full_out}/loss_curves.png')


if __name__ == '__main__':
    main()
