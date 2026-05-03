"""
compare_hpo_results.py  —  Summarise HPO experiments from metrics_history.json

Usage:
  # CMR v3 HPO (Branch 2, 2572):
  python contrastive_pretrain/compare_hpo_results.py \
    --pattern 'contrastive_pretrain/checkpoints_cmr_hpo_*'

  # Fundus-CMR HPO (Branch 1, 2582):
  python contrastive_pretrain/compare_hpo_results.py \
    --pattern 'contrastive_pretrain/checkpoints_fundus_cmr_*'
"""
import argparse, glob, json, pathlib
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pattern', required=True)
    ap.add_argument('--metric', default='mean_auc')
    args = ap.parse_args()

    rows = []
    for d in sorted(glob.glob(args.pattern)):
        hist_f  = pathlib.Path(d) / 'metrics_history.json'
        conf_f  = pathlib.Path(d) / 'config.json'
        if not hist_f.exists():
            print(f'  [skip] no metrics_history.json in {d}')
            continue
        history = json.loads(hist_f.read_text())
        config  = json.loads(conf_f.read_text()) if conf_f.exists() else {}
        best_ep = max(history, key=lambda r: r.get(args.metric, 0))
        rows.append({
            'dir':       pathlib.Path(d).name,
            'best_auc':  best_ep.get(args.metric, float('nan')),
            'best_ep':   best_ep.get('epoch', -1),
            'last_ep':   history[-1].get('epoch', -1),
            'lam1':      config.get('lam1', config.get('lam', '-')),
            'lam2':      config.get('lam2', '-'),
            'lam3':      config.get('lam3', '-'),
            'bs':        f"{config.get('batch_size','-')}/{config.get('unfreeze_batch_size', config.get('batch_size','-'))}",
            'enc_lr':    config.get('enc_lr', config.get('fundus_lr', '-')),
            'depth':     config.get('transformer_depth', '-'),
        })

    if not rows:
        print('No results found.')
        return

    df = pd.DataFrame(rows).sort_values('best_auc', ascending=False)
    pd.set_option('display.max_colwidth', 60)
    pd.set_option('display.width', 160)
    print(df.to_string(index=False))

if __name__ == '__main__':
    main()
