"""
Subgroup AUC analysis for CMR model trained with composite_ischemic_hd.

For each ICD subgroup (I20, I21, I25):
  - Positives : patients with that specific ICD code
  - Negatives : healthy controls (composite_ischemic_hd == 0)
  - Excluded  : composite=1 but that specific code=0 (other subtypes)

Usage:
  python contrastive_pretrain/eval_subgroup_cmr.py \
    --ckpt contrastive_pretrain/checkpoints_cmr_composite_I25/best.pth \
    --n_cls 2
"""
import argparse, pathlib, sys
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from contrastive_pretrain.models_cmr_v3 import CMREncoderV3, NUM_FRAMES
from contrastive_pretrain.models_dualtower import TaskHead

TEST_CSV   = str(ROOT / 'contrastive_pretrain/task_reports/task1_cmr_test.csv')
CMR_V3_DIR = '/data/home/shujia/UKB/CMRI/preprocessed_cmr_v3'
MEDSAM_CKPT = str(ROOT / 'pretrained_weights/hf_cache/'
    'models--flaviagiammarino--medsam-vit-base/blobs/'
    'b80a96478503f89e76f1f7bbba50cfcd4ec9e7467f0d5185310216b33946ec9c')

SUBGROUPS = ['prevalent_I20', 'prevalent_I21', 'prevalent_I25']

def load_npy(path):
    x = np.load(path).astype(np.float32)   # (16, H, W) or (16, 224, 224)
    return torch.from_numpy(x).unsqueeze(0)  # (1, 16, H, W)

@torch.no_grad()
def run_inference(encoder, head, df, device, batch_size=32):
    encoder.eval(); head.eval()
    all_scores, all_labels = [], {col: [] for col in ['composite_ischemic_hd'] + SUBGROUPS}

    paths = df['path'].tolist()
    for start in range(0, len(paths), batch_size):
        batch_paths = paths[start:start + batch_size]
        tensors = []
        for p in batch_paths:
            try:
                tensors.append(load_npy(p))
            except Exception:
                tensors.append(torch.zeros(1, NUM_FRAMES, 224, 224))
        x = torch.cat(tensors, dim=0).to(device)           # (B, 16, 224, 224)
        _, cls_out   = encoder(x)                           # cls_out: (B, 768)
        cls_list, _  = head(cls_out)                        # list of (B,) tensors
        scores       = torch.sigmoid(cls_list[0]).cpu().numpy()  # composite head
        all_scores.extend(scores.tolist())

    for col in ['composite_ischemic_hd'] + SUBGROUPS:
        all_labels[col] = df[col].fillna(0).astype(int).tolist()

    return np.array(all_scores), all_labels

def subgroup_auc(scores, labels_dict):
    composite = np.array(labels_dict['composite_ischemic_hd'])
    neg_mask  = (composite == 0)

    results = {}
    for col in SUBGROUPS:
        icd = np.array(labels_dict[col])
        pos_mask = (icd == 1)
        mask = pos_mask | neg_mask          # keep this subgroup + healthy controls
        y    = icd[mask]
        s    = scores[mask]
        if y.sum() < 5:
            results[col] = float('nan')
            continue
        results[col] = roc_auc_score(y, s)

    # overall composite AUC for reference
    results['composite_overall'] = roc_auc_score(composite, scores)
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',  type=str,
                        default='contrastive_pretrain/checkpoints_cmr_composite_I25/best.pth')
    parser.add_argument('--n_cls', type=int, default=2,
                        help='Number of cls heads in the checkpoint (2 for composite+one ICD)')
    parser.add_argument('--spatial_pool', type=int, default=4)
    parser.add_argument('--transformer_depth', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    encoder = CMREncoderV3(
        proj_dim=256, embed_dim=768,
        spatial_pool=args.spatial_pool,
        transformer_heads=8, transformer_depth=args.transformer_depth,
        medsam_ckpt=None, freeze_backbone=True,
    ).to(device)
    head = TaskHead(in_dim=768, n_cls=args.n_cls, n_reg=3).to(device)

    ckpt = torch.load(args.ckpt, map_location=device)
    encoder.load_state_dict(ckpt['encoder'])
    head.load_state_dict(ckpt['head'])
    print(f'[loaded] {args.ckpt}')

    df = pd.read_csv(TEST_CSV)
    # Override paths to v3 preprocessed npy (16-frame)
    v3_dir = pathlib.Path(CMR_V3_DIR)
    df['path'] = df['eid'].apply(lambda e: str(v3_dir / f'{e}.npy'))
    available = df['path'].apply(lambda p: pathlib.Path(p).exists())
    df = df[available].reset_index(drop=True)
    print(f'[test]  {len(df)} samples with v3 npy available')
    for col in SUBGROUPS:
        n = df[col].fillna(0).astype(int).sum()
        print(f'  {col}: {n} positives')

    scores, labels_dict = run_inference(encoder, head, df, device)

    results = subgroup_auc(scores, labels_dict)

    print('\n===== Subgroup AUC (composite head, vs healthy controls) =====')
    print(f"  composite_overall  AUC = {results['composite_overall']:.4f}  "
          f"(n_pos={int(np.array(labels_dict['composite_ischemic_hd']).sum())})")
    for col in SUBGROUPS:
        icd    = np.array(labels_dict[col])
        n_pos  = int((icd & (np.array(labels_dict['composite_ischemic_hd']) == 1)).sum())
        auc_v  = results[col]
        bar    = '█' * int(auc_v * 20) if not np.isnan(auc_v) else ''
        print(f"  {col:<20} AUC = {auc_v:.4f}  (n_pos={n_pos})  {bar}")

    best  = max(SUBGROUPS, key=lambda c: results[c])
    worst = min(SUBGROUPS, key=lambda c: results[c])
    print(f'\n  Best  subgroup: {best}  ({results[best]:.4f})')
    print(f'  Worst subgroup: {worst}  ({results[worst]:.4f})')

if __name__ == '__main__':
    main()
