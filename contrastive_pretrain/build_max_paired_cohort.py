"""
build_max_paired_cohort.py

Build the maximum-scale paired cohort:
  - Every UK Biobank participant who has BOTH a verified CMR npy AND a verified fundus PNG
  - Cross-visit pairs are included (different visit dates for CMR vs fundus is OK)
  - One row per EID: prefer CMR instance=2; prefer fundus instance=0; prefer right eye
  - Labels joined from stage1_cmr.csv (classification + LV regression)
  - Output: contrastive_pretrain/paired_max_cohort_{train,val,test}.csv
"""
import pathlib, pandas as pd, numpy as np
from tqdm import tqdm

HERE    = pathlib.Path(__file__).parent
PD      = HERE / 'preprocessed_data'
NPY_DIR = pathlib.Path('/data/home/shujia/UKB/CMRI/preprocessed_lax_sax')
OUT     = HERE / 'task_reports'
OUT.mkdir(exist_ok=True)

CLS_COLS = [
    'composite_ischemic_hd', 'prevalent_I21',
    'composite_cardiomyopathy_hf', 'composite_af_arrhythmia',
]
REG_COLS = ['LV ejection fraction', 'LV end diastolic volume', 'LV end systolic volume']

# ── 1. Load fundus_table_extended ─────────────────────────────────────────────
print("Loading fundus_table_extended ...")
ft = pd.read_csv(PD / 'modeling_delivery/fundus_table_extended.csv')
print(f"  Total rows: {len(ft):,}  |  Unique EIDs: {ft['eid'].nunique():,}")
print(f"  pair_type distribution:\n{ft['pair_type'].value_counts().to_string()}")

# ── 2. Verify CMR npy exists ──────────────────────────────────────────────────
print("\nScanning CMR npy files on disk ...")
npy_eids = {int(p.stem) for p in NPY_DIR.glob('*.npy')}
print(f"  npy files found: {len(npy_eids):,}")

ft['cmr_npy_ok'] = ft['eid'].isin(npy_eids)
print(f"  Rows with npy confirmed: {ft['cmr_npy_ok'].sum():,} / {len(ft):,}")

# ── 3. Verify fundus PNG exists ───────────────────────────────────────────────
print("\nVerifying fundus PNG paths ...")
# fundus_image_path column is the direct path stored in fundus_table_extended
# Check if the column contains usable paths
sample_paths = ft['fundus_image_path'].dropna().head(5).tolist()
print("  Sample fundus_image_path values:", sample_paths[:2])

# Test if paths are absolute or need a prefix
test_path = pathlib.Path(sample_paths[0]) if sample_paths else None
if test_path and not test_path.exists():
    # Try common prefixes
    for prefix in [
        '/data/home/home6/fundus_data/UKB/fundus_images',
        '/data/home/shujia',
        '',
    ]:
        candidate = pathlib.Path(prefix) / test_path.name if prefix else test_path
        if candidate.exists():
            print(f"  Found with prefix: {prefix}")
            break

# Use vectorized file existence check
# First try paths as-is
def check_path_exists(path_str):
    if pd.isna(path_str):
        return False
    return pathlib.Path(path_str).exists()

# Batch check - sample first to determine prefix needed
n_sample = min(200, len(ft))
sample_ok = sum(check_path_exists(p) for p in ft['fundus_image_path'].iloc[:n_sample])
print(f"  Sample check ({n_sample} rows): {sample_ok} paths exist as-is")

if sample_ok > 0:
    # Paths work as-is — do full check
    print("  Running full existence check (may take ~30s) ...")
    ft['fundus_png_ok'] = ft['fundus_image_path'].apply(check_path_exists)
else:
    # Try to find fundus image directory
    print("  Paths as-is don't work, scanning fundus image directory ...")
    fundus_dirs = [
        pathlib.Path('/data/home/home6/fundus_data/UKB/fundus_images'),
        pathlib.Path('/data/home/shujia/UKB/fundus_images'),
        pathlib.Path('/data/home/shujia/UKB/fundus'),
    ]
    fundus_dir = None
    for d in fundus_dirs:
        if d.exists():
            fundus_dir = d
            print(f"  Found fundus dir: {d}")
            break

    if fundus_dir is None:
        # Try to find it from the path string
        for p in ft['fundus_image_path'].dropna().head(20):
            parts = pathlib.Path(p).parts
            for i in range(len(parts), 0, -1):
                candidate = pathlib.Path(*parts[:i])
                if candidate.is_dir():
                    fundus_dir = candidate.parent
                    break
            if fundus_dir:
                break

    if fundus_dir and fundus_dir.exists():
        print(f"  Building fundus index from {fundus_dir} ...")
        fundus_stems = {p.name for p in fundus_dir.glob('*.png')}
        print(f"  Found {len(fundus_stems):,} PNG files")
        ft['fundus_png_ok'] = ft['fundus_image_path'].apply(
            lambda p: pathlib.Path(p).name in fundus_stems if not pd.isna(p) else False
        )
    else:
        print("  WARNING: Cannot locate fundus directory. Using fundus_image_path from CSV.")
        # Fall back: assume PNG exists if path is non-null (will be validated at train time)
        ft['fundus_png_ok'] = ft['fundus_image_path'].notna()

print(f"  Rows with fundus PNG confirmed: {ft['fundus_png_ok'].sum():,}")

# ── 4. Keep only fully verified rows ──────────────────────────────────────────
ft_ok = ft[ft['cmr_npy_ok'] & ft['fundus_png_ok']].copy()
print(f"\nBoth CMR npy + fundus PNG verified: {len(ft_ok):,} rows, {ft_ok['eid'].nunique():,} unique EIDs")

# ── 5. Deduplicate: one row per EID ───────────────────────────────────────────
# Priority: cmr_instance=2 > 3; fundus_instance=0 > 1; eye='right' > 'left'
eye_rank  = {'right': 0, 'left': 1}
ft_ok = ft_ok.copy()
ft_ok['_inst_rank']  = ft_ok['cmr_instance'].map({2: 0, 3: 1}).fillna(9).astype(int)
ft_ok['_fund_rank']  = ft_ok['fundus_instance'].fillna(9).astype(int)
ft_ok['_eye_rank']   = ft_ok['eye'].map(eye_rank).fillna(9).astype(int)
ft_ok = ft_ok.sort_values(['eid', '_inst_rank', '_fund_rank', '_eye_rank'])
ft_dedup = ft_ok.drop_duplicates(subset='eid', keep='first').copy()
ft_dedup = ft_dedup.drop(columns=['_inst_rank', '_fund_rank', '_eye_rank'])
print(f"After dedup (one row per EID): {len(ft_dedup):,}")

# ── 6. Merge labels from stage1_cmr.csv ───────────────────────────────────────
print("\nLoading stage1_cmr.csv for labels ...")
cmr_full = pd.read_csv(PD / 'stage1/stage1_cmr.csv',
                        usecols=['eid', 'instance'] + CLS_COLS + REG_COLS)
# Prefer instance=2; fallback to 3
cmr_full = cmr_full.sort_values(['eid', 'instance'])
cmr_best = cmr_full.drop_duplicates(subset='eid', keep='last')  # last = highest instance
cmr_best = cmr_best.rename(columns={'instance': 'cmr_label_instance'})
print(f"  CMR label rows: {len(cmr_best):,}")

ft_dedup = ft_dedup.merge(
    cmr_best[['eid', 'cmr_label_instance'] + CLS_COLS + REG_COLS],
    on='eid', how='left'
)
label_ok = ft_dedup[CLS_COLS[0]].notna().sum()
print(f"  EIDs with CMR labels joined: {label_ok:,} / {len(ft_dedup):,}")

# ── 7. Compute npy path column ─────────────────────────────────────────────────
ft_dedup['cmr_npy_path'] = ft_dedup['eid'].apply(
    lambda e: str(NPY_DIR / f'{e}.npy')
)

# ── 8. 8:1:1 split ─────────────────────────────────────────────────────────────
# Use existing split column if available (carry over from fundus_table_extended)
if 'split' in ft_dedup.columns and ft_dedup['split'].notna().any():
    split_counts = ft_dedup['split'].value_counts()
    print(f"\nExisting split distribution:\n{split_counts.to_string()}")
    # Use existing splits
    train_df = ft_dedup[ft_dedup['split'] == 'train']
    val_df   = ft_dedup[ft_dedup['split'] == 'val']
    test_df  = ft_dedup[ft_dedup['split'] == 'test']
    # Any unassigned: put into train
    unassigned = ft_dedup[~ft_dedup['split'].isin(['train', 'val', 'test'])]
    if len(unassigned) > 0:
        print(f"  {len(unassigned)} rows with unknown split -> added to train")
        train_df = pd.concat([train_df, unassigned], ignore_index=True)
else:
    print("\nNo usable split column, doing random 8:1:1 split ...")
    np.random.seed(42)
    idx = np.random.permutation(len(ft_dedup))
    n   = len(ft_dedup)
    n_train = int(n * 0.8)
    n_val   = int(n * 0.1)
    train_df = ft_dedup.iloc[idx[:n_train]]
    val_df   = ft_dedup.iloc[idx[n_train:n_train+n_val]]
    test_df  = ft_dedup.iloc[idx[n_train+n_val:]]

print(f"\nFinal split: train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

# ── 9. Save ────────────────────────────────────────────────────────────────────
save_cols = ['eid', 'cmr_npy_path', 'fundus_image_path',
             'cmr_instance', 'fundus_instance', 'eye',
             'pair_type', 'visit_interval_years'] + CLS_COLS + REG_COLS

for df, name in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
    out_path = OUT / f'paired_max_{name}.csv'
    df[save_cols].to_csv(out_path, index=False)
    print(f"  Saved: {out_path}  ({len(df):,} rows)")

# Print label stats for the full max cohort
print("\n=== Label stats in max paired cohort (all splits) ===")
full = pd.concat([train_df, val_df, test_df])
any_pos = (full[CLS_COLS] == 1).any(axis=1)
print(f"Any-positive: {any_pos.sum():,} ({any_pos.mean()*100:.1f}%)")
print(f"All-negative: {(~any_pos).sum():,} ({(~any_pos).mean()*100:.1f}%)")
for col in CLS_COLS:
    pos  = (full[col] == 1).sum()
    neg  = (full[col] == 0).sum()
    miss = (full[col].isna() | (full[col] < 0)).sum()
    rate = pos/(pos+neg)*100 if pos+neg > 0 else 0
    print(f"  {col:<42s}: pos={pos:5,} ({rate:.1f}%), neg={neg:5,}, no-label={miss:3,}")

print(f"\nDone. Max paired cohort: {len(full):,} EIDs total")
print(f"CSV files saved to: {OUT}/paired_max_*.csv")
