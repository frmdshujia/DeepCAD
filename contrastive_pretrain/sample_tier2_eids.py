"""Sample 5K EIDs for Stage 2 Tier 2 training from fundus_table_extended train split.

Stratified by pair_type coverage (plan A):
  - group "same_any":  EIDs that have at least one same_inst (fundus inst=2 ↔ CMR inst=2) row
  - group "cross_only": EIDs that only have cross_inst (fundus inst=3 ↔ CMR inst=2)
  We keep the same (same:cross) ratio in the sample as in the train pool, so same_inst
  signal is not diluted but we still hit ~5K scale to test scaling.

Output:
  - contrastive_pretrain/stage2_tier2_5k_eids.txt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path("/data/home/shujia/CHD/model_train/RETFound_MAE-main")
EXTENDED_TABLE = REPO / "contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv"
DEFAULT_OUT = REPO / "contrastive_pretrain/stage2_tier2_5k_eids.txt"

SEED = 20260422


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--include_legacy_tier1", action="store_true", default=True,
                    help="Always include the 1196 Tier 1 EIDs (they already have CMR npy preprocessed).")
    args = ap.parse_args()

    df = pd.read_csv(EXTENDED_TABLE)
    train = df[df["split"] == "train"].copy()
    print(f"[load] train rows: {len(train)}, unique eids: {train['eid'].nunique()}", flush=True)

    # per-EID has-same_inst flag
    eid_has_same = train.groupby("eid")["pair_type"].apply(
        lambda s: (s == "same_inst").any()
    )
    same_any_eids = eid_has_same[eid_has_same].index.tolist()
    cross_only_eids = eid_has_same[~eid_has_same].index.tolist()
    print(
        f"[stratify] train: same_any={len(same_any_eids)}  cross_only={len(cross_only_eids)}",
        flush=True,
    )

    rng = np.random.default_rng(SEED)

    # seed set: Tier 1 pool that already has CMR npy preprocessed (saves preproc time)
    already_done = set()
    tier1_txt = REPO / "contrastive_pretrain/stage2_tier1_1196_eids.txt"
    tier0_txt = REPO / "contrastive_pretrain/stage2_tier0_eids.txt"
    if args.include_legacy_tier1 and tier1_txt.is_file():
        already_done.update(int(x) for x in tier1_txt.read_text().split())
        if tier0_txt.is_file():
            already_done.update(int(x) for x in tier0_txt.read_text().split())
        print(f"[seed] tier1+tier0 EIDs (will pre-include any that fall in train split): {len(already_done)}")

    train_eid_set = set(train["eid"].tolist())
    seed_in_train = [e for e in already_done if e in train_eid_set]
    seed_same = [e for e in seed_in_train if e in set(same_any_eids)]
    seed_cross = [e for e in seed_in_train if e in set(cross_only_eids)]
    print(f"[seed] in-train seed: same={len(seed_same)}  cross={len(seed_cross)}")

    # target count per group -- keep same/cross ratio in train pool
    total_train = len(same_any_eids) + len(cross_only_eids)
    frac_same = len(same_any_eids) / total_train
    n_same_target = int(round(args.n * frac_same))
    n_cross_target = args.n - n_same_target
    print(f"[target] same={n_same_target}  cross={n_cross_target}")

    # sample remainder (after seeds) from each group
    same_pool = [e for e in same_any_eids if e not in seed_same]
    cross_pool = [e for e in cross_only_eids if e not in seed_cross]

    n_same_more = max(0, n_same_target - len(seed_same))
    n_cross_more = max(0, n_cross_target - len(seed_cross))
    n_same_more = min(n_same_more, len(same_pool))
    n_cross_more = min(n_cross_more, len(cross_pool))

    same_sorted = np.array(sorted(same_pool), dtype=np.int64)
    cross_sorted = np.array(sorted(cross_pool), dtype=np.int64)
    same_pick = rng.choice(same_sorted, size=n_same_more, replace=False)
    cross_pick = rng.choice(cross_sorted, size=n_cross_more, replace=False)

    picked = (
        list(seed_same) + list(same_pick) + list(seed_cross) + list(cross_pick)
    )
    picked = sorted(set(int(x) for x in picked))
    print(f"[pick] total unique EIDs: {len(picked)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for e in picked:
            f.write(f"{e}\n")
    print(f"[write] {args.out}  count={len(picked)}")

    # summary
    picked_set = set(picked)
    sub = train[train["eid"].isin(picked_set)]
    print("\n=== Tier 2 pick summary ===")
    print(f"unique EIDs            : {len(picked)}")
    print(f"total fundus rows      : {len(sub)}")
    print(f"pair_type distribution :")
    print(sub["pair_type"].value_counts())
    print(f"\nvisit_interval_years describe:")
    print(sub["visit_interval_years"].describe())


if __name__ == "__main__":
    main()
