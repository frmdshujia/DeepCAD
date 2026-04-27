"""Build fundus_table_extended.csv for Stage 2 Tier 2 (+Tier 3 prep).

Pair policy (per user decision 2026-04-22):
  - same-instance: fundus_instance == 2 and CMR_instance == 2
  - plus: fundus_instance == 3 and CMR_instance == 2  (late fundus -> early CMR)

Split policy:
  - Inherit the 1983 legacy-EID splits from fundus_table.csv (rigid).
  - New EIDs are split 80/10/10 by EID with seed=20260421.
  - Same-EID rows always share a split.

Filter:
  - fundus_image_path must exist on disk.
  - Both {eid}_20208_2_0.zip (LAX) and {eid}_20209_2_0.zip (SAX) must exist.

Output:
  - contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv

Also prints a sanity summary (counts, per-split / per-pair_type / per-(f_inst, c_inst)).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


FUNDUS_DIR = Path("/data/home/home6/fundus_data/UKB/fundus_images")
CMR_ZIP_DIR = Path("/data/home/shujia/UKB/CMRI/downloaded")

REPO = Path("/data/home/shujia/CHD/model_train/RETFound_MAE-main")
LEGACY_TABLE = REPO / "contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table.csv"
COHORT1_CSV = REPO / "contrastive_pretrain/raw/cohort1_participant.csv"

OUT_PATH = REPO / "contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv"

# UKB field 21015 = left eye, 21016 = right eye
FIELD_LEFT = 21015
FIELD_RIGHT = 21016

FNAME_RE = re.compile(r"^(\d+)_(\d+)_(\d+)_(\d+)\.png$")

NEW_EID_SPLIT_SEED = 20260421


def scan_fundus(fundus_dir: Path, want_instances=(2, 3)) -> pd.DataFrame:
    """Scan fundus png directory and return one row per existing file.

    Columns: eid, fundus_field, fundus_instance, fundus_array, eye, fundus_image_path
    """
    rows = []
    for name in os.listdir(fundus_dir):
        m = FNAME_RE.match(name)
        if m is None:
            continue
        eid = int(m.group(1))
        field = int(m.group(2))
        instance = int(m.group(3))
        array = int(m.group(4))
        if instance not in want_instances:
            continue
        if field == FIELD_LEFT:
            eye = "left"
        elif field == FIELD_RIGHT:
            eye = "right"
        else:
            continue
        rows.append((eid, field, instance, array, eye, str(fundus_dir / name)))
    df = pd.DataFrame(
        rows,
        columns=[
            "eid",
            "fundus_field",
            "fundus_instance",
            "fundus_array",
            "eye",
            "fundus_image_path",
        ],
    )
    return df


def check_cmr_pair(eid: int, cmr_dir: Path, instance: int = 2) -> bool:
    lax = cmr_dir / f"{eid}_20208_{instance}_0.zip"
    sax = cmr_dir / f"{eid}_20209_{instance}_0.zip"
    return lax.is_file() and sax.is_file()


def load_visit_dates(cohort_csv: Path) -> pd.DataFrame:
    """Return DataFrame indexed by eid with columns visit_date_i0..i3."""
    usecols = [
        "Participant ID",
        "Date of attending assessment centre | Instance 0",
        "Date of attending assessment centre | Instance 1",
        "Date of attending assessment centre | Instance 2",
        "Date of attending assessment centre | Instance 3",
    ]
    df = pd.read_csv(cohort_csv, usecols=usecols, low_memory=False)
    df = df.rename(columns={
        "Participant ID": "eid",
        "Date of attending assessment centre | Instance 0": "visit_date_i0",
        "Date of attending assessment centre | Instance 1": "visit_date_i1",
        "Date of attending assessment centre | Instance 2": "visit_date_i2",
        "Date of attending assessment centre | Instance 3": "visit_date_i3",
    })
    df["eid"] = df["eid"].astype(int)
    return df.set_index("eid")


def assign_new_eid_split(new_eids: np.ndarray, seed: int = NEW_EID_SPLIT_SEED) -> Dict[int, str]:
    """Split new EIDs 80/10/10 by EID with a fixed seed."""
    rng = np.random.default_rng(seed)
    order = np.array(new_eids, dtype=np.int64)
    order = order[np.argsort(order)]  # stable ordering
    perm = rng.permutation(len(order))
    shuffled = order[perm]
    n = len(shuffled)
    n_train = int(round(n * 0.8))
    n_val = int(round(n * 0.1))
    n_test = n - n_train - n_val
    train_ids = shuffled[:n_train]
    val_ids = shuffled[n_train : n_train + n_val]
    test_ids = shuffled[n_train + n_val :]
    mapping: Dict[int, str] = {}
    for eid in train_ids:
        mapping[int(eid)] = "train"
    for eid in val_ids:
        mapping[int(eid)] = "val"
    for eid in test_ids:
        mapping[int(eid)] = "test"
    return mapping


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument(
        "--pair_instances",
        type=str,
        default="2,3",
        help="comma-separated fundus_instance list to keep (CMR inst fixed at 2)",
    )
    ap.add_argument("--cmr_instance", type=int, default=2)
    args = ap.parse_args()

    pair_instances = tuple(int(x) for x in args.pair_instances.split(","))

    # 1) Scan disk
    print(f"[scan] fundus dir: {FUNDUS_DIR}", flush=True)
    df_img = scan_fundus(FUNDUS_DIR, want_instances=pair_instances)
    print(f"[scan] fundus rows (inst {pair_instances}): {len(df_img)}", flush=True)
    print(f"[scan] unique eids with any fundus inst {pair_instances}: {df_img['eid'].nunique()}", flush=True)

    # 2) CMR zip existence filter -- keep only EIDs with BOTH LAX+SAX at cmr_instance
    print(f"[scan] checking CMR zip existence (inst={args.cmr_instance}) ...", flush=True)
    unique_eids = df_img["eid"].unique()
    eid_has_cmr = {eid: check_cmr_pair(eid, CMR_ZIP_DIR, args.cmr_instance) for eid in unique_eids}
    keep_eids = {eid for eid, ok in eid_has_cmr.items() if ok}
    print(f"[scan] EIDs with both LAX+SAX zip at inst={args.cmr_instance}: {len(keep_eids)}", flush=True)
    df_img = df_img[df_img["eid"].isin(keep_eids)].copy()
    print(f"[scan] fundus rows after CMR-zip filter: {len(df_img)}", flush=True)

    # 3) Visit dates
    print("[scan] loading visit dates from cohort1 ...", flush=True)
    visits = load_visit_dates(COHORT1_CSV)
    print(f"[scan] cohort1 rows with eid: {len(visits)}", flush=True)

    def _get_date(eid, inst):
        try:
            v = visits.at[eid, f"visit_date_i{inst}"]
            if pd.isna(v):
                return None
            return str(v)
        except KeyError:
            return None

    # 4) Build output rows
    out_rows = []
    cmr_inst = args.cmr_instance
    for row in df_img.itertuples(index=False):
        eid = int(row.eid)
        f_inst = int(row.fundus_instance)
        eye = row.eye
        arr = int(row.fundus_array)
        path = row.fundus_image_path

        vd_f = _get_date(eid, f_inst)
        vd_c = _get_date(eid, cmr_inst)
        interval_years = None
        if vd_f is not None and vd_c is not None:
            try:
                d_f = pd.Timestamp(vd_f)
                d_c = pd.Timestamp(vd_c)
                interval_years = (d_c - d_f).days / 365.25
            except Exception:
                interval_years = None

        pair_type = "same_inst" if f_inst == cmr_inst else "cross_inst"
        out_rows.append({
            "eid": eid,
            "fundus_instance": f_inst,
            "fundus_array": arr,
            "eye": eye,
            "fundus_image_path": path,
            "cmr_instance": cmr_inst,
            "cmr_lax_zip": str(CMR_ZIP_DIR / f"{eid}_20208_{cmr_inst}_0.zip"),
            "cmr_sax_zip": str(CMR_ZIP_DIR / f"{eid}_20209_{cmr_inst}_0.zip"),
            "pair_type": pair_type,
            "visit_date_fundus": vd_f,
            "visit_date_cmr": vd_c,
            "visit_interval_years": interval_years,
        })

    out = pd.DataFrame(out_rows)
    print(f"[build] total rows: {len(out)}, unique eids: {out['eid'].nunique()}", flush=True)

    # 5) Splits — inherit legacy, new EIDs 80/10/10
    print("[split] loading legacy splits ...", flush=True)
    legacy = pd.read_csv(LEGACY_TABLE, usecols=["eid", "split"])
    legacy["eid"] = legacy["eid"].astype(int)
    legacy_eid_split = legacy.drop_duplicates("eid").set_index("eid")["split"].to_dict()
    print(f"[split] legacy unique eids: {len(legacy_eid_split)}", flush=True)

    all_eids = out["eid"].unique()
    legacy_overlap = [e for e in all_eids if e in legacy_eid_split]
    new_eids = np.array([e for e in all_eids if e not in legacy_eid_split], dtype=np.int64)
    print(
        f"[split] legacy overlap: {len(legacy_overlap)}, new eids to split: {len(new_eids)}",
        flush=True,
    )

    new_split_map = assign_new_eid_split(new_eids, seed=NEW_EID_SPLIT_SEED)

    def _split_for(eid: int) -> str:
        if eid in legacy_eid_split:
            return legacy_eid_split[eid]
        return new_split_map[eid]

    out["split"] = out["eid"].map(_split_for)

    # 6) Write
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out = out.sort_values(["eid", "fundus_instance", "eye", "fundus_array"]).reset_index(drop=True)
    out.to_csv(args.out, index=False)
    print(f"[write] {args.out}  rows={len(out)}", flush=True)

    # 7) Sanity summary
    print("\n=== sanity summary ===")
    print(f"rows            = {len(out)}")
    print(f"unique eids     = {out['eid'].nunique()}")
    print("\nsplit × unique eids:")
    print(out.drop_duplicates("eid")["split"].value_counts())
    print("\nsplit × rows:")
    print(out["split"].value_counts())
    print("\npair_type × rows:")
    print(out["pair_type"].value_counts())
    print("\n(fundus_instance, cmr_instance) × rows:")
    print(out.groupby(["fundus_instance", "cmr_instance"]).size())
    print("\nvisit_interval_years describe:")
    print(out["visit_interval_years"].describe())

    # check same-eid invariants
    by_eid = out.groupby("eid")["split"].nunique()
    bad = int((by_eid > 1).sum())
    print(f"\nsame-eid single-split invariant  ok={bad==0}  (violations={bad})")

    # check legacy splits unchanged
    legacy_df = pd.read_csv(LEGACY_TABLE, usecols=["eid", "split"]).drop_duplicates("eid")
    new_df = out.drop_duplicates("eid")[["eid", "split"]]
    merged = legacy_df.merge(new_df, on="eid", suffixes=("_legacy", "_new"))
    changed = int((merged["split_legacy"] != merged["split_new"]).sum())
    print(f"legacy split unchanged invariant ok={changed==0}  (changed={changed} of {len(merged)} overlap)")


if __name__ == "__main__":
    main()
