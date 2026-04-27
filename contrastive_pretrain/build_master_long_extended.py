"""Extract visit + event date table from cohort1_participant.csv for all EIDs in
fundus_table_extended.csv.

Output: contrastive_pretrain/raw/master_long_extended.csv

Columns:
  eid
  visit_date_i0 / i1 / i2 / i3        (field 53)
  e10..e14_date                       (diabetes variants)
  i10..i13, i15_date                  (hypertension)
  i20, i21, i22, i24, i25_date        (ischemic heart / MI / CAD)
  i34, i35, i42, i43_date             (valve / cardiomyopathy)
  i47, i48, i49, i50_date             (arrhythmia / HF)

NOTE (known gaps vs DATA_REQUEST_FOR_AGENT.md):
  - stroke_date (I63/I64) -- NOT in cohort1.csv. Document separately.
  - death_date (field 40000) -- NOT in cohort1.csv.
  - censoring_date -- NOT in cohort1.csv.
  These gaps matter for Tier 3 "event_between" filtering; Tier 2 does not need them.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

REPO = Path("/data/home/shujia/CHD/model_train/RETFound_MAE-main")
COHORT1_CSV = REPO / "contrastive_pretrain/raw/cohort1_participant.csv"
EXTENDED_TABLE = REPO / "contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv"
OUT_CSV = REPO / "contrastive_pretrain/raw/master_long_extended.csv"


EVENT_COL_MAP = {
    "Date E10 first reported (insulin-dependent diabetes mellitus)": "e10_date",
    "Date E11 first reported (non-insulin-dependent diabetes mellitus)": "e11_date",
    "Date E12 first reported (malnutrition-related diabetes mellitus)": "e12_date",
    "Date E13 first reported (other specified diabetes mellitus)": "e13_date",
    "Date E14 first reported (unspecified diabetes mellitus)": "e14_date",
    "Date I10 first reported (essential (primary) hypertension)": "i10_date",
    "Date I11 first reported (hypertensive heart disease)": "i11_date",
    "Date I12 first reported (hypertensive renal disease)": "i12_date",
    "Date I13 first reported (hypertensive heart and renal disease)": "i13_date",
    "Date I15 first reported (secondary hypertension)": "i15_date",
    "Date I20 first reported (angina pectoris)": "i20_date",
    "Date I21 first reported (acute myocardial infarction)": "i21_date",
    "Date I22 first reported (subsequent myocardial infarction)": "i22_date",
    "Date I24 first reported (other acute ischaemic heart diseases)": "i24_date",
    "Date I25 first reported (chronic ischaemic heart disease)": "i25_date",
    "Date I34 first reported (nonrheumatic mitral valve disorders)": "i34_date",
    "Date I35 first reported (nonrheumatic aortic valve disorders)": "i35_date",
    "Date I42 first reported (cardiomyopathy)": "i42_date",
    "Date I43 first reported (cardiomyopathy in diseases classified elsewhere)": "i43_date",
    "Date I47 first reported (paroxysmal tachycardia)": "i47_date",
    "Date I48 first reported (atrial fibrillation and flutter)": "i48_date",
    "Date I49 first reported (other cardiac arrhythmias)": "i49_date",
    "Date I50 first reported (heart failure)": "i50_date",
}

VISIT_COL_MAP = {
    "Date of attending assessment centre | Instance 0": "visit_date_i0",
    "Date of attending assessment centre | Instance 1": "visit_date_i1",
    "Date of attending assessment centre | Instance 2": "visit_date_i2",
    "Date of attending assessment centre | Instance 3": "visit_date_i3",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=OUT_CSV)
    ap.add_argument("--all_eids", action="store_true",
                    help="Include all EIDs from cohort1.csv, not just those in extended table.")
    args = ap.parse_args()

    usecols = ["Participant ID"] + list(VISIT_COL_MAP.keys()) + list(EVENT_COL_MAP.keys())
    print(f"[load] reading {len(usecols)} cols from cohort1 ...", flush=True)
    df = pd.read_csv(COHORT1_CSV, usecols=usecols, low_memory=False)
    df = df.rename(columns={"Participant ID": "eid", **VISIT_COL_MAP, **EVENT_COL_MAP})
    df["eid"] = df["eid"].astype(int)
    print(f"[load] cohort1 rows: {len(df)}", flush=True)

    if not args.all_eids:
        ext = pd.read_csv(EXTENDED_TABLE, usecols=["eid"])
        keep = set(ext["eid"].astype(int).tolist())
        df = df[df["eid"].isin(keep)].reset_index(drop=True)
        print(f"[filter] restricted to extended-table EIDs: {len(df)}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"[write] {args.out}  rows={len(df)}", flush=True)

    print("\n=== coverage summary ===")
    for col in ["visit_date_i0", "visit_date_i1", "visit_date_i2", "visit_date_i3"]:
        n = df[col].notna().sum()
        print(f"{col:20s}  {n:>7d} / {len(df)}  ({100.0*n/len(df):.1f}%)")
    print()
    for col in list(EVENT_COL_MAP.values()):
        n = df[col].notna().sum()
        print(f"{col:20s}  {n:>7d} / {len(df)}  ({100.0*n/len(df):.1f}%)")


if __name__ == "__main__":
    main()
