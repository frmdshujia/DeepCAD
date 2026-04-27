"""
统计需求总表 1–7。数据源（均相对 contrastive_pretrain/preprocessed_data）:
  stage1/stage1_fundus.csv
  stage2/cmr/stage2_cmr.csv  ← CMR 人群、LAX+SAX、表3–5 的 CMR 与「配对」标签均据此表（instance=2 去重）
  stage2/fundus/stage2_fundus_dual.csv
  modeling_delivery/fundus_table_extended.csv
  raw/master_long.csv（表6）
  磁盘: UKB/CMRI/downloaded, preprocessed_lax_sax
"""
from __future__ import annotations

import glob
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PD = os.path.join(REPO, "contrastive_pretrain", "preprocessed_data")
RAW = os.path.join(REPO, "contrastive_pretrain", "raw")
CMR_ZIP = "/data/home/shujia/UKB/CMRI/downloaded"
NPY_DIR = "/data/home/shujia/UKB/CMRI/preprocessed_lax_sax"

ST1_FU = os.path.join(PD, "stage1", "stage1_fundus.csv")
ST2_CMR = os.path.join(PD, "stage2", "cmr", "stage2_cmr.csv")
ST2_FU = os.path.join(PD, "stage2", "fundus", "stage2_fundus_dual.csv")
EXT = os.path.join(PD, "modeling_delivery", "fundus_table_extended.csv")
MASTER = os.path.join(RAW, "master_long.csv")
# 备选：部分环境 raw 未入仓
for _alt in (os.path.join(REPO, "contrastive_pretrain", "raw", "master_long.csv"),):
    if os.path.isfile(_alt):
        MASTER = _alt
        break

COL_L = "Fundus retinal eye image (left)"
COL_R = "Fundus retinal eye image (right)"
SAX = "Short axis heart images - DICOM"
LAX = "Long axis heart images - DICOM"
LVEF, LVEDV, LVESV = "LV ejection fraction", "LV end diastolic volume", "LV end systolic volume"
M_LVEA = "LV ejection fraction | Instance 2(participant - p24103_i2)"
M_LVEB = "LV ejection fraction | Instance 2(participant - p22420_i2)"
M_LVD1 = "LV end diastolic volume | Instance 2(participant - p24100_i2)"
M_LVD2 = "LV end diastolic volume | Instance 2(participant - p22421_i2)"

CLASS_TASKS = [
    "composite_ischemic_hd",
    "prevalent_I21",
    "composite_cardiomyopathy_hf",
    "composite_af_arrhythmia",
    "prevalent_I25",
    "prevalent_I50",
    "composite_hypertensive_hd",
]


def inst2_dedup(df: pd.DataFrame) -> pd.DataFrame:
    if "instance" not in df.columns:
        return df.drop_duplicates("eid").reset_index(drop=True)
    s = df[df["instance"] == 2].copy()
    return s.drop_duplicates("eid", keep="first").reset_index(drop=True)


def fundus_prefer_inst2(df: pd.DataFrame) -> pd.DataFrame:
    """每人一行：优先 instance=2，否则保留首次出现。"""
    if "instance" not in df.columns:
        return df.drop_duplicates("eid").reset_index(drop=True)
    d = df.copy()
    d["_p"] = (d["instance"] == 2).astype(int)
    return d.sort_values(["eid", "_p"], ascending=[True, False]).drop_duplicates("eid").drop(columns=["_p"])


def pos_rate(s: pd.Series) -> tuple[int, int, float]:
    s = pd.to_numeric(s, errors="coerce")
    valid = s.notna()
    n = int(valid.sum())
    if n == 0:
        return 0, 0, 0.0
    s2 = s[valid]
    pos = int((s2 >= 0.5).sum())
    return pos, n, 100.0 * pos / n


def fmt_pr(pos: int, n: int, rate: float) -> str:
    if n == 0:
        return "0 / 0 / —"
    return f"{pos} / {n} / {rate:.2f}%"


def run_all():
    os.chdir(REPO)
    print("REPO =", REPO, "\n")

    # 预加载
    print("读表…", flush=True)
    fu = pd.read_csv(ST1_FU, low_memory=False)
    st2c = pd.read_csv(ST2_CMR, low_memory=False)
    ex = pd.read_csv(EXT, low_memory=False) if os.path.isfile(EXT) else None
    eid_ext = set(ex["eid"].unique()) if ex is not None else set()

    fu_u = fundus_prefer_inst2(fu)
    st2c_i2 = inst2_dedup(st2c)
    pair = st2c_i2[st2c_i2["eid"].isin(eid_ext)].copy() if eid_ext else st2c_i2.iloc[0:0]

    # ----- 表1 -----
    print("=" * 80)
    print("表1：各模态人群基础规模")
    print("=" * 80)
    n_fu = fu["eid"].nunique()
    n_cmr = st2c["eid"].nunique()
    # Short axis = SAX(20209), Long axis = LAX(20208)
    sax_ok = st2c[SAX].notna() & (st2c[SAX].astype(str).str.strip().str.len() > 0)
    lax_ok = st2c[LAX].notna() & (st2c[LAX].astype(str).str.strip().str.len() > 0)
    m_both = st2c[sax_ok & lax_ok]
    n_laxsax = m_both["eid"].nunique()
    if COL_L in fu.columns and COL_R in fu.columns:
        g = fu.groupby("eid").agg({COL_L: "max", COL_R: "max"})
        n_left = int((g[COL_L] == 1).sum())
        n_right = int((g[COL_R] == 1).sum())
        n_both_row = int(fu[(fu[COL_L] == 1) & (fu[COL_R] == 1)]["eid"].nunique())
    else:
        n_left = n_right = n_both_row = -1
    print(f"眼底人群唯一EID (stage1/stage1_fundus.csv)                    {n_fu}")
    print(f"CMR人群唯一EID (stage2/cmr/stage2_cmr.csv)                    {n_cmr}")
    print(f"同时LAX+SAX zip 路径非空 (stage2_cmr) 的 EID 数                 {n_laxsax}")
    print(
        f"有左眼(任一行{COL_L}==1) 的EID                            {n_left}\n"
        f"有右眼(任一行{COL_R}==1) 的EID                            {n_right}\n"
        f"至少一行左右眼同时==1 的EID                                 {n_both_row}"
    )
    st2e = 0
    if os.path.isfile(ST2_FU):
        st2e = pd.read_csv(ST2_FU, usecols=["eid"])["eid"].nunique()
    print(
        f"\n(备注) fundus_table_extended 唯一EID(全量可配对) {len(eid_ext)}  |  "
        f"stage2 交叉子表唯一EID {st2e}"
    )

    # ----- 表2 -----
    print("\n" + "=" * 80)
    print("表2：配对人群规模 (fundus_table_extended)")
    print("=" * 80)
    if ex is None:
        print("无 extended 表，跳过表2。")
    else:
        dfn = ex.dropna(subset=["visit_date_fundus", "visit_date_cmr"])
        ex1 = dfn.drop_duplicates(subset=["eid", "fundus_instance", "cmr_instance"], keep="first")
        vi = pd.to_numeric(ex1["visit_interval_years"], errors="coerce").dropna()
        if len(vi) > 0:
            print(
                f"唯一EID(全表) {ex['eid'].nunique()}  行数(含多眼) {len(ex)}"
            )
            q1, med, q3 = float(vi.quantile(0.25)), float(vi.quantile(0.5)), float(vi.quantile(0.75))
            print(
                f"visit 间隔(年)  中位 {med:.3f}  P25 {q1:.3f}  P75 {q3:.3f}  最小 {float(vi.min()):.3f}  最大 {float(vi.max()):.3f}"
            )
            ab = vi.abs()
            ntot = len(ab)
            le2 = int((ab <= 2).sum())
            b25 = int(((ab > 2) & (ab <= 5)).sum())
            g5 = int((ab > 5).sum())
            print(
                f"|间隔|≤2年     {le2} / {ntot} = {100*le2/ntot:.2f}%\n"
                f"2<|间隔|≤5年  {b25} / {ntot} = {100*b25/ntot:.2f}%\n"
                f"|间隔|>5年    {g5} / {ntot} = {100*g5/ntot:.2f}%  (以 eid+inst 去重后每对一行)"
            )
        strict = ex1
        if "pair_type" in ex1.columns:
            same_d = ex1["visit_date_fundus"].astype(str) == ex1["visit_date_cmr"].astype(str)
            strict = ex1[(ex1["pair_type"] == "same_inst") & same_d]
        n_s = strict["eid"].nunique()
        print(
            f"同 visit 严配对 (pair_type==same_inst 且 两日期字符串相同) 的 EID: {n_s}\n"
            f"全量可配对(任意跨 visit) 的 EID: {ex['eid'].nunique()}"
        )

    # ----- 表3 -----
    print("\n" + "=" * 80)
    print("表3：分类任务 — CMR=stage2_cmr inst2; 眼底=stage1_fundus(优先 inst2 一行/人); 配对= extended EID ∩ stage2_cmr inst2")
    print("=" * 80)
    fu_d = fundus_prefer_inst2(fu)
    for t in CLASS_TASKS:
        o = []
        for lab, d in [("CMR (stage2_cmr i2)", st2c_i2), ("眼底 (stage1_fundus i2)", fu_d), ("配对", pair)]:
            if t not in d.columns:
                o.append(f"{lab}: 无列")
                continue
            p, n, r = pos_rate(d[t])
            o.append(f"{lab}:   {fmt_pr(p, n, r)}")
        print(f"【{t}】\n  " + "\n  ".join(o))

    # ----- 表4 -----
    print("\n" + "=" * 80)
    print("表4：回归 (LV 来自 stage2_cmr / 子集 配对)")
    print("=" * 80)
    fu_eids = set(fu["eid"].unique())
    st2c_fundus = st2c_i2[st2c_i2["eid"].isin(fu_eids)]
    for col, name in [(LVEF, "LVEF"), (LVEDV, "LVEDV"), (LVESV, "LVESV")]:
        for lab, d in [
            ("CMR (stage2 i2)", st2c_i2),
            ("眼底 EID ∩ stage2_cmr i2 (用于 LV 标签)", st2c_fundus),
            ("配对 EID ∩ stage2_cmr i2", pair),
        ]:
            if col not in d.columns:
                print(f"  {name} / {lab}: 无列 {col}")
                continue
            v = pd.to_numeric(d[col], errors="coerce").dropna()
            if len(v) == 0:
                print(f"  {name} {lab}  非空 0")
                continue
            print(
                f"  {name} {lab}  非空 {len(v)}  均 {v.mean():.4f}  SD {v.std():.4f}  "
                f"P25 {v.quantile(0.25):.4f}  中 {v.median():.4f}  P75 {v.quantile(0.75):.4f}  "
                f"min {v.min():.4f}  max {v.max():.4f}"
            )

    # ----- 表5 -----
    print("\n" + "=" * 80)
    print("表5：人口学 (Age, Sex, BMI; 吸烟: stage1 表无此列则显示「表内无」)")
    print("=" * 80)
    age_col, sex_col, bmi_col = (
        "Age when attended assessment centre",
        "Sex",
        "Body mass index (BMI)",
    )
    smoke = [c for c in fu_u.columns if "smok" in c.lower() or "smoking" in c.lower()]
    for lab, d in [
        ("CMR (stage2_cmr i2)", st2c_i2),
        ("眼底 (stage1_fundus i2)", fu_d),
        ("配对 (extended∩stage2_cmr i2)", pair),
    ]:
        age = pd.to_numeric(d[age_col], errors="coerce") if age_col in d.columns else pd.Series(dtype=float)
        sex = d[sex_col] if sex_col in d.columns else pd.Series(dtype=object)
        bmi = pd.to_numeric(d[bmi_col], errors="coerce") if bmi_col in d.columns else pd.Series(dtype=float)
        a = age.dropna()
        b = bmi.dropna()
        print(f"\n{lab}  N={len(d)}")
        if len(a) > 0:
            print(
                f"  年龄:  均值±SD {a.mean():.2f}±{a.std():.2f}  中位 {a.median():.1f}  (P25–P75 {a.quantile(0.25):.1f}–{a.quantile(0.75):.1f})"
            )
        if len(sex) > 0 and sex.notna().any():
            m = (sex == "Male").sum()
            f = (sex == "Female").sum()
            t = m + f
            if t > 0:
                print(f"  男 {m} ({100*m/t:.1f}%)  女 {f} ({100*f/t:.1f}%)")
        if len(b) > 0:
            print(f"  BMI:   均值±SD {b.mean():.2f}±{b.std():.2f}")
    print(f"  吸烟:  自动检测列 {smoke if smoke else '无'}")

    # ----- 表6 -----
    print("\n" + "=" * 80)
    print("表6：眼底+master LV (Instance2, 两路 coalesce)")
    print("=" * 80)
    if not os.path.isfile(MASTER):
        print("  无 master_long.csv，跳过。")
    else:
        m = pd.read_csv(
            MASTER,
            usecols=["eid", "instance", M_LVEA, M_LVEB, M_LVD1, M_LVD2],
            low_memory=False,
        )
        m2 = m[m["instance"] == 2].drop_duplicates("eid")
        lv1 = pd.to_numeric(m2[M_LVEA], errors="coerce")
        lv2 = pd.to_numeric(m2[M_LVEB], errors="coerce")
        lvef = lv1.where(lv1.notna(), lv2)
        d1 = pd.to_numeric(m2[M_LVD1], errors="coerce")
        d2 = pd.to_numeric(m2[M_LVD2], errors="coerce")
        lvedv = d1.where(d1.notna(), d2)
        m2 = m2.assign(lvef=lvef, lvedv=lvedv)
        eid_fu = set(fu["eid"].unique())
        mm = m2[m2["eid"].isin(eid_fu)]
        print(
            f"  眼底 EID 在 master inst2: LVEF 非空 {int(mm['lvef'].notna().sum())}  |  "
            f"LVEDV 非空 {int(mm['lvedv'].notna().sum())}  |  双非空 {int((mm['lvef'].notna() & mm['lvedv'].notna()).sum())}"
        )
        inter = m2["eid"].isin(eid_fu) & m2["eid"].isin(eid_ext) & m2["lvef"].notna() & m2["lvedv"].notna()
        print(f"  与 extended 配对 EID 交集且双LV 非空: {int(inter.sum())}")

    # ----- 表7 -----
    print("\n" + "=" * 80)
    print("表7：CMR 序列 (磁盘) 与 npy")
    print("=" * 80)
    n8 = n9 = n13 = n89 = 0
    if os.path.isdir(CMR_ZIP):
        n8 = len(glob.glob(os.path.join(CMR_ZIP, "*_20208_*_0.zip")))
        n9 = len(glob.glob(os.path.join(CMR_ZIP, "*_20209_*_0.zip")))
        n13 = len(glob.glob(os.path.join(CMR_ZIP, "*_20213_*_0.zip")))
        for p in glob.glob(os.path.join(CMR_ZIP, "*_20208_2_0.zip")):
            b = os.path.basename(p)
            eid = b.split("_")[0]
            if os.path.isfile(os.path.join(CMR_ZIP, f"{eid}_20209_2_0.zip")):
                n89 += 1
    print(
        f"  20208 (LAX) zip 文件数(任意 instance):     {n8}\n"
        f"  20209 (SAX) zip 文件数:                  {n9}\n"
        f"  20213 (T1 map 等) zip 数:                 {n13}\n"
        f"  同时有 eid_20208_2_0 与 eid_20209_2_0: {n89}"
    )
    if os.path.isdir(NPY_DIR):
        fns = glob.glob(os.path.join(NPY_DIR, "*.npy"))[:20000]
        sh = []
        for p in fns:
            try:
                a = np.load(p, mmap_mode="r")
                sh.append(tuple(a.shape))
            except Exception as e:  # noqa
                sh.append((str(e)[:20],))
        c = Counter(sh)
        print(f"  npy 抽样 {len(fns)}: shape 频数 (前5) {c.most_common(5)}")


if __name__ == "__main__":
    if not os.path.isfile(ST1_FU):
        print("未找到", ST1_FU, "请在仓库根下运行", file=sys.stderr)
        sys.exit(1)
    if not os.path.isfile(ST2_CMR):
        print("未找到", ST2_CMR, file=sys.stderr)
        sys.exit(1)
    run_all()
