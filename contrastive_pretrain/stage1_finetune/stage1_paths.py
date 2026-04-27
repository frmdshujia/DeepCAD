"""
从 stage1 表展开为「每张眼底图一行」，含 fundus_image_path。

支持两种 CSV：
1) 新表：含 fundus_image_path_left / fundus_image_path_right（与左右眼可用性列一致时写入路径）
2) 旧表：仅用 UKB 规则 glob（21015 左 / 21016 右）
"""
from __future__ import annotations

import glob
import os
from typing import List

import numpy as np
import pandas as pd

# 默认任务顺序：先复合标签，再 ICD prevalent（与实验优先级一致）
COMPOSITE_TASKS = [
    "composite_diabetes",
    "composite_hypertension",
    "composite_ischemic_hd",
    "composite_cardiomyopathy_hf",
    "composite_af_arrhythmia",
    "composite_hypertensive_hd",
    "composite_valvular_hd",
]
ICD_PREVALENT_TASKS = [
    "prevalent_I20",
    "prevalent_I21",
    "prevalent_I22",
    "prevalent_I24",
    "prevalent_I25",
    "prevalent_I42",
    "prevalent_I43",
    "prevalent_I50",
]

COL_LEFT = "Fundus retinal eye image (left)"
COL_RIGHT = "Fundus retinal eye image (right)"
COL_PATH_LEFT = "fundus_image_path_left"
COL_PATH_RIGHT = "fundus_image_path_right"


def _glob_field(fundus_root: str, eid: int, field: int, instance) -> List[str]:
    pat = os.path.join(fundus_root, f"{eid}_{field}_{instance}_*.png")
    return sorted(glob.glob(pat))


def expand_rows_to_image_paths(df: pd.DataFrame, fundus_root: str) -> pd.DataFrame:
    """
    旧表：无显式路径列时，按 eid/instance + 左右眼可用性 glob 21015/21016。
    输出每行一张图，含 fundus_image_path, eye。
    """
    if COL_LEFT not in df.columns or COL_RIGHT not in df.columns:
        raise ValueError(f"CSV 需含列 {COL_LEFT}, {COL_RIGHT}")

    rows_out = []
    for _, r in df.iterrows():
        eid = int(r["eid"])
        inst = r["instance"]
        try:
            inst = int(inst)
        except (TypeError, ValueError):
            inst = int(float(inst))

        lf = int(r[COL_LEFT]) if pd.notna(r[COL_LEFT]) else 0
        rf = int(r[COL_RIGHT]) if pd.notna(r[COL_RIGHT]) else 0

        if lf == 1:
            for p in _glob_field(fundus_root, eid, 21015, inst):
                rows_out.append({**r.to_dict(), "fundus_image_path": p, "eye": "left"})
        if rf == 1:
            for p in _glob_field(fundus_root, eid, 21016, inst):
                rows_out.append({**r.to_dict(), "fundus_image_path": p, "eye": "right"})

    out = pd.DataFrame(rows_out)
    if len(out) == 0:
        return out
    return out.reset_index(drop=True)


def expand_rows_from_explicit_path_columns(df: pd.DataFrame) -> pd.DataFrame:
    """新表：使用 fundus_image_path_left / fundus_image_path_right。"""
    if COL_LEFT not in df.columns or COL_RIGHT not in df.columns:
        raise ValueError(f"CSV 需含列 {COL_LEFT}, {COL_RIGHT}")
    if COL_PATH_LEFT not in df.columns or COL_PATH_RIGHT not in df.columns:
        raise ValueError(f"CSV 需含列 {COL_PATH_LEFT}, {COL_PATH_RIGHT}")

    rows_out = []
    for _, r in df.iterrows():
        lf = int(r[COL_LEFT]) if pd.notna(r[COL_LEFT]) else 0
        rf = int(r[COL_RIGHT]) if pd.notna(r[COL_RIGHT]) else 0
        if lf == 1:
            p = r.get(COL_PATH_LEFT)
            if pd.notna(p) and str(p).strip():
                rows_out.append(
                    {**r.to_dict(), "fundus_image_path": str(p).strip(), "eye": "left"}
                )
        if rf == 1:
            p = r.get(COL_PATH_RIGHT)
            if pd.notna(p) and str(p).strip():
                rows_out.append(
                    {**r.to_dict(), "fundus_image_path": str(p).strip(), "eye": "right"}
                )
    out = pd.DataFrame(rows_out)
    if len(out) == 0:
        return out
    return out.reset_index(drop=True)


def prepare_image_frame(df: pd.DataFrame, fundus_root: str) -> pd.DataFrame:
    """
    若存在显式左右路径列则用之；否则回退到 UKB glob。
    """
    if COL_PATH_LEFT in df.columns and COL_PATH_RIGHT in df.columns:
        return expand_rows_from_explicit_path_columns(df)
    return expand_rows_to_image_paths(df, fundus_root)


def filter_target_valid(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    if target_col not in df.columns:
        raise ValueError(f"缺少目标列: {target_col}")
    s = df[target_col]
    m = s.notna() & (s.astype(str) != "nan")
    df = df.loc[m].copy()
    y = pd.to_numeric(df[target_col], errors="coerce")
    df[target_col] = y
    df = df.loc[y.notna()].copy()
    df = df.loc[np.isin(df[target_col].values, [0, 1, 0.0, 1.0])].copy()
    df[target_col] = df[target_col].astype(np.int64)
    return df.reset_index(drop=True)


def split_subset(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    return df.loc[df["split"].astype(str) == split_name].reset_index(drop=True)
