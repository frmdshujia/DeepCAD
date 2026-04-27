#!/usr/bin/env python3
"""
在固定 train/val/test（如 377 人测试队列）上一次性跑下游对照：

  线1 — 眼底 encoder 线性探针：A=RETFound 初始化，B=对比学习 checkpoint（groups A,B × cls/proj）
  线3 — 仅用 fundus 表中的 14 维 PC（数值）做探针（PC→下游），作强基线
  线4 — 用 cmr_table 中对应 visit 的 14 维 PC 过训练好的 CMREncoder，再探针（检验 MLP 相对原始 PC 的信息）

线2（大规模人群探针/微调）由单独流程处理，不在此脚本运行。

输出：JSON、宽表 CSV、长表 CSV、ASSESSMENT.md；二分类含 test 与全队列分层 ACC（阈值 0.5）。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile

from contrastive_pretrain.linear_probe_stage2_sweep import (
    build_feature_tensors,
    discover_target_columns,
    extract_cache_key,
    load_group_a,
    load_group_b,
    load_merged_full,
    mask_split,
    run_one_target,
)
from contrastive_pretrain.models_contrast import CMREncoder

ImageFile.LOAD_TRUNCATED_IMAGES = True

DEFAULT_PC_COLS = [
    "M1_PC1", "M1_PC2", "M2_PC1", "M2_PC2", "M2_PC3", "M3_PC1", "M3_PC2",
    "M4_PC1", "M4_PC2", "M5_PC1", "M5_PC2", "M6_PC1", "M6_PC2", "M6_PC3",
]


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_merged_with_cmr_pc(
    fundus_csv: str, stage2_csv: str, cmr_csv: str, pc_cols: list[str]
) -> pd.DataFrame:
    m = load_merged_full(fundus_csv, stage2_csv)
    cm = pd.read_csv(_abs(cmr_csv), low_memory=False)
    key = ["eid", "instance"]
    miss = set(pc_cols + key) - set(cm.columns)
    if miss:
        raise ValueError(f"cmr_csv 缺少列: {miss}")
    sub = cm[key + pc_cols].drop_duplicates(subset=key)
    ren = {c: f"{c}_cmr" for c in pc_cols}
    sub = sub.rename(columns=ren)
    n_before = len(m)
    m = m.merge(sub, on=key, how="inner")
    print(f"[merge+cmr] N={len(m)}（merge 前 fundus+stage2 行数={n_before}；inner 会丢弃无 CMR PC 的行）")
    for c in pc_cols:
        assert f"{c}_cmr" in m.columns
    return m


def discover_targets_excluding_cmr_suffix(df: pd.DataFrame, max_missing_frac: float):
    raw = discover_target_columns(df, max_missing_frac)
    out = []
    for t in raw:
        if t["name"].endswith("_cmr"):
            continue
        if t["name"] in {f"{c}_cmr" for c in DEFAULT_PC_COLS}:
            continue
        out.append(t)
    return out


@torch.no_grad()
def embed_pc_cmr_mlp(
    df: pd.DataFrame,
    pc_cols: list[str],
    encoder: nn.Module,
    device: torch.device,
    batch_size: int = 4096,
) -> torch.Tensor:
    cols = [f"{c}_cmr" for c in pc_cols]
    x = torch.tensor(df[cols].values.astype(np.float32), dtype=torch.float32)
    outs = []
    encoder.eval()
    for s in range(0, len(x), batch_size):
        b = x[s : s + batch_size].to(device)
        outs.append(encoder(b).cpu())
    return torch.cat(outs, dim=0)


def build_pc14_tensors(df_tr, df_va, df_te, pc_cols: list[str]) -> tuple:
    def _t(df):
        return torch.tensor(df[pc_cols].values.astype(np.float32), dtype=torch.float32)

    return _t(df_tr).cpu(), _t(df_va).cpu(), _t(df_te).cpu()


def binary_test_pos_neg(df_te, name: str) -> tuple[int, int]:
    y = pd.to_numeric(df_te[name], errors="coerce")
    y = y.dropna()
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return pos, neg


def feature_key_to_line_id(key: str) -> str:
    if key in ("A_cls", "A_proj", "B_cls", "B_proj"):
        return "1_fundus_encoder"
    if key == "C_pc14":
        return "3_fundus_pc14"
    if key == "D_cmr_mlp":
        return "4_cmr_pc_mlp"
    return "other"


def _flatten_run_to_row(prefix: str, rr: dict, kind: str) -> dict:
    out: dict = {
        prefix + "status": "ok",
        prefix + "n_train": rr.get("n_train", ""),
        prefix + "n_val": rr.get("n_val", ""),
        prefix + "n_test": rr.get("n_test", ""),
    }
    if kind == "binary":
        for k in (
            "test_auroc",
            "test_auprc",
            "test_accuracy",
            "test_acc_on_true_positive",
            "test_acc_on_true_negative",
            "test_balanced_accuracy",
            "test_n_pos",
            "test_n_neg",
            "cohort_accuracy",
            "cohort_acc_on_true_positive",
            "cohort_acc_on_true_negative",
            "cohort_balanced_accuracy",
            "cohort_n_pos",
            "cohort_n_neg",
        ):
            out[prefix + k] = rr.get(k, "")
    else:
        out[prefix + "test_mae"] = rr.get("test_mae", "")
        out[prefix + "test_pearson_r"] = rr.get("test_pearson_r", "")
    return out


def build_long_metrics_table(results: dict, feature_keys: list[str]) -> pd.DataFrame:
    """长表：每行 = 一个 (目标, 特征/对照线)，便于筛选与作图。"""
    rows = []
    for name in sorted(results["by_column"].keys()):
        info = results["by_column"][name]
        kind = info["kind"]
        for fk in feature_keys:
            rr = info["runs"].get(fk, {})
            base = {
                "target_name": name,
                "kind": kind,
                "feature_key": fk,
                "line_id": feature_key_to_line_id(fk),
                "missing_frac_merged_cohort": info["missing_frac"],
                "table_test_pos": info.get("test_pos", ""),
                "table_test_neg": info.get("test_neg", ""),
            }
            if rr.get("skipped"):
                base["status"] = "skipped"
                base["skip_reason"] = rr.get("reason", "")
                rows.append(base)
                continue
            base["status"] = "ok"
            base["n_train"] = rr.get("n_train", "")
            base["n_val"] = rr.get("n_val", "")
            base["n_test"] = rr.get("n_test", "")
            if kind == "binary":
                for k in (
                    "test_auroc",
                    "test_auprc",
                    "test_accuracy",
                    "test_acc_on_true_positive",
                    "test_acc_on_true_negative",
                    "test_balanced_accuracy",
                    "test_n_pos",
                    "test_n_neg",
                    "cohort_accuracy",
                    "cohort_acc_on_true_positive",
                    "cohort_acc_on_true_negative",
                    "cohort_balanced_accuracy",
                    "cohort_n_pos",
                    "cohort_n_neg",
                ):
                    base[k] = rr.get(k, "")
            else:
                base["test_mae"] = rr.get("test_mae", "")
                base["test_pearson_r"] = rr.get("test_pearson_r", "")
            rows.append(base)
    return pd.DataFrame(rows)


def write_assessment_md(path: str, results: dict, feature_keys: list[str]) -> None:
    """简要解读：指标含义、稀有阳性、各线平均表现（仅作辅助，非因果结论）。"""
    lines = [
        "# 下游线性探针结果解读（377 人 test 队列）",
        "",
        "## 指标说明",
        "",
        "- **Test 集**：`test_auroc` / `test_auprc` / `test_accuracy` 及在真实阳性/阴性上的准确率（`test_acc_on_true_positive` ≈ 敏感度，`test_acc_on_true_negative` ≈ 特异度，阈值 0.5）为 **主报告** 的泛化表现。",
        "- **Cohort（train+val+test 有标签行）**：`cohort_*` 在同一探针下对 **全合并队列** 前向得到的 ACC；含训练集，**不**等同独立测试性能，仅用于查看在完整人群上的拟合与分层 ACC。",
        "- 线 1：`A_*` / `B_*`（眼底 encoder × cls/proj）；线 3：`C_pc14`；线 4：`D_cmr_mlp`。",
        "",
        "## 稀有阳性提醒（二分类）",
        "",
    ]
    risky = []
    for name, info in results["by_column"].items():
        if info.get("kind") != "binary":
            continue
        tp = info.get("test_pos", 0)
        if isinstance(tp, int) and tp < 10:
            risky.append(f"- `{name}`：test 阳性约 **{tp}** 例，AUROC/ACC 波动大，需谨慎解读。")
    lines.extend(risky if risky else ["- （无 test 阳性 <10 的列，或列未统计）", ""])
    lines.append("")
    lines.append("## 各特征键在二分类上的平均 test AUROC（仅 ok 行，nanmean）")
    lines.append("")
    bin_targets = [n for n, inf in results["by_column"].items() if inf.get("kind") == "binary"]
    for fk in feature_keys:
        vals = []
        for n in bin_targets:
            rr = results["by_column"][n]["runs"].get(fk, {})
            if rr.get("skipped"):
                continue
            v = rr.get("test_auroc")
            if v is not None and v == v:  # not nan
                vals.append(float(v))
        if vals:
            lines.append(f"- **{fk}**：mean AUROC = {float(np.nanmean(vals)):.4f}（n={len(vals)} 个目标）")
        else:
            lines.append(f"- **{fk}**：无可用二分类结果")
    lines.append("")
    lines.append("## 回归任务上的平均 |Pearson r|（test，仅 ok 行）")
    lines.append("")
    reg_targets = [n for n, inf in results["by_column"].items() if inf.get("kind") == "reg"]
    for fk in feature_keys:
        vals = []
        for n in reg_targets:
            rr = results["by_column"][n]["runs"].get(fk, {})
            if rr.get("skipped"):
                continue
            v = rr.get("test_pearson_r")
            if v is not None and v == v:
                vals.append(abs(float(v)))
        if vals:
            lines.append(f"- **{fk}**：mean |r| = {float(np.nanmean(vals)):.4f}（n={len(vals)}）")
        else:
            lines.append(f"- **{fk}**：无可用回归结果")
    lines.append("")
    path = _abs(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fundus_csv", type=str, default="contrastive_pretrain/preprocessed_data/fundus_table.csv")
    p.add_argument("--stage2_csv", type=str, default="contrastive_pretrain/preprocessed_data/stage2_cmr.csv")
    p.add_argument("--cmr_csv", type=str, default="contrastive_pretrain/preprocessed_data/cmr_table.csv")
    p.add_argument("--retfound_ckpt", type=str, default="RETFound_cfp_weights.pth")
    p.add_argument("--contrastive_ckpt", type=str, required=True, help="含 fundus_model + cmr_encoder")
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cache_dir", type=str, default="output_dir/downstream_probe_bundle_cache")
    p.add_argument("--skip_feature_cache", action="store_true")
    p.add_argument("--max_missing_frac", type=float, default=0.35)
    p.add_argument("--min_train", type=int, default=80)
    p.add_argument("--epochs_reg", type=int, default=100)
    p.add_argument("--epochs_bin", type=int, default=180)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--mlp_hidden_layers", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--output_json", type=str, default="output_dir/downstream_probe_bundle/results.json")
    p.add_argument("--output_csv", type=str, default="output_dir/downstream_probe_bundle/wide_metrics.csv")
    p.add_argument("--output_long_csv", type=str, default="output_dir/downstream_probe_bundle/metrics_long.csv")
    p.add_argument("--output_assessment_md", type=str, default="output_dir/downstream_probe_bundle/ASSESSMENT.md")
    p.add_argument("--skip_fundus_ab", action="store_true", help="跳过线1，仅跑线3/4（省时间）")
    p.add_argument("--skip_pc_baselines", action="store_true", help="跳过线3/4，仅跑线1")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    pc_cols = DEFAULT_PC_COLS

    merged = load_merged_with_cmr_pc(args.fundus_csv, args.stage2_csv, args.cmr_csv, pc_cols)
    targets = discover_targets_excluding_cmr_suffix(merged, args.max_missing_frac)
    print(f"[discover] 下游目标数={len(targets)}（已排除 cmr 后缀列）")

    df_tr = merged[merged["split"] == "train"].reset_index(drop=True)
    df_va = merged[merged["split"] == "val"].reset_index(drop=True)
    df_te = merged[merged["split"] == "test"].reset_index(drop=True)

    cache_root = _abs(args.cache_dir)
    os.makedirs(cache_root, exist_ok=True)

    feature_banks: dict[str, tuple] = {}

    groups_ab = ["A", "B"]
    reps = ["cls", "proj"]

    if not args.skip_fundus_ab:
        for g in groups_ab:
            if g == "A":
                model = load_group_a(args.proj_dim, args.drop_path, args.retfound_ckpt, device)
            else:
                model = load_group_b(args.proj_dim, args.drop_path, args.contrastive_ckpt, device)
            for rep in reps:
                key = extract_cache_key(g, rep)
                path_pt = os.path.join(cache_root, f"{key}.pt")
                if not args.skip_feature_cache and os.path.isfile(path_pt):
                    print(f"[cache] 加载 {key} ← {path_pt}")
                    blob = torch.load(path_pt, map_location="cpu")
                    feature_banks[key] = (blob["Xt"], blob["Xv"], blob["Xe"])
                else:
                    print(f"[cache] 计算 {key} …")
                    Xt, Xv, Xe = build_feature_tensors(
                        df_tr, df_va, df_te, model, rep, device, args.batch_size, args.num_workers
                    )
                    feature_banks[key] = (Xt, Xv, Xe)
                    if not args.skip_feature_cache:
                        torch.save({"Xt": Xt, "Xv": Xv, "Xe": Xe}, path_pt)
                torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()

    if not args.skip_pc_baselines:
        Xt, Xv, Xe = build_pc14_tensors(df_tr, df_va, df_te, pc_cols)
        feature_banks["C_pc14"] = (Xt, Xv, Xe)
        print("[features] C_pc14 fundus 表 14 维 PC（数值）")

        ckpt = torch.load(_abs(args.contrastive_ckpt), map_location="cpu")
        ca = ckpt.get("args") or {}
        pdim = int(ca.get("proj_dim", args.proj_dim))
        cmr_enc = CMREncoder(in_dim=14, hidden_dim=128, out_dim=pdim)
        cmr_enc.load_state_dict(ckpt["cmr_encoder"], strict=True)
        cmr_enc.to(device)
        cmr_enc.eval()
        for p in cmr_enc.parameters():
            p.requires_grad_(False)

        Zt = embed_pc_cmr_mlp(df_tr, pc_cols, cmr_enc, device)
        Zv = embed_pc_cmr_mlp(df_va, pc_cols, cmr_enc, device)
        Ze = embed_pc_cmr_mlp(df_te, pc_cols, cmr_enc, device)
        feature_banks["D_cmr_mlp"] = (Zt.cpu(), Zv.cpu(), Ze.cpu())
        print(f"[features] D_cmr_mlp CMREncoder 嵌入 dim={pdim}（输入为 cmr_table 的 14 维 PC）")
        del cmr_enc
        torch.cuda.empty_cache()

    results = {
        "note": "线1=A/B×cls/proj 眼底；线3=C 14PC；线4=D CMREncoder(cmr PC)；线2 大规模未在此运行",
        "fundus_csv": args.fundus_csv,
        "stage2_csv": args.stage2_csv,
        "cmr_csv": args.cmr_csv,
        "contrastive_ckpt": args.contrastive_ckpt,
        "n_merged": len(merged),
        "targets_meta": targets,
        "feature_keys": list(feature_banks.keys()),
        "by_column": {},
    }

    for tinfo in targets:
        cname = tinfo["name"]
        results["by_column"][cname] = {
            "kind": tinfo["kind"],
            "missing_frac": tinfo["missing_frac"],
            "runs": {},
        }
        if tinfo["kind"] == "binary":
            tp, tn = binary_test_pos_neg(df_te, cname)
            results["by_column"][cname]["test_pos"] = tp
            results["by_column"][cname]["test_neg"] = tn

        for key in feature_banks:
            Xt, Xv, Xe = feature_banks[key]
            r = run_one_target(
                cname,
                tinfo["kind"],
                df_tr,
                df_va,
                df_te,
                Xt,
                Xv,
                Xe,
                device,
                args,
            )
            r["feature_key"] = key
            results["by_column"][cname]["runs"][key] = r
            if not r.get("skipped"):
                if tinfo["kind"] == "binary":
                    print(
                        f"  [{key}] {cname[:42]:<42}  AUROC={r['test_auroc']:.4f}  "
                        f"AUPRC={r['test_auprc']:.4f}  ACC={r['test_accuracy']:.4f}  "
                        f"sens@0.5={r['test_acc_on_true_positive']:.4f}  spec@0.5={r['test_acc_on_true_negative']:.4f}"
                    )
                else:
                    print(
                        f"  [{key}] {cname[:42]:<42}  MAE={r['test_mae']:.4f}  r={r['test_pearson_r']:.4f}"
                    )

    out_json = _abs(args.output_json)
    os.makedirs(os.path.dirname(out_json) or ".", exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_json}")

    # 宽表 + 长表 + 解读
    rows = []
    all_keys = list(feature_banks.keys())
    for name in sorted(results["by_column"].keys()):
        info = results["by_column"][name]
        row = {
            "target_name": name,
            "kind": info["kind"],
            "missing_frac_merged_cohort": info["missing_frac"],
            "table_test_pos": info.get("test_pos", ""),
            "table_test_neg": info.get("test_neg", ""),
        }
        for key in all_keys:
            prefix = key + "_"
            rr = info["runs"].get(key, {})
            if rr.get("skipped"):
                row[prefix + "status"] = "skipped"
                row[prefix + "skip_reason"] = rr.get("reason", "")
                continue
            row.update(_flatten_run_to_row(prefix, rr, info["kind"]))
        rows.append(row)
    out_csv = _abs(args.output_csv)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")

    long_df = build_long_metrics_table(results, all_keys)
    out_long = _abs(args.output_long_csv)
    os.makedirs(os.path.dirname(out_long) or ".", exist_ok=True)
    long_df.to_csv(out_long, index=False)
    print(f"Wrote {out_long}")

    write_assessment_md(args.output_assessment_md, results, all_keys)


if __name__ == "__main__":
    main()
