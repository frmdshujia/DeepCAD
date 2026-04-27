#!/usr/bin/env python3
"""
单张 Manhattan 风格图（非多子图）：

  横轴：按顺序排列「下游任务 | 指标」。每个任务可占 2 个或 3 个横坐标槽位（回归默认 2：
  Pearson r、R²；二分类默认 3：AUROC、AUPRC、balanced accuracy），可用参数改。

  纵轴：该指标在 **test** 上的数值（不同指标量纲不同，会在同一 y 轴上混排；读图以横轴标签为准）。

  每个「任务|指标」槽位上 **三个模型共用同一横坐标**（默认 `--dot_offset 0`），三点落在 **一条竖线** 上，仅纵坐标不同。

用法：
  python contrastive_pretrain/plot_manhattan_three_models.py \\
    --controlled_csv output_dir/downstream_probe_bundle_377_e2e_controlled/wide_metrics.csv \\
    --no_cov_csv output_dir/downstream_probe_bundle_377_no_cov_residual/wide_metrics.csv \\
    --out_png output_dir/manhattan_three_models_onepanel.png
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def _f(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    if isinstance(x, str) and (x.strip() == "" or x.strip().lower() == "nan"):
        return np.nan
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def _setup_chinese_font():
    candidates = [
        "Noto Sans CJK SC",
        "WenQuanYi Zen Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Arial Unicode MS",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            plt.rcParams["axes.unicode_minus"] = False
            return name
    plt.rcParams["axes.unicode_minus"] = False
    return None


def _status_ok(row: pd.Series, prefix: str) -> bool:
    c = f"{prefix}_status"
    return c in row.index and str(row[c]).strip() == "ok"


def _col(row: pd.Series, prefix: str, suffix: str) -> float:
    c = f"{prefix}_{suffix}"
    if c not in row.index:
        return np.nan
    return _f(row[c])


def _value_for_reg_metric(
    row: pd.Series, prefix: str, key: str
) -> float:
    """key: pearson | r2"""
    if not _status_ok(row, prefix):
        return np.nan
    r = _col(row, prefix, "test_pearson_r")
    if key == "pearson":
        return r
    if key == "r2":
        return r**2 if np.isfinite(r) else np.nan
    raise ValueError(key)


def _value_for_bin_metric(row: pd.Series, prefix: str, key: str) -> float:
    """key: auroc | auprc | balanced"""
    if not _status_ok(row, prefix):
        return np.nan
    m = {
        "auroc": "test_auroc",
        "auprc": "test_auprc",
        "balanced": "test_balanced_accuracy",
    }
    if key not in m:
        raise ValueError(key)
    return _col(row, prefix, m[key])


def _parse_reg_keys(s: str) -> list[str]:
    allowed = {"pearson", "r2"}
    out = [x.strip() for x in s.split(",") if x.strip()]
    for k in out:
        if k not in allowed:
            raise SystemExit(f"--reg_metrics 仅支持: {allowed}，收到: {k}")
    return out


def _parse_bin_keys(s: str) -> list[str]:
    allowed = {"auroc", "auprc", "balanced"}
    out = [x.strip() for x in s.split(",") if x.strip()]
    for k in out:
        if k not in allowed:
            raise SystemExit(f"--bin_metrics 仅支持: {allowed}，收到: {k}")
    return out


def _load_task_allowlist(path: str) -> set[str]:
    """每行一个 target_name；# 开头为注释。"""
    out = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            out.add(line)
    return out


def _metric_label(kind: str, key: str, locale: str) -> str:
    if locale == "zh":
        reg = {"pearson": "Pearson r", "r2": "R²"}
        bn = {"auroc": "AUROC", "auprc": "AUPRC", "balanced": "Balanced ACC"}
    else:
        reg = {"pearson": "Pearson r", "r2": "R²"}
        bn = {"auroc": "AUROC", "auprc": "AUPRC", "balanced": "Balanced ACC"}
    if kind == "reg":
        return reg[key]
    return bn[key]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--controlled_csv", type=str, default="output_dir/downstream_probe_bundle_377_e2e_controlled/wide_metrics.csv")
    p.add_argument("--no_cov_csv", type=str, default="output_dir/downstream_probe_bundle_377_no_cov_residual/wide_metrics.csv")
    p.add_argument("--baseline_prefix", type=str, default="A_cls")
    p.add_argument("--b_controlled_prefix", type=str, default="B_cls")
    p.add_argument("--b_no_cov_prefix", type=str, default="B_cls")
    p.add_argument(
        "--reg_metrics",
        type=str,
        default="pearson,r2",
        help="回归任务每个任务展开的指标，逗号分隔：pearson, r2",
    )
    p.add_argument(
        "--bin_metrics",
        type=str,
        default="auroc,auprc,balanced",
        help="二分类每个任务展开的指标：auroc, auprc, balanced",
    )
    p.add_argument("--out_png", type=str, default="output_dir/manhattan_three_models_onepanel.png")
    p.add_argument("--out_pdf", type=str, default="")
    p.add_argument("--dpi", type=int, default=160)
    p.add_argument("--fig_width", type=float, default=0.0, help="0=按槽位数自动变宽")
    p.add_argument("--fig_height", type=float, default=7.0)
    p.add_argument(
        "--dot_offset",
        type=float,
        default=0.0,
        help="默认 0：同一「任务|指标」槽位上三模型共用一个横坐标，三点落在一条竖线上；若需横向微错开可设 0.15 等",
    )
    p.add_argument("--locale", type=str, choices=("en", "zh"), default="en")
    p.add_argument(
        "--filter_tasks_file",
        type=str,
        default="",
        help="只保留表中列出的 target_name（每行一个，见 manhattan_favorable_allowlist.txt）",
    )
    p.add_argument("--title", type=str, default="", help="覆盖默认图标题")
    args = p.parse_args()

    reg_keys = _parse_reg_keys(args.reg_metrics)
    bin_keys = _parse_bin_keys(args.bin_metrics)

    base = args.controlled_csv if os.path.isabs(args.controlled_csv) else os.path.join(ROOT, args.controlled_csv)
    nov = args.no_cov_csv if os.path.isabs(args.no_cov_csv) else os.path.join(ROOT, args.no_cov_csv)
    df_c = pd.read_csv(base)
    df_n = pd.read_csv(nov)

    common = set(df_c["target_name"]) & set(df_n["target_name"])
    df_c = df_c[df_c["target_name"].isin(common)].sort_values("target_name").reset_index(drop=True)
    df_n = df_n[df_n["target_name"].isin(common)].sort_values("target_name").reset_index(drop=True)
    if not (df_c["target_name"].values == df_n["target_name"].values).all():
        df_n = df_n.set_index("target_name").reindex(df_c["target_name"]).reset_index()

    if args.filter_tasks_file:
        fpath = args.filter_tasks_file if os.path.isabs(args.filter_tasks_file) else os.path.join(ROOT, args.filter_tasks_file)
        allow = _load_task_allowlist(fpath)
        missing = allow - set(df_c["target_name"])
        if missing:
            print(f"[warn] allowlist 中有 {len(missing)} 个任务不在合并后的表中，已忽略: {list(missing)[:5]}...", file=sys.stderr)
        df_c = df_c[df_c["target_name"].isin(allow)].reset_index(drop=True)
        df_n = df_n[df_n["target_name"].isin(df_c["target_name"])].reset_index(drop=True)
        df_n = df_n.set_index("target_name").reindex(df_c["target_name"]).reset_index()
        print(f"[filter] 保留任务数={len(df_c)}（来自 {fpath}）")

    if args.locale == "zh":
        _setup_chinese_font()
        labels = {
            "baseline": f"基线 ({args.baseline_prefix})",
            "b_corr": f"矫正后 CL ({args.b_controlled_prefix})",
            "b_raw": f"非矫正 CL ({args.b_no_cov_prefix})",
        }
        ylab = "指标值（test）"
        title = args.title or "下游线性探针：三模型对照（横轴 = 任务 | 指标）"
    else:
        labels = {
            "baseline": f"Baseline ({args.baseline_prefix})",
            "b_corr": f"Contrastive corrected ({args.b_controlled_prefix})",
            "b_raw": f"Contrastive raw ({args.b_no_cov_prefix})",
        }
        ylab = "Metric value (test)"
        title = (
            args.title
            or "Linear probe: 3 models (x = task | metric)"
        )

    colors = {"baseline": "#7f7f7f", "b_corr": "#1f77b4", "b_raw": "#d62728"}
    markers = {"baseline": "o", "b_corr": "^", "b_raw": "s"}
    prefixes = {
        "baseline": args.baseline_prefix,
        "b_corr": args.b_controlled_prefix,
        "b_raw": args.b_no_cov_prefix,
    }

    # 构建槽位：(row_idx, kind, metric_key, xtick)
    slots: list[tuple[int, str, str, str]] = []
    for i, row in df_c.iterrows():
        tname = str(row["target_name"])
        kind = str(row["kind"])
        if kind == "reg":
            for mk in reg_keys:
                lab = _metric_label("reg", mk, args.locale)
                slots.append((i, "reg", mk, f"{tname}\n| {lab}"))
        elif kind == "binary":
            for mk in bin_keys:
                lab = _metric_label("binary", mk, args.locale)
                slots.append((i, "binary", mk, f"{tname}\n| {lab}"))
        else:
            continue

    n_slot = len(slots)
    if n_slot == 0:
        print("无可用槽位", file=sys.stderr)
        sys.exit(1)

    def pick_val(row_c: pd.Series, row_n: pd.Series, model_key: str, sk: str, kind: str) -> float:
        pref = prefixes[model_key]
        if model_key == "b_raw":
            row = row_n
        else:
            row = row_c
        if kind == "reg":
            return _value_for_reg_metric(row, pref, sk)
        return _value_for_bin_metric(row, pref, sk)

    xs = np.arange(n_slot, dtype=float)
    w = float(args.dot_offset)
    fig_w = args.fig_width if args.fig_width > 0 else min(48.0, max(14.0, n_slot * 0.38))
    fig, ax = plt.subplots(figsize=(fig_w, args.fig_height), constrained_layout=True)

    # 默认 w=0：同一槽位三模型共用横坐标，三点落在一条竖线上（仅 y 不同）
    for mi, model_key in enumerate(["baseline", "b_corr", "b_raw"]):
        ys = []
        for (row_i, kind, sk, _) in slots:
            row_c = df_c.iloc[row_i]
            row_n = df_n.iloc[row_i]
            ys.append(pick_val(row_c, row_n, model_key, sk, kind))
        off = (mi - 1) * w
        ax.scatter(
            xs + off,
            ys,
            c=colors[model_key],
            marker=markers[model_key],
            s=52,
            alpha=0.9,
            label=labels[model_key],
            edgecolors="white",
            linewidths=0.55,
            zorder=3 + mi,
        )

    ax.set_xticks(xs)
    ax.set_xticklabels([s[3] for s in slots], rotation=90, ha="center", fontsize=5)
    ax.set_ylabel(ylab)
    ax.set_xlabel("Downstream task | metric" if args.locale == "en" else "下游任务 | 指标")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.35)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.92)
    # y 范围：根据数据
    all_y = []
    for mi, model_key in enumerate(["baseline", "b_corr", "b_raw"]):
        for (row_i, kind, sk, _) in slots:
            row_c = df_c.iloc[row_i]
            row_n = df_n.iloc[row_i]
            all_y.append(pick_val(row_c, row_n, model_key, sk, kind))
    yy = np.array(all_y, dtype=float)
    yy = yy[np.isfinite(yy)]
    if len(yy):
        lo, hi = float(np.nanmin(yy)), float(np.nanmax(yy))
        pad = (hi - lo) * 0.06 + 0.02
        ax.set_ylim(lo - pad, hi + pad)

    out_png = args.out_png if os.path.isabs(args.out_png) else os.path.join(ROOT, args.out_png)
    os.makedirs(os.path.dirname(out_png) or ".", exist_ok=True)
    fig.savefig(out_png, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {out_png}  (n_slots={n_slot})")
    if args.out_pdf:
        out_pdf = args.out_pdf if os.path.isabs(args.out_pdf) else os.path.join(ROOT, args.out_pdf)
        os.makedirs(os.path.dirname(out_pdf) or ".", exist_ok=True)
        fig.savefig(out_pdf, bbox_inches="tight")
        print(f"Wrote {out_pdf}")

    plt.close()


if __name__ == "__main__":
    main()
