"""
linear_probe_stage2_sweep.py
在 fundus 与 stage2_cmr 对齐的队列上，对 stage2 中「可数值化」的衍生量/病史等列
批量做冻结 encoder + 线性探针（默认单层）。

流程：
  1) 合并 fundus_table + stage2_cmr（全列）
  2) 可选：缓存各组(A/B)×表示(cls/proj) 的 train/val/test 特征矩阵（避免每个目标重复前向）
  3) 自动跳过非数值列、路径/日期、以及缺失过多的列；二值列走 BCE，其余走回归

用法示例：
  python contrastive_pretrain/linear_probe_stage2_sweep.py \\
    --contrastive_ckpt output_dir/contrast_finetune_e2e_20260412_174935/checkpoint_best.pth \\
    --cache_dir output_dir/stage2_probe_feature_cache \\
    --output_json output_dir/stage2_linear_probe_sweep.json
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
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from contrastive_pretrain.linear_probe_clinical import (
    make_probe_head,
    train_chd_probe,
    train_lvef_probe,
)
from contrastive_pretrain.linear_probe_clinical import extract_features_cls_dual, extract_features_proj_dual
from contrastive_pretrain.models_contrast import FundusContrastModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 不参与建模的列（键、路径、明显类别文本）
EXCLUDE_PREFIX = tuple()
EXCLUDE_NAMES = {
    "eid",
    "instance",
    "split",
    "fundus_image_path",
    "eye",
    "Sex",
    "Date of attending assessment centre",
    "Short axis heart images - DICOM",
    "Long axis heart images - DICOM",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--fundus_csv", type=str, default="contrastive_pretrain/preprocessed_data/fundus_table.csv")
    p.add_argument("--stage2_csv", type=str, default="contrastive_pretrain/preprocessed_data/stage2_cmr.csv")
    p.add_argument("--retfound_ckpt", type=str, default="RETFound_cfp_weights.pth")
    p.add_argument("--contrastive_ckpt", type=str, required=True)
    p.add_argument("--proj_dim", type=int, default=256)
    p.add_argument("--drop_path", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--cache_dir", type=str, default="output_dir/stage2_probe_feature_cache")
    p.add_argument("--skip_feature_cache", action="store_true", help="不读写缓存（每次重算特征）")
    p.add_argument(
        "--groups",
        type=str,
        default="A,B",
        help="逗号分隔：A=RETFound 初始化，B=对比学习 fundus_model",
    )
    p.add_argument("--representations", type=str, default="cls,proj", help="cls 与/或 proj")
    p.add_argument("--mlp_hidden_layers", type=int, default=0, choices=[0, 1, 2])
    p.add_argument("--mlp_hidden_dim", type=int, default=512)
    p.add_argument("--epochs_reg", type=int, default=100)
    p.add_argument("--epochs_bin", type=int, default=180)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--min_train", type=int, default=80, help="训练样本过少则跳过该列")
    p.add_argument("--max_missing_frac", type=float, default=0.35, help="列整体缺失比例超过此值则跳过")
    p.add_argument("--output_json", type=str, default="output_dir/stage2_linear_probe_sweep.json")
    p.add_argument(
        "--export_csv",
        type=str,
        default="",
        help="宽表 CSV：每行一个目标，列含 A_cls/A_proj/B_cls/B_proj 的指标与跳过原因",
    )
    p.add_argument("--lvef_outlier_report", action="store_true", help="额外打印 LV ejection fraction 异常值统计")
    return p.parse_args()


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(ROOT, p)


def load_merged_full(fundus_csv: str, stage2_csv: str) -> pd.DataFrame:
    fu = pd.read_csv(_abs(fundus_csv))
    st = pd.read_csv(_abs(stage2_csv), low_memory=False)
    if "eid" not in st.columns or "instance" not in st.columns:
        raise ValueError("stage2 需含 eid, instance")
    st = st.drop_duplicates(subset=["eid", "instance"])
    m = fu.merge(st, on=["eid", "instance"], how="inner", suffixes=("", "_st2"))
    if "split_st2" in m.columns:
        if (m["split"] != m["split_st2"]).any():
            raise ValueError("fundus 与 stage2 split 不一致")
        m = m.drop(columns=["split_st2"])
    exist_mask = m["fundus_image_path"].apply(os.path.exists)
    if (~exist_mask).any():
        m = m[exist_mask].reset_index(drop=True)
    print(f"[merge] N={len(m)}  train/val/test={len(m[m.split=='train'])}/{len(m[m.split=='val'])}/{len(m[m.split=='test'])}")
    return m


class FundusPathsDataset(Dataset):
    def __init__(self, paths: list, train_aug: bool):
        self.paths = paths
        self.train_aug = train_aug
        mean, std = IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
        if train_aug:
            self.t = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.64, 1.0), ratio=(3 / 4, 4 / 3)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )
        else:
            self.t = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            )

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        img = self.t(img)
        return img, torch.tensor(0.0), torch.tensor(0.0)


def extract_cache_key(group: str, rep: str) -> str:
    return f"{group}_{rep}"


@torch.no_grad()
def build_feature_tensors(
    df_tr, df_va, df_te, model: FundusContrastModel, representation: str, device, batch_size: int, num_workers: int
):
    def _loader(paths, aug):
        ds = FundusPathsDataset(paths, aug)
        return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    lt = _loader(df_tr["fundus_image_path"].tolist(), True)
    lv = _loader(df_va["fundus_image_path"].tolist(), False)
    lte = _loader(df_te["fundus_image_path"].tolist(), False)
    if representation == "cls":
        Xt, _, _ = extract_features_cls_dual(lt, model.backbone, device)
        Xv, _, _ = extract_features_cls_dual(lv, model.backbone, device)
        Xe, _, _ = extract_features_cls_dual(lte, model.backbone, device)
    else:
        Xt, _, _ = extract_features_proj_dual(lt, model, device)
        Xv, _, _ = extract_features_proj_dual(lv, model, device)
        Xe, _, _ = extract_features_proj_dual(lte, model, device)
    return Xt.cpu(), Xv.cpu(), Xe.cpu()


def load_group_a(proj_dim, drop_path, retfound_path, device):
    m = FundusContrastModel(proj_dim=proj_dim, drop_path_rate=drop_path)
    m.load_pretrained(_abs(retfound_path))
    m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def load_group_b(proj_dim, drop_path, contrast_ckpt, device):
    ckpt = torch.load(_abs(contrast_ckpt), map_location="cpu")
    ca = ckpt.get("args") or {}
    pdim = int(ca.get("proj_dim", proj_dim))
    dp = float(ca.get("drop_path", drop_path))
    m = FundusContrastModel(proj_dim=pdim, drop_path_rate=dp)
    m.load_state_dict(ckpt["fundus_model"], strict=True)
    m.to(device)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def discover_target_columns(df: pd.DataFrame, max_missing_frac: float) -> list[dict]:
    """返回 {name, kind: 'binary'|'reg', missing_frac}"""
    out = []
    n = len(df)
    for c in df.columns:
        if c in EXCLUDE_NAMES:
            continue
        if c in ("split", "fundus_image_path"):
            continue
        s = pd.to_numeric(df[c], errors="coerce")
        miss = float(s.isna().mean())
        if miss > max_missing_frac:
            continue
        v = s.dropna()
        if len(v) < 10:
            continue
        u = np.unique(v.values)
        if len(u) <= 2:
            # 二值 / 极少水平：若落在 {0,1} 则二分类
            uu = set(float(x) for x in u)
            if uu <= {0.0, 1.0}:
                out.append({"name": c, "kind": "binary", "missing_frac": miss})
                continue
        # 回归：需连续变化
        if len(u) < 5:
            continue
        out.append({"name": c, "kind": "reg", "missing_frac": miss})
    return out


def mask_split(X: torch.Tensor, y: np.ndarray):
    """y 与 X 行对齐；去掉 NaN。"""
    m = ~np.isnan(y.astype(float))
    return X[m], torch.tensor(y[m], dtype=torch.float32)


def run_one_target(
    name: str,
    kind: str,
    df_tr,
    df_va,
    df_te,
    Xt,
    Xv,
    Xe,
    device,
    args,
):
    y_tr = pd.to_numeric(df_tr[name], errors="coerce").values.astype(np.float64)
    y_va = pd.to_numeric(df_va[name], errors="coerce").values.astype(np.float64)
    y_te = pd.to_numeric(df_te[name], errors="coerce").values.astype(np.float64)
    X_tr, y_tr_t = mask_split(Xt, y_tr)
    X_val, y_val_t = mask_split(Xv, y_va)
    X_test, y_te_t = mask_split(Xe, y_te)
    if len(X_tr) < args.min_train or len(X_val) < 8 or len(X_test) < 8:
        return {"skipped": True, "reason": "too_few_after_dropna"}

    if kind == "binary":
        # 训练/验证/测试均需两类（否则 AUROC 无定义）
        def _both_classes(y):
            u = set(float(t.item()) for t in y)
            return u >= {0.0, 1.0}

        if not _both_classes(y_tr_t) or not _both_classes(y_val_t) or not _both_classes(y_te_t):
            return {"skipped": True, "reason": "not_both_classes_in_split"}
        stats = train_chd_probe(
            X_tr,
            y_tr_t,
            X_val,
            y_val_t,
            X_test,
            y_te_t,
            device,
            args.epochs_bin,
            args.lr,
            args.weight_decay,
            args.seed,
            args.patience,
            args.mlp_hidden_layers,
            args.mlp_hidden_dim,
        )
        return {
            "skipped": False,
            "task": "binary",
            "n_train": int(len(X_tr)),
            "n_val": int(len(X_val)),
            "n_test": int(len(X_test)),
            "test_auroc": stats["test_auroc"],
            "test_auprc": stats["test_auprc"],
            "test_accuracy": stats["test_accuracy"],
            "test_acc_on_true_positive": stats["test_acc_on_true_positive"],
            "test_acc_on_true_negative": stats["test_acc_on_true_negative"],
            "test_balanced_accuracy": stats["test_balanced_accuracy"],
            "test_n_pos": stats["test_n_pos"],
            "test_n_neg": stats["test_n_neg"],
            "cohort_accuracy": stats["cohort_accuracy"],
            "cohort_acc_on_true_positive": stats["cohort_acc_on_true_positive"],
            "cohort_acc_on_true_negative": stats["cohort_acc_on_true_negative"],
            "cohort_balanced_accuracy": stats["cohort_balanced_accuracy"],
            "cohort_n_pos": stats["cohort_n_pos"],
            "cohort_n_neg": stats["cohort_n_neg"],
            "epochs_ran": stats["epochs_ran"],
        }
    stats = train_lvef_probe(
        X_tr,
        y_tr_t,
        X_val,
        y_val_t,
        X_test,
        y_te_t,
        device,
        args.epochs_reg,
        args.lr,
        args.weight_decay,
        args.seed,
        args.patience,
        args.mlp_hidden_layers,
        args.mlp_hidden_dim,
    )
    return {
        "skipped": False,
        "task": "regression",
        "n_train": int(len(X_tr)),
        "n_val": int(len(X_val)),
        "n_test": int(len(X_test)),
        "test_mae": stats["test_mae"],
        "test_pearson_r": stats["test_pearson_r"],
        "epochs_ran": stats["epochs_ran"],
    }


def export_wide_csv(results: dict, csv_path: str, groups: list, reps: list) -> None:
    """将 by_column 展平为一行一目标、四路探针分列。"""
    keys_order = [extract_cache_key(g, r) for g in groups for r in reps]
    rows = []
    for name in sorted(results["by_column"].keys()):
        info = results["by_column"][name]
        row = {
            "target_name": name,
            "kind": info["kind"],
            "missing_frac_merged_cohort": info["missing_frac"],
        }
        runs = info.get("runs", {})
        for key in keys_order:
            prefix = key + "_"
            r = runs.get(key, {})
            if not r:
                row[prefix + "status"] = "missing"
                continue
            if r.get("skipped"):
                row[prefix + "status"] = "skipped"
                row[prefix + "skip_reason"] = r.get("reason", "")
                for k in ("n_train", "n_val", "n_test", "test_mae", "test_pearson_r", "test_auroc", "test_auprc"):
                    row[prefix + k] = ""
            else:
                row[prefix + "status"] = "ok"
                row[prefix + "n_train"] = r.get("n_train", "")
                row[prefix + "n_val"] = r.get("n_val", "")
                row[prefix + "n_test"] = r.get("n_test", "")
                if r.get("task") == "binary":
                    row[prefix + "test_mae"] = ""
                    row[prefix + "test_pearson_r"] = ""
                    row[prefix + "test_auroc"] = r.get("test_auroc", "")
                    row[prefix + "test_auprc"] = r.get("test_auprc", "")
                else:
                    row[prefix + "test_mae"] = r.get("test_mae", "")
                    row[prefix + "test_pearson_r"] = r.get("test_pearson_r", "")
                    row[prefix + "test_auroc"] = ""
                    row[prefix + "test_auprc"] = ""
        rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = _abs(csv_path)
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[export] 宽表 CSV → {csv_path}  (rows={len(df)})")


def lvef_outlier_report(df: pd.DataFrame):
    col = "LV ejection fraction"
    if col not in df.columns:
        return
    v = pd.to_numeric(df[col], errors="coerce").dropna()
    q1, q3 = v.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    z = (v - v.mean()) / v.std()
    print(
        f"\n[LVEF 粗查] n={len(v)}  mean={v.mean():.2f}  std={v.std():.2f}  "
        f"min={v.min():.2f}  max={v.max():.2f}"
    )
    print(f"  IQR 异常 (outside [{lo:.1f},{hi:.1f}]): {int(((v<lo)|(v>hi)).sum())} ({100*((v<lo)|(v>hi)).mean():.2f}%)")
    print(f"  |z|>3: {int((z.abs()>3).sum())}  |z|>4: {int((z.abs()>4).sum())}")


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    merged = load_merged_full(args.fundus_csv, args.stage2_csv)
    if args.lvef_outlier_report:
        lvef_outlier_report(merged)

    targets = discover_target_columns(merged, args.max_missing_frac)
    print(f"[discover] 可测列数={len(targets)}（已排除键/路径/高缺失列）")

    df_tr = merged[merged["split"] == "train"].reset_index(drop=True)
    df_va = merged[merged["split"] == "val"].reset_index(drop=True)
    df_te = merged[merged["split"] == "test"].reset_index(drop=True)

    groups = [g.strip() for g in args.groups.split(",") if g.strip()]
    reps = [r.strip() for r in args.representations.split(",") if r.strip()]
    cache_root = _abs(args.cache_dir)
    os.makedirs(cache_root, exist_ok=True)

    feature_banks: dict[str, tuple] = {}

    for g in groups:
        if g == "A":
            model = load_group_a(args.proj_dim, args.drop_path, args.retfound_ckpt, device)
        elif g == "B":
            model = load_group_b(args.proj_dim, args.drop_path, args.contrastive_ckpt, device)
        else:
            raise ValueError(f"未知 group={g}")
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
                    print(f"  已保存 {path_pt}")
            torch.cuda.empty_cache()
        del model
        torch.cuda.empty_cache()

    results: dict = {
        "n_merged": len(merged),
        "targets_meta": targets,
        "groups": groups,
        "representations": reps,
        "mlp_hidden_layers": args.mlp_hidden_layers,
        "by_column": {},
    }

    for tinfo in targets:
        cname = tinfo["name"]
        results["by_column"][cname] = {"kind": tinfo["kind"], "missing_frac": tinfo["missing_frac"], "runs": {}}
        for g in groups:
            for rep in reps:
                key = extract_cache_key(g, rep)
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
                results["by_column"][cname]["runs"][key] = r
                if not r.get("skipped"):
                    if tinfo["kind"] == "binary":
                        print(
                            f"  [{key}] {cname[:50]:<50}  AUROC={r['test_auroc']:.4f}  AUPRC={r['test_auprc']:.4f}"
                        )
                    else:
                        print(
                            f"  [{key}] {cname[:50]:<50}  MAE={r['test_mae']:.4f}  r={r['test_pearson_r']:.4f}"
                        )

    out_path = _abs(args.output_json)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {out_path}")

    if args.export_csv:
        export_wide_csv(results, args.export_csv, groups, reps)


if __name__ == "__main__":
    main()
