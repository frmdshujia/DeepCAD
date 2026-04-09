"""
基于 internal test set 的 ROC operating points 计算 Low / Moderate / High 三层分界阈值。

流程:
  1. 从 train/val/test dataset.pkl 构建 path -> y (0=阴性, 1=阳性)
  2. 从 infer_dist/results/gpu*.csv 读取 path -> p (模型概率)
  3. 匹配得到 (p_i, y_i) 列表（抽样估计，因仅部分样本有推理结果）
  4. 枚举阈值 t，计算 Sensitivity = TP/(TP+FN), Specificity = TN/(TN+FP)
  5. t_low: 满足 Sensitivity ≥ 0.90 的阈值中，选最接近 90% sens 或 spec 最高者
  6. t_high: 满足 Specificity ≥ 0.95 的阈值中，选最接近 95% spec 或 sens 最高者

输出:
  t_low: p < t_low → Low risk
  t_high: p ≥ t_high → High risk
  t_low ≤ p < t_high → Moderate risk

用法:
  python infer_dist/compute_cutoffs.py [--results-dir infer_dist/results] [--sens 0.90] [--spec 0.95]
"""

import csv
import pickle
import argparse
import os
from pathlib import Path
import numpy as np

# ─── 数据路径 ─────────────────────────────────────────────────────────────────
BASIC_CLASSIFY = "/data/home/shujia/dataset/CHD/basic_classify"
DATASETS = ["UKB", "SDPP", "SHCC", "WHTM", "PUDM"]
SPLITS = ["train", "val", "test"]


def load_path_to_label(splits=None):
    """从各数据集的 dataset.pkl 构建 path -> label (0/1)。
    splits: 使用的划分，默认 ['train','val','test']；若只用 test 可传 ['test'] 作为 internal test set
    """
    splits = splits or SPLITS
    path_to_label = {}
    for ds in DATASETS:
        for split in splits:
            pkl_path = os.path.join(BASIC_CLASSIFY, ds, split, "dataset.pkl")
            if not os.path.exists(pkl_path):
                continue
            with open(pkl_path, "rb") as f:
                d = pickle.load(f)
            # d = {'0': [paths...], '1': [paths...]}
            for label_str, paths in d.items():
                y = int(label_str)
                for p in paths:
                    path_to_label[p] = y
    return path_to_label


def load_path_to_prob_and_dataset(results_dir):
    """从 gpu*.csv 加载 path -> (prob, dataset)"""
    path_to_info = {}
    for f in sorted(Path(results_dir).glob("gpu*.csv")):
        with open(f) as fp:
            for row in csv.DictReader(fp):
                prob_str = row.get("prob", "").strip()
                if not prob_str:
                    continue
                try:
                    prob = float(prob_str)
                except ValueError:
                    continue
                path = row.get("path", "").strip()
                dataset = row.get("dataset", "").strip().upper()
                if path and dataset:
                    path_to_info[path] = (prob, dataset)
    return path_to_info


def compute_metrics_at_threshold(probs, labels, t):
    """阈值 t: p >= t 判为阳性"""
    pred_pos = probs >= t
    pred_neg = ~pred_pos
    actual_pos = labels == 1
    actual_neg = labels == 0

    TP = np.sum(pred_pos & actual_pos)
    FN = np.sum(pred_neg & actual_pos)
    TN = np.sum(pred_neg & actual_neg)
    FP = np.sum(pred_pos & actual_neg)

    sens = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    spec = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    return sens, spec, TP, FN, TN, FP


def find_cutoffs(probs, labels, target_sens=0.90, target_spec=0.95):
    """
    枚举所有概率值作为候选阈值（排序后逐档）。
    返回 t_low (sensitivity ≥ target_sens) 和 t_high (specificity ≥ target_spec)
    """
    probs = np.array(probs)
    labels = np.array(labels)

    # 候选阈值：所有出现过的概率 + 0, 1，去重排序
    candidates = np.unique(np.concatenate([[0, 1], probs]))
    candidates = np.sort(candidates)[::-1]  # 从高到低

    t_low_candidates = []  # (t, sens, spec) 满足 sens >= target_sens
    t_high_candidates = []  # (t, sens, spec) 满足 spec >= target_spec

    for t in candidates:
        sens, spec, _, _, _, _ = compute_metrics_at_threshold(probs, labels, t)
        if sens >= target_sens:
            t_low_candidates.append((t, sens, spec))
        if spec >= target_spec:
            t_high_candidates.append((t, sens, spec))

    # t_low: 满足 sensitivity ≥ target 的阈值中，选 最接近 target_sens 的
    # 若有多解，可选 spec 最高的（rule out 时兼顾一点特异性）
    if not t_low_candidates:
        t_low = None
        print(f"  [警告] 未找到满足 Sensitivity ≥ {target_sens} 的阈值")
    else:
        # 选 sens 最接近 target_sens 且 >= target_sens 的
        best = min(t_low_candidates, key=lambda x: (abs(x[1] - target_sens), -x[2]))
        t_low = best[0]
        print(f"  t_low = {t_low:.4f}  (Sensitivity={best[1]:.4f}, Specificity={best[2]:.4f})")

    # t_high: 满足 specificity ≥ target 的阈值中，选 最接近 target_spec 的
    if not t_high_candidates:
        t_high = None
        print(f"  [警告] 未找到满足 Specificity ≥ {target_spec} 的阈值")
    else:
        best = min(t_high_candidates, key=lambda x: (abs(x[2] - target_spec), -x[1]))
        t_high = best[0]
        print(f"  t_high = {t_high:.4f} (Sensitivity={best[1]:.4f}, Specificity={best[2]:.4f})")

    return t_low, t_high


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default=str(Path(__file__).parent / "results"),
        help="gpu*.csv 所在目录",
    )
    parser.add_argument("--sens", default=0.90, type=float, help="t_low 目标 sensitivity")
    parser.add_argument("--spec", default=0.95, type=float, help="t_high 目标 specificity")
    parser.add_argument("--test-only", action="store_true",
        help="仅用 test 划分（internal test set），避免 train/val 数据泄漏")
    parser.add_argument("--out", default=None, help="输出 JSON 路径，默认打印到终端")
    args = parser.parse_args()

    print("=" * 60)
    print("DeepCAD 阈值计算 (ROC operating points)")
    print("=" * 60)

    # 1. 加载标签
    use_splits = ["test"] if args.test_only else SPLITS
    print("\n[1] 加载 path -> label ...")
    if args.test_only:
        print("  [仅用 test 划分 = internal test set]")
    path_to_label = load_path_to_label(splits=use_splits)
    print(f"  有标签样本: {len(path_to_label):,}")

    # 2. 加载推理概率（含 dataset）
    print("\n[2] 加载 path -> (prob, dataset) ...")
    path_to_info = load_path_to_prob_and_dataset(args.results_dir)
    print(f"  已推理样本: {len(path_to_info):,}")

    # 3. 匹配 (p, y, dataset)，并按 UKB / 中国队列 分组
    ukb_data = []   # [(prob, label), ...]
    cn_data = []    # [(prob, label), ...]  SDPP+SHCC+WHTM+PUDM
    CN_DATASETS = {"SDPP", "SHCC", "WHTM", "PUDM"}

    for path, (prob, dataset) in path_to_info.items():
        if path not in path_to_label:
            continue
        label = path_to_label[path]
        if dataset == "UKB":
            ukb_data.append((prob, label))
        elif dataset in CN_DATASETS:
            cn_data.append((prob, label))

    n_ukb = len(ukb_data)
    n_cn = len(cn_data)
    n_matched = n_ukb + n_cn
    print(f"\n[3] 匹配到有标签的推理样本: {n_matched:,}")
    print(f"  UKB: {n_ukb:,}  (欧洲人群)")
    print(f"  中国队列 (SDPP+SHCC+WHTM+PUDM): {n_cn:,}")
    if n_cn == 0 and n_ukb > 0:
        print("  [注] 推理按 UKB→SDPP→SHCC→WHTM→PUDM 顺序，若中国队列为 0 表示推理尚未完成该部分")

    if n_matched < 100:
        print("  [警告] 匹配样本较少，结果为抽样估计，建议推理完成后再算")
        if n_matched < 10:
            print("  样本过少，无法可靠计算阈值，退出")
            return

    # 4. 分别计算 UKB 和 中国队列 的阈值
    import json
    out_data = {
        "target_sens": args.sens,
        "target_spec": args.spec,
        "ukb": {},
        "chinese": {},
    }

    for group_name, data, min_n in [
        ("UKB", ukb_data, 20),
        ("中国队列", cn_data, 20),
    ]:
        if len(data) < min_n:
            print(f"\n[4] {group_name}: 样本不足 {min_n}，跳过（推理完成后重新运行可获得）")
            continue
        probs = np.array([x[0] for x in data])
        labels = np.array([x[1] for x in data])
        n_pos = int(np.sum(labels == 1))
        n_neg = int(np.sum(labels == 0))
        print(f"\n[4] {group_name}: n={len(data):,} (阳性={n_pos}, 阴性={n_neg})")
        print(f"  计算阈值 (Sens≥{args.sens}, Spec≥{args.spec}) ...")
        t_low, t_high = find_cutoffs(probs, labels, args.sens, args.spec)

        grp_key = "chinese" if group_name == "中国队列" else "ukb"
        out_data[grp_key] = {
            "n_matched": len(data),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "t_low": t_low,
            "t_high": t_high,
        }
        print(f"  Low: p < {t_low:.4f}" if t_low else "  t_low: -")
        print(f"  Moderate: {t_low:.4f} ≤ p < {t_high:.4f}" if t_low and t_high else "")
        print(f"  High: p ≥ {t_high:.4f}" if t_high else "  t_high: -")

    # 5. 汇总输出
    print("\n" + "=" * 60)
    print("  分层定义汇总")
    print("=" * 60)
    for grp, name in [("ukb", "UKB"), ("chinese", "中国队列")]:
        d = out_data.get(grp, {})
        if d and d.get("t_low") is not None and d.get("t_high") is not None:
            print(f"  [{name}] Low < {d['t_low']:.4f} | Moderate {d['t_low']:.4f}-{d['t_high']:.4f} | High ≥ {d['t_high']:.4f}")
    print("=" * 60)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"\n结果已保存: {args.out}")


if __name__ == "__main__":
    main()
