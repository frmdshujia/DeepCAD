#!/usr/bin/env python3
"""统计眼底门控模型在正/负样本上的概率分布"""
import os, sys, csv
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fundus_gate.inference import FundusGateChecker

OUT_FILE = "/data/home/shujia/CHD/model_train/RETFound_MAE-main/fundus_gate/prob_stats_result.txt"

def main():
    ckpt = "/data/home/shujia/CHD/model_train/RETFound_MAE-main/fundus_gate/checkpoints/best_fundus_gate.pth"
    checker = FundusGateChecker(ckpt_path=ckpt, threshold=0.5, device='cpu')

    lines = []

    # 1. 眼底正样本（val集前80张）
    val_csv = "/data/home/shujia/CHD/model_train/RETFound_MAE-main/fundus_gate/val.csv"
    fundus = [r['path'] for r in csv.DictReader(open(val_csv)) if int(r['label'])==1 and os.path.exists(r['path'])]
    sample = fundus[:80]

    probs = []
    for i, p in enumerate(sample):
        _, pr = checker.check(p)
        probs.append(pr)
        if (i+1) % 20 == 0:
            print(f"  眼底 {i+1}/80...", flush=True)

    probs = np.array(probs)
    lines.append("=" * 60)
    lines.append("眼底图像（正样本，val集前80张）概率分布")
    lines.append("=" * 60)
    lines.append(f"样本数: {len(probs)}")
    lines.append(f"均值:   {probs.mean():.4f}")
    lines.append(f"标准差: {probs.std():.4f}")
    lines.append(f"最小:   {probs.min():.4f}")
    lines.append(f"最大:   {probs.max():.4f}")
    lines.append(f"中位数: {np.median(probs):.4f}")
    lines.append("分位 1/5/25/50/75/95/99: " + str(np.percentile(probs,[1,5,25,50,75,95,99]).round(4).tolist()))
    hist, _ = np.histogram(probs, np.linspace(0,1,21))
    lines.append("\n直方图:")
    for i in range(20):
        bar = '#' * int(hist[i]/max(hist)*30) if hist.max()>0 else ''
        lines.append(f"  [{0.05*i:.2f}-{0.05*(i+1):.2f}]: {hist[i]:3d} {bar}")

    # 2. 负样本（CIFAR前100张，加载快）
    cifar_dir = "/data/home/shujia/CHD/model_train/RETFound_MAE-main/fundus_gate/negative_samples/cifar100"
    neg_paths = sorted(Path(cifar_dir).glob("*.png"))[:100] if os.path.exists(cifar_dir) else []
    neg_paths = [str(p) for p in neg_paths]

    neg_probs = []
    for i, p in enumerate(neg_paths):
        _, pr = checker.check(p)
        neg_probs.append(pr)
    neg_probs = np.array(neg_probs) if neg_probs else np.array([])

    lines.append("\n" + "=" * 60)
    lines.append("负样本（CIFAR-100 前100张）概率分布")
    lines.append("=" * 60)
    if len(neg_probs) > 0:
        lines.append(f"样本数: {len(neg_probs)}  均值: {neg_probs.mean():.4f}  最小: {neg_probs.min():.4f}  最大: {neg_probs.max():.4f}")

    # 3. jAccount Logo
    logo = "/data/home/shujia/.cursor/projects/data-home-shujia-CHD-model-train-RETFound-MAE-main/assets/__2026-03-24_05.59.34-fde364bf-56bc-4983-926b-488ff56deb0c.png"
    lines.append("\n" + "=" * 60)
    lines.append("jAccount Logo（以假乱真）")
    lines.append("=" * 60)
    if os.path.exists(logo):
        _, prob = checker.check(logo)
        lines.append(f"预测概率: {prob:.4f}")
        lines.append(f"判定: {'眼底' if prob>=0.5 else '非眼底'}")
    else:
        lines.append("Logo 文件不存在")

    text = "\n".join(lines)
    with open(OUT_FILE, "w") as f:
        f.write(text)
    print(text)
    print(f"\n结果已保存到: {OUT_FILE}")

if __name__ == "__main__":
    main()
