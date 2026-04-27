"""
多卡并行推理脚本 (DataLoader 版，多线程 I/O 加速)
- 4 x RTX 3090, 每卡负责约 1/4 数据
- 输出: results/gpu{i}.csv  (dataset, path, prob)
- 推理完成后自动合并并生成 dist_data.json

用法:
  python run_inference.py

TTA（可选；默认与对比学习几何 7 视角一致，约 7× 前向）:
  export INFER_USE_TTA=1
  export INFER_TTA_MODES=orig,hflip,vflip,rot90,rot180,rot270   # 可省略，与 util.tta.DEFAULT_TTA_MODES 一致
  python run_inference.py
"""

import os, sys, pickle, csv, json, time
import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision import transforms
from tqdm import tqdm

# ─── 路径配置 ─────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT = (
    "/data/home/shujia/CHD/result_retfound/basic_classify/"
    "finetune_train_UKB_SDPP_focal05/UKB_SDPP_1to1_focal05checkpoint-best.pth"
)
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT_DIR, exist_ok=True)

DATASET_PKL = {
    "UKB":  "/data/home/shujia/dataset/CHD/basic_classify/UKB/pictures-individual.pkl",
    "SDPP": "/data/home/shujia/dataset/CHD/basic_classify/SDPP/pictures-individual.pkl",
    "SHCC": "/data/home/shujia/dataset/CHD/basic_classify/SHCC/pictures-individual.pkl",
    "WHTM": "/data/home/shujia/dataset/CHD/basic_classify/WHTM/pictures-individual.pkl",
    "PUDM": "/data/home/shujia/dataset/CHD/basic_classify/PUDM/pictures-individual.pkl",
}

BATCH_SIZE   = 64     # 每 GPU 批次大小
IO_WORKERS   = 4      # DataLoader 读图线程数；4 GPU × 4 = 16 并发读（减少竞争）
N_GPUS       = 4
BINS         = 50     # 直方图分箱数，与前端一致

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ─── Dataset ──────────────────────────────────────────────────────────────────
class FundusDataset(Dataset):
    def __init__(self, items, transform):
        """items: list of (dataset_name, img_path)"""
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        ds_name, img_path = self.items[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = self.transform(img)
            valid = 1
        except Exception:
            tensor = torch.zeros(3, 224, 224)
            valid = 0
        return tensor, ds_name, img_path, valid


def collate_fn(batch):
    tensors, ds_names, paths, valids = zip(*batch)
    return torch.stack(tensors), list(ds_names), list(paths), list(valids)


# ─── 单卡推理 worker ──────────────────────────────────────────────────────────
def worker_fn(gpu_id, work_items, out_dir, checkpoint):
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)

    # 加载模型
    sys.path.insert(0, BASE_DIR)
    import models_vit
    from util.tta import DEFAULT_TTA_MODES, forward_logits_tta, parse_tta_modes

    use_tta = os.environ.get("INFER_USE_TTA", "").lower() in ("1", "true", "yes")
    tta_modes = (
        parse_tta_modes(os.environ.get("INFER_TTA_MODES", DEFAULT_TTA_MODES))
        if use_tta
        else None
    )
    model = models_vit.vit_large_patch16(
        num_classes=2, drop_path_rate=0.2, global_pool=False,
    )
    ckpt = torch.load(checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    dataset = FundusDataset(work_items, TRANSFORM)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=IO_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=IO_WORKERS > 0,
        collate_fn=collate_fn,
    )

    out_path = os.path.join(out_dir, f"gpu{gpu_id}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "path", "prob"])

        desc = f"GPU{gpu_id}"
        with torch.no_grad():
            for tensors, ds_names, paths, valids in tqdm(
                loader, desc=desc, position=gpu_id, leave=True
            ):
                tensors = tensors.to(device, non_blocking=True)
                if tta_modes is not None:
                    out = forward_logits_tta(model, tensors, tta_modes)
                else:
                    out = model(tensors)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().tolist()
                for ds, pth, prob, valid in zip(ds_names, paths, probs, valids):
                    if valid:
                        writer.writerow([ds, pth, f"{prob:.6f}"])
                    else:
                        writer.writerow([ds, pth, ""])
                f.flush()  # 每批落盘，避免中断丢失

    print(f"GPU{gpu_id} 完成 → {out_path}")


# ─── 聚合 → dist_data.json ────────────────────────────────────────────────────
def merge_and_compute_dist(out_dir):
    ukb_probs, cn_probs = [], []
    CN_SETS = {"SDPP", "SHCC", "WHTM", "PUDM"}

    for f in sorted(Path(out_dir).glob("gpu*.csv")):
        with open(f) as fp:
            for row in csv.DictReader(fp):
                if not row["prob"]:
                    continue
                prob = float(row["prob"])
                ds   = row["dataset"]
                if ds == "UKB":
                    ukb_probs.append(prob)
                elif ds in CN_SETS:
                    cn_probs.append(prob)

    print(f"\n聚合: UKB={len(ukb_probs):,}, 中国队列={len(cn_probs):,}")

    def to_hist(probs):
        arr = np.array(probs)
        hist, _ = np.histogram(arr, bins=BINS, range=(0.0, 1.0))
        return hist.tolist()

    data = {
        "bins":  BINS,
        "ukb":   to_hist(ukb_probs),
        "cn":    to_hist(cn_probs),
        "ukb_n": len(ukb_probs),
        "cn_n":  len(cn_probs),
    }

    out_json = os.path.join(os.path.dirname(out_dir), "dist_data.json")
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"分布数据: {out_json}")

    for name, probs in [("UKB (Europe)", ukb_probs), ("中国队列", cn_probs)]:
        if probs:
            arr = np.array(probs)
            print(f"  {name}: mean={arr.mean():.4f}, std={arr.std():.4f}, "
                  f"median={np.median(arr):.4f}")
    return out_json


# ─── 主函数 ───────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("多卡并行推理 (DataLoader + 多线程 I/O)")
    print(f"  权重:       {CHECKPOINT}")
    print(f"  GPUs:       {N_GPUS}")
    print(f"  Batch/GPU:  {BATCH_SIZE}")
    print(f"  IO workers: {IO_WORKERS}")
    if os.environ.get("INFER_USE_TTA", "").lower() in ("1", "true", "yes"):
        sys.path.insert(0, BASE_DIR)
        from util.tta import DEFAULT_TTA_MODES as _DEF_TTA
        print(f"  TTA:        ON | modes={os.environ.get('INFER_TTA_MODES', _DEF_TTA)}")
    else:
        print("  TTA:        off (export INFER_USE_TTA=1 开启)")
    print("=" * 60)

    # 1. 加载图像路径
    print("\n加载图像路径 ...")
    all_items = []
    for ds_name, pkl_path in DATASET_PKL.items():
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        paths = [p for p in data.keys() if os.path.exists(p)]
        print(f"  {ds_name}: {len(paths):,} 张")
        for p in paths:
            all_items.append((ds_name, p))
    print(f"总计: {len(all_items):,} 张")

    # 2. 按 GPU 切分（循环分配保证均匀）
    chunks = [all_items[i::N_GPUS] for i in range(N_GPUS)]
    for i, c in enumerate(chunks):
        print(f"  GPU{i}: {len(c):,} 张")

    # 3. 多进程启动
    print("\n启动推理进程 ...")
    t0 = time.time()
    processes = []
    for gpu_id in range(N_GPUS):
        p = mp.Process(
            target=worker_fn,
            args=(gpu_id, chunks[gpu_id], OUT_DIR, CHECKPOINT),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    elapsed = time.time() - t0
    print(f"\n推理完成，耗时 {elapsed/60:.1f} 分钟")

    # 4. 聚合并生成 JSON
    out_json = merge_and_compute_dist(OUT_DIR)
    print(f"\n下一步: python infer_dist/update_web_dist.py --dist {out_json}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
