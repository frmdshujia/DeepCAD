"""
CPU 单张图片推理耗时与算力测试脚本
用法:
  python benchmark_cpu.py --checkpoint <checkpoint.pth> --image <image.jpg> [--runs 10]
"""
import argparse
import time
import os
import sys
import psutil
import threading
import statistics

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

# 确保能 import 项目代码
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
import models_vit

# ─── 参数 ───────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=str,
    default='/data/home/shujia/CHD/result_retfound/basic_classify/'
            'finetune_train_UKB_SDPP_focal05_seed42_ep60/'
            'UKB_SDPP_focal05_seed42_ep60checkpoint-best.pth',
    help='模型 checkpoint 路径')
parser.add_argument('--image', type=str, default=None,
    help='测试图片路径（若不提供则自动生成随机噪声图片）')
parser.add_argument('--runs', type=int, default=10, help='重复推理次数（取平均）')
parser.add_argument('--warmup', type=int, default=3,  help='预热次数（不计入统计）')
args = parser.parse_args()

# ─── 全局强制 CPU ────────────────────────────────────────
os.environ['CUDA_VISIBLE_DEVICES'] = ''
device = torch.device('cpu')

# ─── CPU 监控线程 ────────────────────────────────────────
cpu_samples = []
stop_monitor = threading.Event()

def _monitor():
    proc = psutil.Process(os.getpid())
    while not stop_monitor.is_set():
        cpu_samples.append(proc.cpu_percent(interval=0.2))

monitor_thread = threading.Thread(target=_monitor, daemon=True)

# ─── 图像预处理 ──────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def load_image(path):
    if path and os.path.exists(path):
        img = Image.open(path).convert('RGB')
        print(f"[图片] 使用文件: {path}  原始尺寸: {img.size}")
    else:
        import numpy as np
        arr = (np.random.rand(224, 224, 3) * 255).astype('uint8')
        img = Image.fromarray(arr)
        print("[图片] 未提供图片路径，使用随机噪声图片 (224x224)")
    return transform(img).unsqueeze(0)   # [1, 3, 224, 224]

# ─── 加载模型 ────────────────────────────────────────────
print("\n" + "="*60)
print("  RETFound CPU 单张推理算力测试")
print("="*60)

print(f"\n[1] 加载模型权重: {args.checkpoint}")
t0 = time.time()

model = models_vit.vit_large_patch16(
    num_classes=2,
    drop_path_rate=0.2,
    global_pool=False,
)

ckpt = torch.load(args.checkpoint, map_location='cpu')
state_dict = ckpt['model'] if 'model' in ckpt else ckpt
model.load_state_dict(state_dict)
model.to(device)
model.eval()

load_time = time.time() - t0
print(f"    ✓ 加载耗时: {load_time:.2f} s")

# 模型参数量
n_params = sum(p.numel() for p in model.parameters())
n_params_m = n_params / 1e6

# ─── 准备图片 ────────────────────────────────────────────
print(f"\n[2] 准备输入图片")
tensor = load_image(args.image).to(device)
print(f"    输入张量形状: {list(tensor.shape)}")

# ─── 预热 ───────────────────────────────────────────────
print(f"\n[3] 预热 {args.warmup} 次...")
with torch.no_grad():
    for _ in range(args.warmup):
        _ = model(tensor)
print("    ✓ 预热完成")

# ─── 正式测试 ────────────────────────────────────────────
print(f"\n[4] 正式推理 {args.runs} 次（CPU）...")
proc = psutil.Process(os.getpid())
mem_before = proc.memory_info().rss / 1024**2

# 启动 CPU 监控
cpu_samples.clear()
stop_monitor.clear()
monitor_thread = threading.Thread(target=_monitor, daemon=True)
monitor_thread.start()

latencies = []
with torch.no_grad():
    for i in range(args.runs):
        t_s = time.perf_counter()
        output = model(tensor)
        t_e = time.perf_counter()
        latencies.append((t_e - t_s) * 1000)   # ms

stop_monitor.set()
monitor_thread.join(timeout=2)

mem_after = proc.memory_info().rss / 1024**2

# ─── 输出结果 ────────────────────────────────────────────
prob = torch.softmax(output, dim=1)[0, 1].item()

print("\n" + "="*60)
print("           推理测试清单 (CPU Only)")
print("="*60)

print("\n【模型信息】")
print(f"  模型结构         : ViT-Large / patch16 (RETFound)")
print(f"  参数量           : {n_params_m:.1f} M ({n_params:,} 个)")
print(f"  权重文件大小     : {os.path.getsize(args.checkpoint)/1024**3:.2f} GB")
print(f"  权重加载耗时     : {load_time:.2f} s")

print("\n【输入信息】")
print(f"  输入尺寸         : 224 × 224 × 3")
print(f"  Batch Size       : 1（单张推理）")
print(f"  Patch 数量       : 196（= 14×14）")

print("\n【推理耗时统计】")
print(f"  测试轮次         : {args.runs} 次")
print(f"  平均耗时         : {statistics.mean(latencies):.1f} ms")
print(f"  中位数耗时       : {statistics.median(latencies):.1f} ms")
print(f"  最小耗时         : {min(latencies):.1f} ms")
print(f"  最大耗时         : {max(latencies):.1f} ms")
print(f"  标准差           : {statistics.stdev(latencies):.1f} ms  (稳定性参考)")
print(f"  吞吐量 (QPS)     : {1000/statistics.mean(latencies):.2f} 张/秒")

print("\n【CPU 与内存占用】")
n_cpu = psutil.cpu_count(logical=True)
print(f"  系统逻辑核心数   : {n_cpu}")
if cpu_samples:
    print(f"  推理期间 CPU 占用: 均值 {statistics.mean(cpu_samples):.1f}%  峰值 {max(cpu_samples):.1f}%")
    print(f"    (单进程最大可用 {n_cpu*100:.0f}%，实际占用 {statistics.mean(cpu_samples)/n_cpu:.1f}% 等效核心数"
          f" ≈ {statistics.mean(cpu_samples)/100:.1f} 核)")
print(f"  进程内存 (RSS)   : 推理前 {mem_before:.0f} MB → 推理后 {mem_after:.0f} MB"
      f"  (增量 {mem_after-mem_before:.0f} MB)")
total_mem = psutil.virtual_memory().total / 1024**3
used_mem  = psutil.virtual_memory().used  / 1024**3
print(f"  系统内存总量     : {total_mem:.1f} GB  当前已用 {used_mem:.1f} GB")

print("\n【预测结果示例】")
print(f"  冠心病阳性概率   : {prob:.4f}")
print(f"  预测标签 (thr=0.5): {'阳性 (CHD)' if prob >= 0.5 else '阴性 (Non-CHD)'}")

print("\n【服务器选型建议（纯 CPU 部署参考）】")
mean_ms = statistics.mean(latencies)
print(f"  单张推理均值 {mean_ms:.0f} ms → 预估并发 10 路需 {mean_ms/100:.1f}s/批")
print(f"  建议 CPU 核心   : ≥ 8 核（推荐 16 核以上以保证响应速度）")
print(f"  建议内存容量   : ≥ {max(16, int(mem_after/1024+4)*4)} GB")
print(f"  GPU 可选但非必须: 若加 GPU，推理耗时可降至 ~30-80 ms/张")

print("\n" + "="*60)
print("  测试完成")
print("="*60 + "\n")
