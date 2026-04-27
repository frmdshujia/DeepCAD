# 双塔跨模态心血管预测框架：完整技术方案

> 眼底（RETFound）× 心脏MRI（MedSAM）双向知识迁移  
> 基于 UK Biobank 数据

---

## 目录

1. [背景与核心思路](#1-背景与核心思路)
2. [数据配置](#2-数据配置)
3. [任务设计](#3-任务设计)
4. [整体架构](#4-整体架构)
5. [模型代码](#5-模型代码)
6. [Loss设计](#6-loss设计)
7. [DataLoader设计](#7-dataloader设计)
8. [三阶段训练策略](#8-三阶段训练策略)
9. [消融实验设计](#9-消融实验设计)
10. [评估指标](#10-评估指标)
11. [今日预实验计划](#11-今日预实验计划)
12. [预实验代码](#12-预实验代码)
13. [存储与路径规范](#13-存储与路径规范)

---

## 1. 背景与核心思路

### 1.1 问题背景

- **目标**：给 RETFound（眼底基础模型）注入心脏相关知识，使其能更好地预测心血管疾病
- **核心挑战**：眼底和心脏MRI是两个完全不同的模态，直接做 instance-level InfoNCE 对比学习，模型会优先对齐年龄、性别、体型等混杂因素，而不是心血管特异性特征
- **解决思路**：让 CMR 编码器先通过有监督任务变得 cardiac-specific，再让眼底编码器向这个"动态教师"对齐，同时通过双向 Cross-Attention 让两个模态互相提取对自己有用的信息

### 1.2 核心设计理念

```
传统做法（有问题）：
  RETFound <-> MedSAM
  直接InfoNCE对齐 -> 学到的是混杂因素（年龄/性别/体型）

本方案（改进）：
  CMR塔：有监督任务驱动 -> 变得cardiac-specific
  眼底塔：向cardiac-specific的CMR对齐
  双向交互：各自提取对方有用的信息
  LoRA回流：跨模态知识稳定注入主干，不破坏原有权重
```

### 1.3 为什么用 LoRA 作为知识回流通道

跨模态学到的知识是"外来的"，直接加到全参微调的梯度里会和单模态任务的梯度打架。LoRA 作为低秩旁路，让跨模态知识以低干扰的方式叠加进主干。主干的主要梯度来自单模态任务，跨模态知识是增量式地渗透进来的——类似于"使者带回知识，通过外交渠道而非直接改写政策"。

---

## 2. 数据配置

### 2.1 原始人群规模

| 人群 | 原始人数 | 说明 |
|------|---------|------|
| CMR人群 | 96,784人 | stage1_cmr.csv |
| 眼底人群 | 98,777人 | stage1_fundus.csv |
| 宽松配对池（跨visit） | 33,666人 | 交互支路训练 |
| 严格配对（同visit） | 2,892人 | 验证/测试 |
| 眼底→LVEF可用（跨visit口径A） | 30,773人 | 眼底回归任务 |
| 眼底→LVEF可用（同visit口径B） | 2,457人 | 严格对齐实验 |

### 2.2 阴性下采样策略（加速训练）

**核心原则：阳性样本全量保留，阴性样本抽取10,000人。**

原始数据规模过大（CMR约9.7万，眼底约9.9万），全量训练时间难以承受。
阴性样本下采样到1万，在保证统计充分性的前提下将训练时间压缩约4-5倍。

**采样优先级（三级）：**

```
第一优先级：任意分类任务阳性的人 -> 全部保留
            （composite_ischemic_hd / prevalent_I21 /
              composite_cardiomyopathy_hf / composite_af_arrhythmia
              任意一个为1即保留）

第二优先级：分类全阴性 但 有LVEF/LVEDV回归标签的人 -> 优先进入阴性池
            （保证回归任务的训练数据不因下采样大量流失）

第三优先级：分类全阴性 且 无回归标签的人 -> 补充到凑满10,000阴性
```

**采样代码：**

```python
# build_training_csv.py
# 对CMR和眼底人群分别执行，生成训练用的下采样csv

import pandas as pd
import numpy as np

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]
REG_COL = 'LV ejection fraction'   # 有这列非空即视为有回归标签
N_NEG   = 10_000
SEED    = 42


def smart_downsample(df, n_neg=N_NEG, cls_cols=CLS_COLS,
                     reg_col=REG_COL, seed=SEED):
    """
    三优先级采样：
    1. 分类阳性 -> 全保留
    2. 有回归标签的阴性 -> 优先入池
    3. 纯阴性 -> 补充
    """
    df = df.copy()

    # 判断是否为分类阳性（任意一个任务阳性）
    valid_cls = df[cls_cols].isin([0, 1])   # 排除-1缺失值
    is_pos = (df[cls_cols] == 1).any(axis=1)

    has_reg  = df[reg_col].notna() if reg_col in df.columns else pd.Series(False, index=df.index)

    pos_df          = df[is_pos]                    # 全保留
    neg_with_reg    = df[~is_pos & has_reg]          # 有回归标签的阴性
    neg_pure        = df[~is_pos & ~has_reg]         # 纯阴性

    n_from_reg  = min(len(neg_with_reg), n_neg)
    n_from_pure = max(0, n_neg - n_from_reg)

    neg_sampled = pd.concat([
        neg_with_reg.sample(n_from_reg,  random_state=seed),
        neg_pure.sample(min(n_from_pure, len(neg_pure)), random_state=seed),
    ])

    final = pd.concat([pos_df, neg_sampled]).sample(frac=1, random_state=seed)
    final = final.reset_index(drop=True)

    # 打印采样结果
    print(f'阳性（全保留）:           {len(pos_df):>7,}人')
    print(f'阴性-有回归标签:          {n_from_reg:>7,}人')
    print(f'阴性-纯阴性补充:          {n_from_pure:>7,}人')
    print(f'总计:                    {len(final):>7,}人')
    print()
    for col in cls_cols:
        n_pos = (final[col] == 1).sum()
        rate  = n_pos / len(final)
        print(f'  {col:<40} 阳性: {n_pos:,} ({rate:.2%})')

    return final


# 执行采样
if __name__ == '__main__':
    # CMR人群
    print('=== CMR人群下采样 ===')
    cmr = pd.read_csv('stage1_cmr.csv', low_memory=False)
    cmr_sampled = smart_downsample(cmr)
    cmr_sampled.to_csv('stage1_cmr_sampled.csv', index=False)

    # 眼底人群
    print('\n=== 眼底人群下采样 ===')
    fundus = pd.read_csv('stage1_fundus.csv', low_memory=False)
    fundus_sampled = smart_downsample(fundus)
    fundus_sampled.to_csv('stage1_fundus_sampled.csv', index=False)
```

### 2.3 下采样后预期人群规模

> ⚠️ **Agent任务**：执行上述采样脚本后，填入以下表格的实际数字。

| 人群 | 预期规模 | 实际规模 |
|------|---------|----------------|
| CMR训练集（下采样后） | 约18,000-20,000人 | **21,599人** |
| 眼底训练集（下采样后） | 约18,000-19,000人 | **19,122人** |
| CMR中分类阳性人数 | 约8,000-10,000人 | **11,599人** |
| 眼底中分类阳性人数 | 约8,000-9,000人 | **9,122人** |
| CMR中有回归标签人数 | 约17,000-19,000人 | **19,669人** |
| 眼底中有回归标签人数 | **不确定** | **30,781人**（眼底∩CMR交集33,666人中有LV标签者，与文档预估30,773人吻合） |

### 2.4 下采样后pos_weight重新计算

> ⚠️ **Agent任务**：采样完成后，对CMR和眼底两个采样csv分别运行以下代码，
> 将输出的pos_weight填入下表，并保存为`pos_weights_cmr.json`和`pos_weights_fundus.json`，
> 训练代码直接读取这两个文件，不要硬编码。

```python
# calc_pos_weights.py
import pandas as pd
import json

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]

def calc_pos_weights(csv_path, out_path):
    df = pd.read_csv(csv_path, low_memory=False)
    weights = {}
    print(f'\n{csv_path} 的 pos_weight:')
    for col in CLS_COLS:
        if col not in df.columns:
            print(f'  {col}: 列不存在')
            continue
        n_pos = (df[col] == 1).sum()
        n_neg = (df[col] == 0).sum()
        pw = n_neg / n_pos if n_pos > 0 else 1.0
        weights[col] = round(pw, 2)
        print(f'  {col}: pos={n_pos}, neg={n_neg}, pos_weight={pw:.2f}')

    with open(out_path, 'w') as f:
        json.dump(weights, f, indent=2)
    print(f'已保存到 {out_path}')
    return weights

calc_pos_weights('stage1_cmr_sampled.csv',    'pos_weights_cmr.json')
calc_pos_weights('stage1_fundus_sampled.csv', 'pos_weights_fundus.json')
```

**下采样后各任务pos_weight：**

| 任务 | 原始pos_weight | CMR下采样后 | 眼底下采样后 |
|------|--------------|-----------------|-----------------|
| composite_ischemic_hd | ~14.3 | **2.39** | **2.29** |
| prevalent_I21 | ~39.8 | **8.10** | **7.17** |
| composite_cardiomyopathy_hf | ~95.2 | **20.32** | **23.27** |
| composite_af_arrhythmia | ~14.5 | **2.45** | **3.66** |

### 2.5 配对人群时间间隔分布（不做下采样）

配对人群（33,666人）**不做下采样**，全量用于Stage 2和Stage 3的交互支路训练。
配对样本本来就是稀缺资源，不能再减少。

| 统计项 | 数值 |
|--------|------|
| 中位数 | 5.92年 |
| P25 | 3.70年 |
| P75 | 9.53年 |
| 间隔≤2年 | 11.42%（3,845人） |
| 间隔2-5年 | 23.97%（8,068人） |
| 间隔>5年 | 64.61%（21,746人） |

> **时间权重**：间隔>5年的配对样本Loss权重乘以0.5，降低噪声影响但不丢弃数据。

### 2.6 回归标签归一化参数

> ⚠️ **Agent任务**：归一化参数必须从**下采样后的训练集**重新计算（不能用原始全量数据），
> 运行以下代码后将结果填入表格并保存为`norm_params.json`。

```python
# calc_norm_params.py
import pandas as pd
import json

REG_COLS = {
    'lvef':  'LV ejection fraction',
    'lvedv': 'LV end diastolic volume',
    'lvesv': 'LV end systolic volume',
}

# 从CMR下采样训练集计算（只用训练集，不能包含验证/测试集）
df = pd.read_csv('stage1_cmr_sampled_train.csv', low_memory=False)

norm_params = {}
for key, col in REG_COLS.items():
    if col not in df.columns:
        print(f'{col}: 列不存在')
        continue
    vals = pd.to_numeric(df[col], errors='coerce').dropna()
    norm_params[key] = {
        'mean': round(float(vals.mean()), 4),
        'std':  round(float(vals.std()),  4),
        'min':  round(float(vals.min()),  4),
        'max':  round(float(vals.max()),  4),
        'n':    int(vals.count()),
    }
    print(f'{key}: mean={norm_params[key]["mean"]:.2f}, '
          f'std={norm_params[key]["std"]:.2f}, n={norm_params[key]["n"]:,}')

with open('norm_params.json', 'w') as f:
    json.dump(norm_params, f, indent=2)
print('\n已保存到 norm_params.json')
```

**归一化参数（待Agent用下采样训练集重新计算后填入）：**

| 指标 | 参考值（全量） | 下采样训练集实际值（n=15,733） |
|------|-------------|----------------------|
| LVEF 均值±SD | 59.6±6.4 | **58.72±7.37** |
| LVEDV 均值±SD | 145.9±34.1 | **149.56±36.11** |
| LVESV 均值±SD | 59.6±19.9 | **62.62±22.85** |

> **重要**：`norm_params.json`必须在训练开始前生成并固定，测试时用同一套参数反归一化。

---

## 3. 任务设计

### 3.1 分类任务（两塔共享标签定义）

> ⚠️ **Agent任务**：下表的原始阳性率基于全量数据。下采样后阳性率会升高，
> pos_weight会降低。请执行2.4节的`calc_pos_weights.py`后将实际值填入2.4节的表格，
> 训练代码从`pos_weights_cmr.json`和`pos_weights_fundus.json`读取，不要使用下表的参考值。

| 任务 | 列名 | CMR阳性率（全量参考） | 眼底阳性率（全量参考） | 配对阳性率 |
|------|------|-------------------|-------------------|----------|
| 缺血性心脏病 | composite_ischemic_hd | 6.55% | 5.83% | 7.02% |
| 心肌梗死 | prevalent_I21 | 2.45% | 2.35% | 2.70% |
| 心肌病/心衰 | composite_cardiomyopathy_hf | 1.04% | 0.78% | 1.28% |
| 房颤/心律失常 | composite_af_arrhythmia | 6.45% | 4.16% | 6.78% |

### 3.2 回归任务

| 任务 | 列名 | CMR可用人数 | 眼底可用（口径A） |
|------|------|-----------|----------------|
| 射血分数 | LV ejection fraction | 85,878人 | 30,773人 |
| 舒张末容积 | LV end diastolic volume | 85,878人 | 30,773人 |
| 收缩末容积 | LV end systolic volume | 85,878人 | 30,773人 |

### 3.3 两塔任务分配

```
CMR塔（7个任务，85,878人有回归标签）:
├─ 分类x4: ischemic_hd / I21 / cardiomyopathy_hf / af_arrhythmia
├─ 回归: LVEF
├─ 回归: LVEDV
└─ 回归: LVESV

眼底塔（7个任务，回归标签仅30,773人有，其余mask）:
├─ 分类x4: 同上（全量98,777人参与）
├─ 回归: LVEF   （30,773人有标签，无标签样本mask掉这个loss）
├─ 回归: LVEDV  （同上）
└─ 回归: LVESV  （同上）
```

### 3.4 暂不纳入的任务及原因

| 任务 | 原因 |
|------|------|
| prevalent_I50（心衰） | 与composite_cardiomyopathy_hf高度重叠 |
| composite_valvular_hd | CMR单帧难诊断瓣膜病，信号弱 |
| composite_hypertensive_hd | 阳性率仅0.03%，样本极少（配对人群仅3人） |
| 生存分析/incident | 工程复杂度高，留作后续工作 |

---

## 4. 整体架构

```
+------------------------------------------------------------------+
|                        双塔跨模态框架                             |
+----------------------+-------------------------------------------+
|       眼底塔         |              CMR塔                         |
|   RETFound(ViT-L)    |          MedSAM(ViT-B)                    |
|      1024-dim        |            768-dim                        |
|      全参微调         |            全参微调                        |
|      98,777人        |            96,784人                       |
+----------------------+-------------------------------------------+
|                   配对支路（33,666人）                             |
|                                                                  |
|   z_fundus(1024) --proj--> hidden(512)                          |
|                               |                                  |
|                    双向Cross-Attention                            |
|                    + Sigmoid Gate控制                             |
|                               |                                  |
|   z_cmr(768)    --proj--> hidden(512)                           |
|                                                                  |
|   LoRA_F(r=16)              LoRA_C(r=16)                        |
|   知识回流->眼底塔           知识回流->CMR塔                       |
+------------------------------------------------------------------+
|                         任务头                                    |
|   眼底塔：分类x4 + 回归x3（无标签样本mask loss）                    |
|   CMR塔： 分类x4 + 回归x3                                        |
+------------------------------------------------------------------+

总Loss:
L = L_cmr_task(全量)
  + L_fundus_task(全量)
  + 0.5 * L_cmr_enriched(配对)
  + 0.5 * L_fundus_enriched(配对)
  + lambda * L_align(配对)
```

---

## 5. 模型代码

### 5.1 LoRA层

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALayer(nn.Module):
    """
    低秩知识回流层
    
    作用：跨模态知识的缓冲器，让外来知识以低干扰方式影响主干
    初始化为零输出，训练开始时不影响主干，随训练逐渐注入知识
    
    Args:
        in_dim:  输入维度
        out_dim: 输出维度
        r:       低秩秩数（建议16）
        alpha:   缩放系数，scale = alpha/r
    """
    def __init__(self, in_dim, out_dim, r=16, alpha=32):
        super().__init__()
        self.lora_A = nn.Linear(in_dim, r, bias=False)
        self.lora_B = nn.Linear(r, out_dim, bias=False)
        self.scale = alpha / r
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)  # 关键：初始化为0

    def forward(self, x):
        return self.lora_B(self.lora_A(x)) * self.scale
```

### 5.2 跨模态交互模块

```python
class CrossModalGating(nn.Module):
    """
    双向跨模态交互模块
    
    设计原理：
    1. 眼底和CMR各自投影到公共空间(512-dim)对齐维度
    2. Cross-Attention双向提取：眼底问CMR要什么，CMR问眼底要什么
    3. Sigmoid Gate控制"吸收多少外来知识"，防止早期训练不稳定
    4. LoRA将外来知识低干扰地回流到各自主干
    
    Args:
        dim_f:     眼底编码器输出维度 (RETFound ViT-L = 1024)
        dim_c:     CMR编码器输出维度 (MedSAM ViT-B = 768)
        hidden:    公共空间维度 (512)
        num_heads: 注意力头数 (8)
        r:         LoRA秩 (16)
    """
    def __init__(self, dim_f=1024, dim_c=768, hidden=512, num_heads=8, r=16):
        super().__init__()

        # 维度对齐投影
        self.proj_f = nn.Linear(dim_f, hidden)
        self.proj_c = nn.Linear(dim_c, hidden)

        # 双向Cross-Attention
        self.attn_f2c = nn.MultiheadAttention(hidden, num_heads, batch_first=True)
        self.attn_c2f = nn.MultiheadAttention(hidden, num_heads, batch_first=True)

        # LayerNorm稳定训练
        self.norm_f = nn.LayerNorm(hidden)
        self.norm_c = nn.LayerNorm(hidden)

        # Gating：sigmoid门控制吸收量
        self.gate_f = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())
        self.gate_c = nn.Sequential(nn.Linear(hidden, hidden), nn.Sigmoid())

        # 回投影（hidden -> 原始dim）
        self.back_proj_f = nn.Linear(hidden, dim_f)
        self.back_proj_c = nn.Linear(hidden, dim_c)

        # LoRA知识回流（核心：低秩、低干扰）
        self.lora_f = LoRALayer(dim_f, dim_f, r=r)
        self.lora_c = LoRALayer(dim_c, dim_c, r=r)

    def forward(self, z_f, z_c):
        """
        Args:
            z_f: (B, 1024) 眼底embedding
            z_c: (B, 768)  CMR embedding
        Returns:
            z_f_enriched: (B, 1024) 注入CMR知识后的眼底embedding
            z_c_enriched: (B, 768)  注入眼底知识后的CMR embedding
        """
        # 投影到公共空间
        zf = self.proj_f(z_f).unsqueeze(1)   # (B, 1, 512)
        zc = self.proj_c(z_c).unsqueeze(1)   # (B, 1, 512)

        # 眼底 Query CMR（眼底主动问：CMR里有什么我需要的）
        f_from_c, _ = self.attn_f2c(zf, zc, zc)
        f_from_c = self.norm_f(f_from_c.squeeze(1))   # (B, 512)
        gate_f = self.gate_f(f_from_c)
        delta_f = gate_f * f_from_c                    # 门控后的增量

        # CMR Query 眼底（CMR主动问：眼底里有什么我需要的）
        c_from_f, _ = self.attn_c2f(zc, zf, zf)
        c_from_f = self.norm_c(c_from_f.squeeze(1))   # (B, 512)
        gate_c = self.gate_c(c_from_f)
        delta_c = gate_c * c_from_f

        # 回投影到原始维度
        delta_f = self.back_proj_f(delta_f)            # (B, 1024)
        delta_c = self.back_proj_c(delta_c)            # (B, 768)

        # LoRA知识回流：低秩、低干扰地影响主干
        z_f_enriched = z_f + self.lora_f(delta_f)
        z_c_enriched = z_c + self.lora_c(delta_c)

        return z_f_enriched, z_c_enriched
```

### 5.3 多任务头

```python
class TaskHead(nn.Module):
    """
    多任务头：分类 + 回归
    
    Args:
        in_dim:  输入维度
        n_cls:   分类任务数 (4)
        n_reg:   回归任务数 (3)
        dropout: dropout率 (0.3)
    """
    def __init__(self, in_dim, n_cls=4, n_reg=3, dropout=0.3):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        # 每个分类任务独立头（多标签，各自sigmoid）
        self.cls_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(n_cls)
        ])

        # 每个回归任务独立头
        self.reg_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ) for _ in range(n_reg)
        ])

    def forward(self, z):
        z = self.dropout(z)
        cls_out = [h(z).squeeze(-1) for h in self.cls_heads]  # list of (B,)
        reg_out = [h(z).squeeze(-1) for h in self.reg_heads]  # list of (B,)
        return cls_out, reg_out
```

### 5.4 完整双塔模型

```python
class DualTowerModel(nn.Module):
    """
    双塔跨模态心血管预测模型
    
    支持三种forward模式：
    - 'fundus_only': 眼底单模态（大人群，走主干+任务头）
    - 'cmr_only':    CMR单模态（大人群，走主干+任务头）
    - 'paired':      配对样本（同时走两塔，经过Cross-Attention交互）
    
    Args:
        fundus_encoder: RETFound ViT-L（已加载预训练权重）
        cmr_encoder:    MedSAM ViT-B（已加载预训练权重）
    """
    def __init__(self, fundus_encoder, cmr_encoder, dim_f=1024, dim_c=768):
        super().__init__()
        self.fundus_encoder = fundus_encoder
        self.cmr_encoder    = cmr_encoder

        # 跨模态交互模块
        self.cross_modal = CrossModalGating(
            dim_f=dim_f, dim_c=dim_c,
            hidden=512, num_heads=8, r=16
        )

        # 任务头（两塔各自独立）
        self.head_fundus = TaskHead(in_dim=dim_f, n_cls=4, n_reg=3)
        self.head_cmr    = TaskHead(in_dim=dim_c, n_cls=4, n_reg=3)

        # 对齐投影（降维到256，用于cosine对齐loss）
        self.align_proj_f = nn.Sequential(
            nn.Linear(dim_f, 256), nn.ReLU(), nn.Linear(256, 256)
        )
        self.align_proj_c = nn.Sequential(
            nn.Linear(dim_c, 256), nn.ReLU(), nn.Linear(256, 256)
        )

    def encode_fundus(self, img):
        return self.fundus_encoder(img)   # (B, 1024)

    def encode_cmr(self, npy):
        return self.cmr_encoder(npy)      # (B, 768)

    def forward(self, batch):
        results = {}
        mode = batch['mode']

        if mode == 'fundus_only':
            z_f = self.encode_fundus(batch['fundus'])
            cls, reg = self.head_fundus(z_f)
            results.update({'fundus_cls': cls, 'fundus_reg': reg})

        elif mode == 'cmr_only':
            z_c = self.encode_cmr(batch['cmr'])
            cls, reg = self.head_cmr(z_c)
            results.update({'cmr_cls': cls, 'cmr_reg': reg})

        elif mode == 'paired':
            z_f = self.encode_fundus(batch['fundus'])
            z_c = self.encode_cmr(batch['cmr'])

            # 基础预测（交互前，确保即使交互层失效也有梯度）
            cls_f0, reg_f0 = self.head_fundus(z_f)
            cls_c0, reg_c0 = self.head_cmr(z_c)
            results.update({
                'fundus_cls_base': cls_f0, 'fundus_reg_base': reg_f0,
                'cmr_cls_base':    cls_c0, 'cmr_reg_base':    reg_c0,
            })

            # 跨模态交互
            z_f_en, z_c_en = self.cross_modal(z_f, z_c)

            # 交互后预测
            cls_f1, reg_f1 = self.head_fundus(z_f_en)
            cls_c1, reg_c1 = self.head_cmr(z_c_en)
            results.update({
                'fundus_cls_enriched': cls_f1, 'fundus_reg_enriched': reg_f1,
                'cmr_cls_enriched':    cls_c1, 'cmr_reg_enriched':    reg_c1,
            })

            # 对齐embedding（L2归一化后计算cosine相似度）
            results['align_f'] = F.normalize(self.align_proj_f(z_f), dim=-1)
            results['align_c'] = F.normalize(self.align_proj_c(z_c), dim=-1)

        return results
```

---

## 6. Loss设计

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# ── 从文件读取pos_weight（下采样后重新计算，不能硬编码）──
# Agent任务：确保以下两个文件在训练开始前已生成
# pos_weights_cmr.json    -> CMR塔使用
# pos_weights_fundus.json -> 眼底塔使用
# 文件格式示例：
# {
#   "composite_ischemic_hd": 1.6,
#   "prevalent_I21": 4.2,
#   "composite_cardiomyopathy_hf": 10.0,
#   "composite_af_arrhythmia": 1.6
# }

CLS_COLS = [
    'composite_ischemic_hd',
    'prevalent_I21',
    'composite_cardiomyopathy_hf',
    'composite_af_arrhythmia',
]

def load_pos_weights(json_path):
    with open(json_path) as f:
        d = json.load(f)
    return [d[col] for col in CLS_COLS]

def load_norm_params(json_path='norm_params.json'):
    with open(json_path) as f:
        d = json.load(f)
    return [
        {'mean': d['lvef']['mean'],  'std': d['lvef']['std']},
        {'mean': d['lvedv']['mean'], 'std': d['lvedv']['std']},
        {'mean': d['lvesv']['mean'], 'std': d['lvesv']['std']},
    ]

# 使用示例：
# cmr_pos_weights    = load_pos_weights('pos_weights_cmr.json')
# fundus_pos_weights = load_pos_weights('pos_weights_fundus.json')
# reg_norm           = load_norm_params('norm_params.json')


class MultiTaskLoss(nn.Module):
    def __init__(self, pos_weights=POS_WEIGHTS, reg_norm=REG_NORM):
        super().__init__()
        self.cls_criteria = nn.ModuleList([
            nn.BCEWithLogitsLoss(pos_weight=torch.tensor([w]))
            for w in pos_weights
        ])
        self.reg_norm = reg_norm

    def cls_loss(self, preds, labels):
        """
        preds:  list of (B,) logits，每个任务一个
        labels: (B, 4) float，-1表示标签缺失（跳过这条样本）
        """
        loss = torch.tensor(0.0, device=preds[0].device)
        for i, (pred, crit) in enumerate(zip(preds, self.cls_criteria)):
            valid = labels[:, i] >= 0
            if valid.sum() == 0:
                continue
            loss = loss + crit(pred[valid], labels[valid, i])
        return loss

    def reg_loss(self, preds, targets, reg_mask=None):
        """
        targets:  (B, 3) 原始值（未归一化），NaN表示无标签
        reg_mask: (B, 3) bool，True=有标签（眼底侧需要传入）
        """
        loss = torch.tensor(0.0, device=preds[0].device)
        for i, pred in enumerate(preds):
            mu    = self.reg_norm[i]['mean']
            sigma = self.reg_norm[i]['std']
            target_norm = (targets[:, i] - mu) / sigma
            valid = ~torch.isnan(targets[:, i])
            if reg_mask is not None:
                valid = valid & reg_mask[:, i]
            if valid.sum() == 0:
                continue
            loss = loss + F.mse_loss(pred[valid], target_norm[valid])
        return loss

    def align_loss(self, z_f, z_c):
        """cosine对齐loss，同一人的眼底和CMR embedding靠近"""
        return 1.0 - (z_f * z_c).sum(dim=-1).mean()

    def compute(self, results, batch, lambdas):
        """
        主loss计算入口
        
        lambdas: dict，如 {'align': 0.1}
        配对样本时间权重：>5年间隔降权到0.5（在DataLoader里预处理）
        """
        total = torch.tensor(0.0)
        mode  = batch['mode']
        tw    = batch.get('time_weight', 1.0)   # 时间间隔权重

        if mode == 'fundus_only':
            total = total + self.cls_loss(results['fundus_cls'], batch['cls_labels'])
            total = total + self.reg_loss(results['fundus_reg'], batch['reg_labels'],
                                          batch.get('reg_mask'))

        elif mode == 'cmr_only':
            total = total + self.cls_loss(results['cmr_cls'],  batch['cls_labels'])
            total = total + self.reg_loss(results['cmr_reg'],  batch['reg_labels'])

        elif mode == 'paired':
            # 基础单模态loss（不加时间权重，保证基础梯度稳定）
            total = total + self.cls_loss(results['fundus_cls_base'], batch['cls_labels'])
            total = total + self.cls_loss(results['cmr_cls_base'],    batch['cls_labels'])
            total = total + self.reg_loss(results['cmr_reg_base'],    batch['reg_labels'])
            total = total + self.reg_loss(results['fundus_reg_base'], batch['reg_labels'],
                                          batch.get('reg_mask'))

            # 交互后loss（权重0.5 x 时间权重）
            w = 0.5 * tw
            total = total + w * self.cls_loss(results['fundus_cls_enriched'], batch['cls_labels'])
            total = total + w * self.cls_loss(results['cmr_cls_enriched'],    batch['cls_labels'])
            total = total + w * self.reg_loss(results['cmr_reg_enriched'],    batch['reg_labels'])
            total = total + w * self.reg_loss(results['fundus_reg_enriched'], batch['reg_labels'],
                                              batch.get('reg_mask'))

            # 对齐loss（时间越近权重越高）
            total = total + lambdas['align'] * tw * \
                    self.align_loss(results['align_f'], results['align_c'])

        return total
```

---

## 7. DataLoader设计

```python
import torch
import numpy as np
from torch.utils.data import Sampler


class MixedBatchSampler(Sampler):
    """
    保证每个batch包含固定比例的三种样本
    
    默认比例：CMR:眼底:配对 = 4:4:2
    默认batch_size=80 -> 32 CMR + 32 眼底 + 16 配对
    
    设计原因：
    - 配对样本每个step都要有，否则交互层和LoRA没有梯度
    - 比例固定保证训练动态稳定，不因数据集大小比例影响
    """
    def __init__(self, n_cmr, n_fundus, n_paired,
                 batch_size=80, ratio=(4, 4, 2)):
        self.n_cmr    = n_cmr
        self.n_fundus = n_fundus
        self.n_paired = n_paired
        self.bs       = batch_size
        r_total       = sum(ratio)
        self.bs_cmr   = batch_size * ratio[0] // r_total   # 32
        self.bs_fund  = batch_size * ratio[1] // r_total   # 32
        self.bs_pair  = batch_size * ratio[2] // r_total   # 16

    def __iter__(self):
        cmr_idx  = torch.randperm(self.n_cmr)
        fund_idx = torch.randperm(self.n_fundus)
        pair_idx = torch.randperm(self.n_paired)

        n_batches = min(
            len(cmr_idx)  // self.bs_cmr,
            len(fund_idx) // self.bs_fund,
            len(pair_idx) // self.bs_pair,
        )
        for i in range(n_batches):
            yield {
                'cmr_idx':    cmr_idx [i*self.bs_cmr  : (i+1)*self.bs_cmr].tolist(),
                'fundus_idx': fund_idx[i*self.bs_fund : (i+1)*self.bs_fund].tolist(),
                'paired_idx': pair_idx[i*self.bs_pair : (i+1)*self.bs_pair].tolist(),
            }

    def __len__(self):
        return min(
            self.n_cmr    // self.bs_cmr,
            self.n_fundus // self.bs_fund,
            self.n_paired // self.bs_pair,
        )


def get_time_weight(gap_years, threshold=5.0, low_weight=0.5):
    """
    时间间隔权重
    >5年的配对样本降权到0.5，不完全丢弃（保留数据量）
    """
    return low_weight if gap_years > threshold else 1.0
```

---

## 8. 三阶段训练策略

### Stage 1：单模态多任务全参微调（两个塔独立训练，需重新跑）

**与PDF里实验的区别：**

```
PDF里（旧）：每个任务单独训练一个模型
             心梗一个模型 / 缺血一个模型 / LVEF一个模型...

Stage 1（新）：每个塔一个模型，同时训练所有任务
              CMR塔：分类x4 + 回归x3，共享MedSAM主干，全参微调
              眼底塔：分类x4 + 回归x3，共享RETFound主干，全参微调
              两个塔完全独立，不共享任何参数
```

**为什么要重新跑：**
Stage 2需要一个已经cardiac-specific的多任务编码器作为起点。
单任务的checkpoint只对一个任务负责，embedding空间不够丰富，
直接用于Stage 2会导致交互层学到的是单任务特征而非综合心脏表征。

**CMR塔多任务全参微调：**

```python
# train_stage1_cmr.py
# 数据：96,784人，全参微调MedSAM + 多任务头
# 预计时间：2-3天（4张RTX3090）

import torch
import torch.nn as nn
import torch.nn.functional as F

class SingleTowerCMR(nn.Module):
    """
    Stage 1 CMR单塔多任务模型
    输入：CMR npy (4, 224, 224)
    输出：分类x4 + 回归x3
    """
    def __init__(self, cmr_encoder, dim_c=768):
        super().__init__()
        self.encoder  = cmr_encoder     # MedSAM ViT-B，全参微调
        self.task_head = TaskHead(in_dim=dim_c, n_cls=4, n_reg=3, dropout=0.3)

    def forward(self, x):
        z = self.encoder(x)             # (B, 768)
        cls_out, reg_out = self.task_head(z)
        return cls_out, reg_out


# 优化器：分层学习率
# 主干用小lr（预训练权重不能动太多），任务头用大lr
def build_cmr_optimizer(model):
    return torch.optim.AdamW([
        {'params': model.encoder.parameters(),   'lr': 5e-6},   # 主干
        {'params': model.task_head.parameters(), 'lr': 1e-3},   # 任务头
    ], weight_decay=0.01)


# 学习率调度：warmup 3 epoch + cosine decay
def build_cmr_scheduler(optimizer, n_epochs=50, warmup=3):
    def lr_lambda(epoch):
        if epoch < warmup:
            return epoch / warmup
        progress = (epoch - warmup) / (n_epochs - warmup)
        return 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# 训练配置
CMR_STAGE1_CONFIG = {
    'n_epochs':   50,
    'batch_size': 32,           # 4张3090，每卡8张
    'patience':   15,           # 早停
    'monitor':    'mean_auc',   # 监控验证集AUC均值
    'pos_weights': [14.3, 39.8, 95.2, 14.5],  # 分类任务pos_weight
    'reg_norm': [
        {'mean': 59.6,  'std': 6.4},    # LVEF
        {'mean': 145.9, 'std': 34.1},   # LVEDV
        {'mean': 59.6,  'std': 19.9},   # LVESV
    ],
    'output_dir': '/data/home/experiments/dual_tower/stage1_cmr/',
}

# 启动命令
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --nproc_per_node=4 train_stage1_cmr.py
```

**眼底塔多任务全参微调：**

```python
# train_stage1_fundus.py
# 数据：98,777人（分类任务全量），30,773人（有LVEF/LVEDV标签，回归用mask）
# 预计时间：2-3天（4张RTX3090）

class SingleTowerFundus(nn.Module):
    """
    Stage 1 眼底单塔多任务模型
    输入：眼底PNG图像
    输出：分类x4 + 回归x3（回归标签不全，用mask处理）
    """
    def __init__(self, fundus_encoder, dim_f=1024):
        super().__init__()
        self.encoder   = fundus_encoder    # RETFound ViT-L，全参微调
        self.task_head = TaskHead(in_dim=dim_f, n_cls=4, n_reg=3, dropout=0.3)

    def forward(self, x):
        z = self.encoder(x)                # (B, 1024)
        cls_out, reg_out = self.task_head(z)
        return cls_out, reg_out


# 优化器：RETFound主干lr要更小（参数量更大，ViT-L）
def build_fundus_optimizer(model):
    return torch.optim.AdamW([
        {'params': model.encoder.parameters(),   'lr': 1e-6},   # 主干（更保守）
        {'params': model.task_head.parameters(), 'lr': 1e-3},   # 任务头
    ], weight_decay=0.01)


# 关键：眼底回归任务的mask处理
# 98,777人里只有30,773人有LVEF标签，其余NaN
# DataLoader里对无标签样本，reg_mask对应位置设为False

def compute_fundus_loss(cls_preds, reg_preds, cls_labels,
                        reg_labels, reg_mask, criterion):
    """
    cls_labels: (B, 4)，-1表示缺失
    reg_labels: (B, 3)，NaN表示无标签
    reg_mask:   (B, 3) bool，True=有标签
    """
    loss = criterion.cls_loss(cls_preds, cls_labels)
    loss = loss + criterion.reg_loss(reg_preds, reg_labels, reg_mask)
    return loss


# 训练配置
FUNDUS_STAGE1_CONFIG = {
    'n_epochs':   50,
    'batch_size': 32,
    'patience':   15,
    'monitor':    'mean_auc',
    'output_dir': '/data/home/experiments/dual_tower/stage1_fundus/',
}

# 启动命令
# CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
#     --nproc_per_node=4 train_stage1_fundus.py
```

**两个塔可以同时跑吗？**

```
不能同时跑（4张GPU不够）：
  CMR塔：4张GPU x 2-3天
  眼底塔：4张GPU x 2-3天
  顺序执行，总计4-6天

或者：
  如果有8张GPU，可以同时跑
  CMR塔用GPU 0-3，眼底塔用GPU 4-7
  总计2-3天
```

**Stage 1完成后的验证标准：**

```
CMR塔（多任务）应达到的验证集指标：
  composite_ischemic_hd AUC > 0.65
  prevalent_I21         AUC > 0.65
  LVEDV Pearson         > 0.70   （参考PDF单任务是0.83，多任务略低可接受）
  LVEF  Pearson         > 0.25   （本来就难预测）

眼底塔（多任务）应达到的验证集指标：
  composite_ischemic_hd AUC > 0.60
  prevalent_I21         AUC > 0.60
  LVEF/LVEDV Pearson    > 0.10   （眼底预测心脏结构本来就是间接信号）

如果未达到：检查学习率和pos_weight，不要进入Stage 2
```

**Stage 1产出的checkpoint用于Stage 2：**

```python
# Stage 2加载Stage 1的checkpoint
cmr_stage1_ckpt    = '/data/home/experiments/dual_tower/stage1_cmr/best.pth'
fundus_stage1_ckpt = '/data/home/experiments/dual_tower/stage1_fundus/best.pth'

# 只加载编码器权重（任务头不需要，Stage 2会用新的任务头）
cmr_state    = torch.load(cmr_stage1_ckpt,    map_location='cpu')
fundus_state = torch.load(fundus_stage1_ckpt, map_location='cpu')

model.cmr_encoder.load_state_dict(
    {k.replace('encoder.', ''): v
     for k, v in cmr_state.items() if k.startswith('encoder.')},
    strict=True
)
model.fundus_encoder.load_state_dict(
    {k.replace('encoder.', ''): v
     for k, v in fundus_state.items() if k.startswith('encoder.')},
    strict=True
)
print("Stage 1 checkpoint加载完成，编码器已cardiac-specific化")
```

### Stage 2：交互层冷启动（约2-3天）

```python
# 冻结两个主干，只训练交互层和任务头
# 目的：让CrossModalGating先学会基本对话，不要一上来就三路梯度打架

for param in model.fundus_encoder.parameters():
    param.requires_grad = False
for param in model.cmr_encoder.parameters():
    param.requires_grad = False

trainable_params = (
    list(model.cross_modal.parameters()) +
    list(model.align_proj_f.parameters()) +
    list(model.align_proj_c.parameters()) +
    list(model.head_fundus.parameters()) +
    list(model.head_cmr.parameters())
)

optimizer = torch.optim.AdamW(
    trainable_params, lr=1e-4, weight_decay=0.01
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=20, eta_min=1e-6
)

# 配置
lambdas   = {'align': 0.1}
patience  = 10
n_epochs  = 20
# 监控：验证集cosine similarity均值
```

### Stage 3：全量联合训练（约5-7天）

```python
# 全部解冻，分层学习率
# 主干用极小学习率保护预训练权重，交互层和任务头用较大学习率

optimizer = torch.optim.AdamW([
    # 主干：极小学习率
    {'params': model.fundus_encoder.parameters(), 'lr': 1e-6},
    {'params': model.cmr_encoder.parameters(),    'lr': 1e-6},
    # 交互模块：中等学习率
    {'params': model.cross_modal.parameters(),    'lr': 1e-4},
    # 对齐投影
    {'params': model.align_proj_f.parameters(),   'lr': 1e-4},
    {'params': model.align_proj_c.parameters(),   'lr': 1e-4},
    # 任务头：较大学习率
    {'params': model.head_fundus.parameters(),    'lr': 1e-3},
    {'params': model.head_cmr.parameters(),       'lr': 1e-3},
], weight_decay=0.01)

# warmup 5 epoch + cosine decay to eta_min
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, eta_min=1e-7
)

# 对齐loss权重随训练进度调整
def get_align_lambda(epoch):
    return 0.1 if epoch < 10 else 0.05  # 后期任务loss主导

# batch构成（每batch=80）
# 32 CMR单模态 + 32 眼底单模态 + 16 配对样本

# 梯度裁剪（防止爆炸）
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 早停：patience=15
# 监控：0.6 x 验证集AUC均值 + 0.4 x Pearson均值
def monitor_metric(val_auc, val_pearson):
    return 0.6 * val_auc + 0.4 * val_pearson
```

---

## 9. 消融实验设计

### 实验配置

| 实验 | 配置 | 数据来源 | 新增组件 | 回答的问题 |
|------|------|---------|---------|----------|
| Exp A | CMR单塔多任务全参微调 | 96,784人 | — | CMR单独能做多好（上限） |
| Exp B | 眼底单塔多任务全参微调 | 98,777人 | — | 眼底单独能做多好（上限） |
| Exp C | 双塔+cosine软对齐，无Cross-Attention | 全部 | align loss | 配对数据本身有没有价值 |
| Exp D | 双塔+Cross-Attention，无LoRA | 全部 | CrossModalGating（去掉LoRA直接加） | 交互层有没有价值 |
| Exp E | 完整模型（含LoRA回流） | 全部 | 全部组件 | LoRA有没有额外价值 |

> Exp A和B已在PDF报告中完成，直接复用结果。

### 期望结果链

```
Exp E >= Exp D > Exp C > max(Exp A, Exp B)

每步只多一个组件，故事链条清晰，直接写进论文消融Table
```

### 关键观察点（论文叙事）

```
观察1（最重要）：
  眼底侧：Exp E的眼底AUC > Exp B（眼底单塔）
  -> 说明CMR的cardiac知识成功迁移到了眼底模型

观察2：
  CMR侧：Exp E的CMR AUC >= Exp A（CMR单塔）
  -> 说明眼底信息对CMR也有帮助（不是单向的）

观察3：
  Exp E的LVEDV Pearson与Exp A接近甚至超过
  -> 说明双塔交互没有破坏CMR的几何结构预测能力
```

---

## 10. 评估指标

```python
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.stats import pearsonr, spearmanr
import torch.nn.functional as F


def eval_classification(y_true, y_pred_logits):
    """
    分类任务评估
    AUPRC在阳性率极低时（如1%）比AUROC更能反映真实性能
    """
    y_prob = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
    valid  = y_true >= 0  # 排除缺失标签
    if valid.sum() == 0 or len(np.unique(y_true[valid])) < 2:
        return {'AUROC': 0.5, 'AUPRC': 0.0}
    return {
        'AUROC': roc_auc_score(y_true[valid], y_prob[valid]),
        'AUPRC': average_precision_score(y_true[valid], y_prob[valid]),
    }


def eval_regression(y_true, y_pred_norm, norm_mean, norm_std):
    """
    回归任务评估
    注意：y_pred_norm是归一化后的预测值，需要反归一化
    """
    y_pred = y_pred_norm * norm_std + norm_mean   # 反归一化
    valid  = ~np.isnan(y_true)
    if valid.sum() < 10:
        return {'Pearson': 0.0, 'Spearman': 0.0, 'MAE': 999.0, 'R2': -999.0}
    return {
        'Pearson':  pearsonr(y_true[valid],  y_pred[valid])[0],
        'Spearman': spearmanr(y_true[valid], y_pred[valid])[0],
        'MAE':      mean_absolute_error(y_true[valid], y_pred[valid]),
        'R2':       r2_score(y_true[valid], y_pred[valid]),
    }


def eval_alignment(z_f, z_c):
    """配对embedding对齐质量"""
    return {
        'cosine_similarity': F.cosine_similarity(z_f, z_c, dim=-1).mean().item()
    }


# 综合监控指标（用于早停）
def monitor_metric(cls_results, reg_results):
    auc_list = [v['AUROC'] for v in cls_results.values()]
    prs_list = [v['Pearson'] for v in reg_results.values() if 'Pearson' in v]
    mean_auc = np.mean(auc_list) if auc_list else 0.5
    mean_prs = np.mean(prs_list) if prs_list else 0.0
    return 0.6 * mean_auc + 0.4 * mean_prs
```

---

## 11. 今日预实验计划

### 时间规划

```
上午（现在）：运行数据采样 + Pipeline验证（第一层）
下午：         4张GPU并行跑快速消融（第二层）
傍晚：         看结果决策
晚上：         启动全量训练，挂机跑过夜
```

### 第一层：Pipeline验证（目标1小时内完成）

```
数据规模：CMR=100人，眼底=100人，配对=50人
训练设置：20 epoch，lr=1e-3，不早停

通过标准（全部满足才进入第二层）：
  [1] forward pass不报错
  [2] loss数值合理（不是nan/inf）
  [3] 三种mode（cmr_only/fundus_only/paired）都能运行
  [4] 训练集loss持续下降（不震荡）
  [5] 训练集AUC能到0.8以上（说明能overfit，模型容量够）
  [6] 梯度没有nan/inf

失败处理：说明有代码bug，立即修复，不进入第二层
```

### 第二层：快速消融（目标下午完成）

```
数据规模：
  CMR:    3,000人（训练2,400 + 验证600）
  眼底:   3,000人（训练2,400 + 验证600）
  配对:   1,500人（训练1,200 + 验证300）

训练：每个实验15 epoch，patience=5早停
GPU分配（4张GPU同时跑）：
  GPU 0: Exp A - CMR单塔
  GPU 1: Exp B - 眼底单塔
  GPU 2: Exp C - 双塔无交互
  GPU 3: Exp D - 完整模型

判断标准（只看这一件事）：
  Exp D的验证AUC均值 > max(Exp A AUC, Exp B AUC)
    -> 是：傍晚启动全量训练 ✅
    -> 否：花1小时看loss曲线，调整lambda/lr后决定
```

### 傍晚决策表（填入实验结果）

| 实验 | 验证AUC均值 | LVEDV Pearson | 结论 |
|------|-----------|--------------|------|
| Exp A | __ | __ | CMR基线 |
| Exp B | __ | N/A | 眼底基线 |
| Exp C | __ | __ | 软对齐效果 |
| Exp D | __ | __ | 完整模型 |

**决策：Exp D > max(A,B) -> 晚上启动全量训练**

---

## 12. 预实验代码

### 12.1 数据采样脚本

```python
# sample_mini_dataset.py
"""
从完整数据集中分层采样，生成小规模预实验数据
用法：
  python sample_mini_dataset.py --n_cmr 100 --n_fundus 100 --n_paired 50
  python sample_mini_dataset.py --n_cmr 3000 --n_fundus 3000 --n_paired 1500
"""
import argparse
import json
import pandas as pd
import numpy as np
from pathlib import Path

CLS_COLS = [
    'composite_ischemic_hd', 'prevalent_I21',
    'composite_cardiomyopathy_hf', 'composite_af_arrhythmia',
]
REG_COLS = [
    'LV ejection fraction',
    'LV end diastolic volume',
    'LV end systolic volume',
]


def stratified_sample(df, n, cls_col=None, seed=42):
    """分层采样，尽量保证阳性样本比例不变"""
    df = df.copy()
    if cls_col and cls_col in df.columns:
        pos = df[df[cls_col] == 1]
        neg = df[df[cls_col] == 0]
        n_pos = max(1, int(n * len(pos) / len(df)))
        n_neg = n - n_pos
        sampled = pd.concat([
            pos.sample(min(n_pos, len(pos)), random_state=seed),
            neg.sample(min(n_neg, len(neg)), random_state=seed),
        ])
    else:
        sampled = df.sample(min(n, len(df)), random_state=seed)
    return sampled.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cmr_csv',    default='stage1_cmr.csv')
    parser.add_argument('--fundus_csv', default='stage1_fundus.csv')
    parser.add_argument('--paired_csv', default='stage2_fundus_dual.csv')
    parser.add_argument('--n_cmr',      type=int, default=100)
    parser.add_argument('--n_fundus',   type=int, default=100)
    parser.add_argument('--n_paired',   type=int, default=50)
    parser.add_argument('--outdir',     default='mini_data')
    parser.add_argument('--seed',       type=int, default=42)
    args = parser.parse_args()

    Path(args.outdir).mkdir(exist_ok=True)

    # 采样CMR
    cmr = pd.read_csv(args.cmr_csv, low_memory=False)
    cmr_s = stratified_sample(cmr, args.n_cmr,
                               cls_col='composite_ischemic_hd', seed=args.seed)
    cmr_s.to_csv(f'{args.outdir}/mini_cmr.csv', index=False)
    print(f'CMR: {len(cmr_s)}行 | 阳性率: {cmr_s.get("composite_ischemic_hd", pd.Series([0])).mean():.2%}')

    # 采样眼底
    fundus = pd.read_csv(args.fundus_csv, low_memory=False)
    fundus_s = stratified_sample(fundus, args.n_fundus,
                                  cls_col='composite_ischemic_hd', seed=args.seed)
    fundus_s.to_csv(f'{args.outdir}/mini_fundus.csv', index=False)
    print(f'眼底: {len(fundus_s)}行')

    # 采样配对
    paired = pd.read_csv(args.paired_csv, low_memory=False)
    paired_s = stratified_sample(paired, args.n_paired,
                                  cls_col='composite_ischemic_hd', seed=args.seed)
    paired_s.to_csv(f'{args.outdir}/mini_paired.csv', index=False)
    print(f'配对: {len(paired_s)}行')

    # 计算并保存归一化参数（只从CMR训练集算！）
    norm_params = {}
    for col, key in zip(REG_COLS, ['lvef', 'lvedv', 'lvesv']):
        if col in cmr_s.columns:
            vals = pd.to_numeric(cmr_s[col], errors='coerce').dropna()
            norm_params[key] = {'mean': float(vals.mean()), 'std': float(vals.std())}
            print(f'{key}: mean={norm_params[key]["mean"]:.2f}, std={norm_params[key]["std"]:.2f}')

    with open(f'{args.outdir}/norm_params.json', 'w') as f:
        json.dump(norm_params, f, indent=2)
    print(f'\n归一化参数已保存到 {args.outdir}/norm_params.json')


if __name__ == '__main__':
    main()
```

### 12.2 Pipeline验证脚本

```python
# sanity_check.py
"""
第一层验证：确认整个pipeline能正常运行（使用随机embedding替代真实图像）
用法：python sanity_check.py --n_epochs 20
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def make_fake_batch(n, mode, device, dim_f=1024, dim_c=768):
    """生成假数据，用随机向量替代真实图像embedding"""
    batch = {'mode': mode}
    cls_labels = torch.randint(0, 2, (n, 4)).float()
    reg_labels = torch.randn(n, 3)
    reg_mask   = torch.ones(n, 3).bool()

    if mode == 'fundus_only':
        batch.update({
            'fundus':     torch.randn(n, dim_f),
            'cls_labels': cls_labels,
            'reg_labels': reg_labels,
            'reg_mask':   reg_mask,
        })
    elif mode == 'cmr_only':
        batch.update({
            'cmr':        torch.randn(n, dim_c),
            'cls_labels': cls_labels,
            'reg_labels': reg_labels,
        })
    elif mode == 'paired':
        batch.update({
            'fundus':      torch.randn(n, dim_f),
            'cmr':         torch.randn(n, dim_c),
            'cls_labels':  cls_labels,
            'reg_labels':  reg_labels,
            'reg_mask':    reg_mask,
            'time_weight': 1.0,
        })

    return {k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()}


def run_sanity_check(n_epochs=20):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    print('=' * 60)
    print('PIPELINE SANITY CHECK 开始')
    print('=' * 60)

    # 构建简化模型（直接用embedding输入，不需要真实编码器）
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.cross_modal  = CrossModalGating()
            self.head_fundus  = TaskHead(1024)
            self.head_cmr     = TaskHead(768)
            self.align_proj_f = nn.Linear(1024, 256)
            self.align_proj_c = nn.Linear(768, 256)

        def forward(self, batch):
            results = {}
            mode = batch['mode']
            if mode == 'fundus_only':
                z_f = batch['fundus']
                cls, reg = self.head_fundus(z_f)
                results.update({'fundus_cls': cls, 'fundus_reg': reg})
            elif mode == 'cmr_only':
                z_c = batch['cmr']
                cls, reg = self.head_cmr(z_c)
                results.update({'cmr_cls': cls, 'cmr_reg': reg})
            elif mode == 'paired':
                z_f, z_c = batch['fundus'], batch['cmr']
                cls_f0, reg_f0 = self.head_fundus(z_f)
                cls_c0, reg_c0 = self.head_cmr(z_c)
                z_f_en, z_c_en = self.cross_modal(z_f, z_c)
                cls_f1, reg_f1 = self.head_fundus(z_f_en)
                cls_c1, reg_c1 = self.head_cmr(z_c_en)
                results.update({
                    'fundus_cls_base': cls_f0, 'fundus_reg_base': reg_f0,
                    'cmr_cls_base':    cls_c0, 'cmr_reg_base':    reg_c0,
                    'fundus_cls_enriched': cls_f1, 'fundus_reg_enriched': reg_f1,
                    'cmr_cls_enriched':    cls_c1, 'cmr_reg_enriched':    reg_c1,
                    'align_f': F.normalize(self.align_proj_f(z_f), dim=-1),
                    'align_c': F.normalize(self.align_proj_c(z_c), dim=-1),
                })
            return results

    model     = SimpleModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = MultiTaskLoss()
    lambdas   = {'align': 0.1}

    all_passed = True
    loss_history = []

    for epoch in range(n_epochs):
        epoch_losses = []
        for mode in ['fundus_only', 'cmr_only', 'paired']:
            n = 16 if mode == 'paired' else 32
            batch = make_fake_batch(n, mode, device)

            optimizer.zero_grad()
            results = model(batch)
            loss    = criterion.compute(results, batch, lambdas)

            # 检查1：loss合法性
            if torch.isnan(loss) or torch.isinf(loss):
                print(f'[FAIL] Epoch {epoch} mode={mode}: loss={loss.item()}（nan/inf）')
                all_passed = False
                break

            loss.backward()

            # 检查2：梯度合法性
            max_grad = max(
                p.grad.abs().max().item()
                for p in model.parameters() if p.grad is not None
            )
            if np.isnan(max_grad) or np.isinf(max_grad):
                print(f'[FAIL] Epoch {epoch} mode={mode}: 梯度爆炸 max_grad={max_grad}')
                all_passed = False
                break

            optimizer.step()
            epoch_losses.append(loss.item())

        if not all_passed:
            break

        avg_loss = np.mean(epoch_losses)
        loss_history.append(avg_loss)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f'Epoch {epoch:3d} | Loss: {avg_loss:.4f} | MaxGrad: {max_grad:.6f}')

    # 检查3：loss是否在下降
    if all_passed and len(loss_history) >= 5:
        early_loss = np.mean(loss_history[:3])
        late_loss  = np.mean(loss_history[-3:])
        if late_loss >= early_loss:
            print(f'[WARN] loss没有下降: early={early_loss:.4f}, late={late_loss:.4f}')

    print('\n' + '=' * 60)
    if all_passed:
        print('[PASS] Pipeline验证通过！')
        print('[PASS] 没有发现nan/inf/梯度爆炸')
        print('[NEXT] 可以进入第二层：快速消融实验')
    else:
        print('[FAIL] Pipeline验证失败，请修复后再继续')
    print('=' * 60)
    return all_passed


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_epochs', type=int, default=20)
    args = parser.parse_args()
    # 注意：运行前需要import上面定义的模型类
    run_sanity_check(args.n_epochs)
```

### 12.3 消融训练脚本

```python
# train_ablation.py
"""
快速消融实验脚本（4张GPU并行）

启动命令：
  GPU 0: CUDA_VISIBLE_DEVICES=0 python train_ablation.py --exp A
  GPU 1: CUDA_VISIBLE_DEVICES=1 python train_ablation.py --exp B
  GPU 2: CUDA_VISIBLE_DEVICES=2 python train_ablation.py --exp C
  GPU 3: CUDA_VISIBLE_DEVICES=3 python train_ablation.py --exp D
"""
import argparse
import json
import os
import numpy as np
import torch
from sklearn.metrics import roc_auc_score


EXP_CONFIGS = {
    'A': {'mode': 'cmr_only',    'use_cross_attn': False, 'use_lora': False, 'align_lambda': 0.0},
    'B': {'mode': 'fundus_only', 'use_cross_attn': False, 'use_lora': False, 'align_lambda': 0.0},
    'C': {'mode': 'both',        'use_cross_attn': False, 'use_lora': False, 'align_lambda': 0.1},
    'D': {'mode': 'both',        'use_cross_attn': True,  'use_lora': False, 'align_lambda': 0.1},
    'E': {'mode': 'both',        'use_cross_attn': True,  'use_lora': True,  'align_lambda': 0.1},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp',       choices=['A','B','C','D','E'], required=True)
    parser.add_argument('--data_dir',  default='mini_data')
    parser.add_argument('--n_epochs',  type=int, default=15)
    parser.add_argument('--outdir',    default='ablation_results')
    parser.add_argument('--patience',  type=int, default=5)
    parser.add_argument('--batch_size',type=int, default=32)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    cfg    = EXP_CONFIGS[args.exp]
    print(f'Exp {args.exp} | Config: {cfg} | Device: {device}')

    # ── 这里接入真实模型和DataLoader ──
    # model = build_model_for_exp(args.exp, cfg, ...)
    # model = model.to(device)
    # train_loader, val_loader = build_dataloaders(args.data_dir, cfg)
    # criterion = MultiTaskLoss()
    # optimizer = build_optimizer(model, args.exp)

    lambdas    = {'align': cfg['align_lambda']}
    best_auc   = 0.0
    best_epoch = 0
    history    = []

    for epoch in range(args.n_epochs):
        # train_loss = train_one_epoch(model, train_loader, optimizer,
        #                              criterion, device, lambdas)
        # val_metrics = evaluate(model, val_loader, device, args.exp)

        train_loss  = 0.0                  # 占位，替换为实际值
        val_mean_auc = 0.0                 # 占位，替换为实际值
        val_lvedv_pearson = 0.0            # 占位，替换为实际值

        history.append({
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_auc':    val_mean_auc,
            'val_pearson_lvedv': val_lvedv_pearson,
        })

        print(f'Epoch {epoch:3d} | Loss: {train_loss:.4f} | '
              f'Val AUC: {val_mean_auc:.4f} | LVEDV r: {val_lvedv_pearson:.4f}')

        if val_mean_auc > best_auc:
            best_auc   = val_mean_auc
            best_epoch = epoch
            # torch.save(model.state_dict(),
            #            f'{args.outdir}/exp_{args.exp}_best.pth')

        if epoch - best_epoch >= args.patience:
            print(f'Early stop: patience={args.patience} reached')
            break

    result = {
        'exp': args.exp, 'config': cfg,
        'best_auc': best_auc, 'best_epoch': best_epoch,
        'history': history,
    }
    with open(f'{args.outdir}/exp_{args.exp}_result.json', 'w') as f:
        json.dump(result, f, indent=2)

    print(f'\nExp {args.exp} Done | Best Val AUC: {best_auc:.4f} @ epoch {best_epoch}')
    return result


if __name__ == '__main__':
    main()
```

### 12.4 结果汇总与决策脚本

```python
# summarize_ablation.py
"""
汇总消融实验结果，输出决策建议
用法：python summarize_ablation.py --result_dir ablation_results
"""
import json
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result_dir', default='ablation_results')
    args = parser.parse_args()

    results = {}
    for exp in ['A', 'B', 'C', 'D', 'E']:
        path = f'{args.result_dir}/exp_{exp}_result.json'
        if os.path.exists(path):
            with open(path) as f:
                results[exp] = json.load(f)

    desc = {
        'A': 'CMR单塔（上限）',
        'B': '眼底单塔（上限）',
        'C': '双塔+软对齐，无Cross-Attention',
        'D': '双塔+Cross-Attention，无LoRA',
        'E': '完整模型（含LoRA回流）',
    }

    print('\n' + '=' * 65)
    print('消融实验结果汇总')
    print('=' * 65)
    print(f'{"实验":<6} {"最佳AUC":<12} {"最佳Epoch":<12} {"说明"}')
    print('-' * 65)

    for exp, res in results.items():
        print(f'Exp {exp:<3} {res["best_auc"]:<12.4f} {res["best_epoch"]:<12} {desc.get(exp,"")}')

    print('=' * 65)

    if 'A' in results and 'D' in results:
        auc_a    = results['A']['best_auc']
        auc_b    = results.get('B', {}).get('best_auc', 0.0)
        auc_c    = results.get('C', {}).get('best_auc', 0.0)
        auc_d    = results['D']['best_auc']
        baseline = max(auc_a, auc_b)

        print(f'\n基线 max(A,B): {baseline:.4f}')
        print(f'软对齐   (C):  {auc_c:.4f}  (vs baseline: {auc_c - baseline:+.4f})')
        print(f'完整模型 (D):  {auc_d:.4f}  (vs baseline: {auc_d - baseline:+.4f})')

        print('\n' + '=' * 65)
        if auc_d > baseline:
            print('决策：完整模型优于单塔基线')
            print('建议：今晚启动全量训练！')
            print('\n执行：bash launch_full_training.sh')
        elif auc_d > auc_c:
            print('决策：Cross-Attention有效，但总体未超基线')
            print('建议：检查loss曲线，可能需要调整learning rate或lambda')
        else:
            print('决策：双塔框架未超基线，需要检查设计')
            print('建议：排查数据pipeline和loss计算是否正确')
        print('=' * 65)


if __name__ == '__main__':
    main()
```

### 12.5 启动全量训练脚本

```bash
#!/bin/bash
# launch_full_training.sh
# 预实验通过后，傍晚执行此脚本启动全量训练

OUTPUT_ROOT=/data/home/experiments/dual_tower
mkdir -p $OUTPUT_ROOT/stage2_warmup
mkdir -p $OUTPUT_ROOT/stage3_full
mkdir -p $OUTPUT_ROOT/ablation_full

echo "========================================"
echo "启动全量训练 $(date)"
echo "输出路径: $OUTPUT_ROOT"
echo "========================================"

# Stage 2：交互层冷启动（冻结主干）
echo "[Stage 2] 交互层冷启动..."
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29500 \
    train_stage2.py \
    --output_dir $OUTPUT_ROOT/stage2_warmup \
    --freeze_backbone \
    --n_epochs 20 \
    --lr 1e-4 \
    --batch_size 64 \
    --patience 10 \
    --align_lambda 0.1 \
    2>&1 | tee $OUTPUT_ROOT/stage2_warmup.log

echo "[Stage 2] 完成，启动 Stage 3..."

# Stage 3：全量联合训练
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=29501 \
    train_stage3.py \
    --output_dir $OUTPUT_ROOT/stage3_full \
    --stage2_ckpt $OUTPUT_ROOT/stage2_warmup/best.pth \
    --n_epochs 50 \
    --lr_backbone 1e-6 \
    --lr_interaction 1e-4 \
    --lr_head 1e-3 \
    --batch_size 80 \
    --patience 15 \
    2>&1 | tee $OUTPUT_ROOT/stage3_full.log

echo "========================================"
echo "全量训练完成 $(date)"
echo "========================================"
```

---

## 13. 存储与路径规范

```
警告：根盘只剩74GB（96%已用），所有输出必须写到大盘！

大盘路径：/data/home/experiments/dual_tower/

目录结构：
dual_tower/
├── data_prepared/                   # 预处理后的训练数据（Agent生成）
│   ├── stage1_cmr_sampled.csv       # CMR下采样全集（阳性全量+阴性1万）
│   ├── stage1_cmr_sampled_train.csv # CMR训练集（80%）
│   ├── stage1_cmr_sampled_val.csv   # CMR验证集（10%）
│   ├── stage1_cmr_sampled_test.csv  # CMR测试集（10%，封存不动）
│   ├── stage1_fundus_sampled.csv    # 眼底下采样全集
│   ├── stage1_fundus_sampled_train.csv
│   ├── stage1_fundus_sampled_val.csv
│   ├── stage1_fundus_sampled_test.csv
│   ├── pos_weights_cmr.json         # CMR塔pos_weight（下采样后计算）
│   ├── pos_weights_fundus.json      # 眼底塔pos_weight（下采样后计算）
│   └── norm_params.json             # 回归归一化参数（训练集计算）
│
│   ⚠️ Agent任务：以上所有文件需在训练开始前生成并校验
│   校验要求：
│     1. train/val/test三个集合EID无交叉
│     2. 测试集封存，训练过程中不得读取
│     3. pos_weights和norm_params只从train集计算
│     4. 打印各集合的人数和阳性率确认分布合理
│
├── mini_data/                       # 预实验小数据
│   ├── mini_cmr.csv                 # 100人CMR子集
│   ├── mini_fundus.csv              # 100人眼底子集
│   ├── mini_paired.csv              # 50人配对子集
│   └── norm_params_mini.json        # mini数据的归一化参数
│
├── ablation_results/                # 快速消融结果
│   ├── exp_A_result.json
│   ├── exp_B_result.json
│   ├── exp_C_result.json
│   └── exp_D_result.json
│
├── stage1_cmr/                      # Stage1 CMR塔多任务微调
│   ├── best.pth                     # 最佳验证checkpoint
│   ├── last.pth
│   └── train.log
│
├── stage1_fundus/                   # Stage1 眼底塔多任务微调
│   ├── best.pth
│   ├── last.pth
│   └── train.log
│
├── stage2_warmup/                   # Stage2 交互层冷启动
│   ├── best.pth
│   ├── last.pth
│   └── stage2.log
│
├── stage3_full/                     # Stage3 全量联合训练（最终模型）
│   ├── best.pth
│   ├── last.pth
│   └── stage3.log
│
├── ablation_full/                   # 全量消融实验
│   ├── exp_C/
│   └── exp_D/
│
└── evaluation/                      # 最终评估结果
    ├── classification_results.json
    ├── regression_results.json
    ├── ablation_table.csv
    └── plots/
        ├── lvedv_scatter.png
        ├── lvef_scatter.png
        └── roc_curves.png
```

---

## 附录：关键超参数总览

| 超参数 | 数值 | 说明 |
|--------|------|------|
| LoRA秩 r | 16 | 表达能力与参数量平衡点 |
| LoRA alpha | 32 | scale = alpha/r = 2 |
| Cross-Attention hidden | 512 | 两塔公共空间维度 |
| Cross-Attention heads | 8 | 注意力头数 |
| Batch构成 | 32+32+16=80 | CMR+眼底+配对 |
| Stage2 lr | 1e-4 | 冻结主干时交互层学习率 |
| Stage3 backbone lr | 1e-6 | 全解冻时主干学习率 |
| Stage3 interaction lr | 1e-4 | 交互层学习率 |
| Stage3 head lr | 1e-3 | 任务头学习率 |
| align lambda（前10epoch） | 0.1 | 对齐loss权重 |
| align lambda（后续） | 0.05 | 后期降低，任务loss主导 |
| 时间间隔降权阈值 | 5年 | >5年配对样本loss权重×0.5 |
| Stage2 早停 patience | 10 epoch | |
| Stage3 早停 patience | 15 epoch | |
| Stage3 监控指标 | 0.6×AUC+0.4×Pearson | |
| 梯度裁剪 | max_norm=1.0 | 防止梯度爆炸 |
| Weight decay | 0.01 | AdamW正则化 |
| Dropout | 0.3 | 任务头dropout |
