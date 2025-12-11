# 6 张切片的处理策略分析

## 数据组成

每个样本包含 6 张切片：
- **Cine 序列**（3 张）：
  - ES (end-systolic)
  - Mid beat
  - ED (end-diastolic)
- **T1 Map 序列**（3 张）：
  - 可能也是对应 ES, mid, ED 的 T1 Map 切片

## 方案对比

### 方案 A：每张切片单独处理（推荐）✅

**处理方式**：
- 6 张切片，每张形状 `(1, H, W)` 或 `(H, W)`
- 堆叠成 `(6, 1, H, W)` 或 `(6, H, W)`
- 在 MedSAM encoder 中：
  - 每张切片复制为 3 通道：`(6, 1, H, W) → (6, 3, H, W)`
  - 每张切片独立通过 MedSAM encoder
  - 池化所有 6 张切片的特征

**优点**：
- ✅ 当前代码已经支持（`MRICardioEncoder` 支持多切片）
- ✅ 每张切片独立处理，保留各自特征
- ✅ T1 Map 和 Cine 信息不同，独立处理更合理
- ✅ 不需要修改 encoder 结构
- ✅ 可以学习不同序列之间的关系（通过池化层）

**缺点**：
- ⚠️ 没有显式利用 MMCL 的 3 通道堆叠方式
- ⚠️ 每张切片需要单独通过 encoder（计算量稍大）

**代码示例**：
```python
# 数据加载
cine_slices = load_ukb_cardiac_slices(subject_folder, ['sa_ES.nii.gz', 'sa.nii.gz', 'sa_ED.nii.gz'])  # (3, H, W)
t1_slices = load_t1_slices(subject_folder, ['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz'])  # (3, H, W)

# 堆叠所有切片
all_slices = torch.cat([cine_slices, t1_slices], dim=0)  # (6, H, W)

# 添加通道维度
all_slices = all_slices.unsqueeze(1)  # (6, 1, H, W)

# 通过 encoder（会自动处理）
embedding = mri_encoder(all_slices.unsqueeze(0))  # (1, 6, 1, H, W) -> (1, embed_dim)
```

---

### 方案 B：每张切片 repeat 成 3 通道，然后堆叠

**处理方式**：
- 6 张切片，每张 repeat 成 `(3, H, W)`
- 堆叠成 `(6, 3, H, W)`
- 在 encoder 中，每张 `(3, H, W)` 直接通过 MedSAM

**优点**：
- ✅ 每张切片已经是 3 通道，不需要在 encoder 中复制
- ✅ 形式上更接近 RGB 图像

**缺点**：
- ❌ 重复信息（3 个通道完全相同）
- ❌ 浪费计算资源
- ❌ 没有利用 MMCL 的语义（MMCL 的 3 通道是不同的切片）

**结论**：不推荐，因为只是简单重复，没有增加信息。

---

### 方案 C：分成两组，每组 3 张堆叠成 (3, H, W)

**处理方式**：
- 组1：3 张 Cine 切片 → `(3, H, W)`（参考 MMCL）
- 组2：3 张 T1 Map 切片 → `(3, H, W)`
- 两组分别通过 encoder，然后融合

**优点**：
- ✅ 保留 MMCL 的 3 通道堆叠方式（Cine 组）
- ✅ 两组可以分别学习不同序列的特征

**缺点**：
- ❌ 需要修改 encoder 结构（支持多组输入）
- ❌ 需要额外的融合机制
- ❌ 实现复杂度更高

**结论**：如果特别想保留 MMCL 的方式，可以考虑，但需要较大改动。

---

## 推荐方案：方案 A（每张切片单独处理）

### 理由

1. **当前代码已支持**：`MRICardioEncoder` 已经设计为处理多切片输入
2. **信息保留**：每张切片独立处理，保留各自特征
3. **灵活性**：可以处理不同数量的切片
4. **实现简单**：不需要修改 encoder 结构

### 实现建议

#### 1. 修改数据加载函数

```python
def load_ukb_cardiac_with_t1(
    subject_folder: str,
    include_t1: bool = True
) -> torch.Tensor:
    """
    加载 UKB 心脏 MRI 数据（Cine + T1 Map）
    
    Args:
        subject_folder: 受试者文件夹路径
        include_t1: 是否包含 T1 Map 切片
    
    Returns:
        torch.Tensor: 形状为 (6, H, W) 或 (3, H, W) 的 Tensor
    """
    # 加载 Cine 切片
    cine_slices = load_ukb_cardiac_slices(subject_folder)  # (3, H, W)
    
    if include_t1:
        # 加载 T1 Map 切片
        t1_slices = load_t1_slices(subject_folder)  # (3, H, W)
        # 堆叠
        all_slices = torch.cat([cine_slices, t1_slices], dim=0)  # (6, H, W)
    else:
        all_slices = cine_slices  # (3, H, W)
    
    return all_slices
```

#### 2. 数据流程

```
6 张切片 (6, H, W)
  ↓
添加通道维度 (6, 1, H, W)
  ↓
Batch 维度 (B, 6, 1, H, W)
  ↓
MedSAM Encoder:
  - 每张切片复制为 3 通道: (B*6, 3, H, W)
  - 上采样到 1024x1024
  - 通过 MedSAM: (B*6, 256, 64, 64)
  - 投影: (B*6, 768)
  ↓
Reshape: (B, 6, 768)
  ↓
池化: (B, 768)
```

#### 3. 关于 "repeat 成 3 通道"

**不建议每张切片单独 repeat 成 3 通道**，原因：
- 重复信息，浪费计算
- MedSAM encoder 内部会自动处理单通道到 3 通道的转换
- 当前实现已经是最优的

**建议**：
- 保持每张切片为单通道 `(1, H, W)`
- 让 MedSAM encoder 在内部复制为 3 通道
- 这样既保留了信息，又符合 MedSAM 的要求

---

## 总结

**推荐方案**：方案 A（每张切片单独处理）✅ **已实现**

**关键点**：
1. ✅ 6 张切片，每张 `(1, H, W)`
2. ✅ 堆叠成 `(6, 1, H, W)`
3. ✅ MedSAM encoder 内部处理通道转换和上采样
4. ✅ 池化所有切片的特征

**不需要**：
- ❌ 每张切片单独 repeat 成 3 通道（浪费且无意义）
- ❌ 修改 encoder 结构（当前设计已经很好）

**优势**：
- 简单、高效、灵活
- 保留所有信息
- 符合 MedSAM 的要求

---

## 使用示例

### 方式1：直接使用数据加载函数

```python
from datasets import load_ukb_cardiac_with_t1

# 加载 6 张切片（3 张 Cine + 3 张 T1 Map）
slices = load_ukb_cardiac_with_t1(
    subject_folder='path/to/subject',
    include_t1=True,
    t1_cycle_positions=['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz']
)  # 返回 (6, H, W)

# 添加 batch 和通道维度
slices = slices.unsqueeze(0).unsqueeze(2)  # (1, 6, 1, H, W)

# 通过 encoder（会自动处理）
embedding = mri_encoder(slices)  # (1, 768)
```

### 方式2：使用 RetinaCardiacDataset

```python
from datasets import RetinaCardiacDataset

# 创建数据集（UKB格式，包含T1 Map）
dataset = RetinaCardiacDataset(
    data_csv='data/splits/train.csv',
    retinal_img_size=224,
    mri_img_size=224,
    train=True,
    use_ukb_format=True,  # 使用UKB格式
    include_t1=True,      # 包含T1 Map（6张切片）
    t1_cycle_positions=['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz'],
    mri_base_path='data/ukb_mri',
    live_loading=True
)

# 获取一个样本
sample = dataset[0]
print(sample['x_R'].shape)  # (3, 224, 224) - 视网膜图像
print(sample['x_C'].shape)  # (6, 1, 224, 224) - 6张MRI切片
print(sample['y'])           # tensor(1) - CAD标签
```

### CSV 格式示例

对于UKB格式，CSV中的 `mri_paths` 列应该是受试者文件夹路径：

```csv
subject_id,retinal_path,mri_paths,label
subj_001,retinal_images/subj_001.jpg,668815/subj_001,1
subj_002,retinal_images/subj_002.png,668815/subj_002,0
```

其中 `mri_paths` 指向的文件夹应包含：
- `sa_ES.nii.gz`, `sa.nii.gz`, `sa_ED.nii.gz`（Cine序列）
- `t1_ES.nii.gz`, `t1_mid.nii.gz`, `t1_ED.nii.gz`（T1 Map序列，如果 `include_t1=True`）

