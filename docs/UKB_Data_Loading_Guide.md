# UKB 心脏 MRI 数据加载指南

## 概述

本指南说明如何使用 DeepCAD 加载和处理 UKB（UK Biobank）心脏 MRI 数据，参考 MMCL 的实现方式。

## 数据格式

### UKB 心脏 MRI 数据结构

每个受试者的数据文件夹应包含以下文件：

**Cine 序列**（必需）：
- `sa_ES.nii.gz`：end-systolic（收缩末期）
- `sa.nii.gz`：完整心脏周期（用于提取 mid beat）
- `sa_ED.nii.gz`：end-diastolic（舒张末期）

**T1 Map 序列**（可选）：
- `t1_ES.nii.gz`：T1 Map end-systolic
- `t1_mid.nii.gz`：T1 Map mid beat
- `t1_ED.nii.gz`：T1 Map end-diastolic

### 数据预处理（参考 MMCL）

MMCL 的预处理步骤：
1. 从每个 NIfTI 文件中提取中间 z 轴切片：`im[:,:,im.shape[2]//2]`
2. 对于 `sa.nii.gz`（4D数据），提取中间心跳的中间 z 轴切片
3. Padding 成正方形
4. 堆叠成 `(3, H, W)` 或 `(6, H, W)`（如果包含T1 Map）

## 使用方式

### 方式1：使用数据加载函数

```python
from datasets import load_ukb_cardiac_with_t1, load_ukb_cardiac_slices

# 加载 6 张切片（Cine + T1 Map）
slices_6 = load_ukb_cardiac_with_t1(
    subject_folder='path/to/subject',
    include_t1=True,
    t1_cycle_positions=['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz']
)  # 返回 (6, H, W)

# 加载 3 张切片（仅 Cine）
slices_3 = load_ukb_cardiac_slices(
    subject_folder='path/to/subject'
)  # 返回 (3, H, W)
```

### 方式2：使用 RetinaCardiacDataset

```python
from datasets import RetinaCardiacDataset

# 创建数据集（包含T1 Map，6张切片）
dataset = RetinaCardiacDataset(
    data_csv='data/splits/train.csv',
    retinal_img_size=224,
    mri_img_size=224,
    train=True,
    use_ukb_format=True,      # 启用UKB格式
    include_t1=True,          # 包含T1 Map
    t1_cycle_positions=['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz'],
    mri_base_path='data/ukb_mri',
    live_loading=True
)

# 获取样本
sample = dataset[0]
x_R = sample['x_R']  # (3, 224, 224) - 视网膜图像
x_C = sample['x_C']  # (6, 1, 224, 224) - 6张MRI切片
y = sample['y']      # tensor(1) - CAD标签
```

### CSV 格式

对于UKB格式，CSV文件格式：

```csv
subject_id,retinal_path,mri_paths,label
subj_001,retinal_images/subj_001.jpg,668815/subj_001,1
subj_002,retinal_images/subj_002.png,668815/subj_002,0
```

**说明**：
- `mri_paths` 列应该是受试者文件夹路径（相对于 `mri_base_path`）
- 文件夹应包含上述的 NIfTI 文件

## 数据流程（方案A）

```
UKB 数据文件夹
  ↓
load_ukb_cardiac_with_t1()
  ↓
提取关键切片：
  - Cine: ES, mid, ED (3张)
  - T1 Map: ES, mid, ED (3张)
  ↓
堆叠: (6, H, W)
  ↓
添加通道维度: (6, 1, H, W)
  ↓
Batch维度: (B, 6, 1, H, W)
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

## 配置选项

### RetinaCardiacDataset 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `use_ukb_format` | bool | False | 是否使用UKB格式（参考MMCL） |
| `include_t1` | bool | False | 是否包含T1 Map切片（仅当 `use_ukb_format=True` 时有效） |
| `t1_cycle_positions` | List[str] | None | T1 Map的周期位置列表 |

### 切片数量

- `use_ukb_format=True, include_t1=False`: 3 张切片（仅Cine）
- `use_ukb_format=True, include_t1=True`: 6 张切片（Cine + T1 Map）
- `use_ukb_format=False`: 由 `max_mri_slices` 决定

## 注意事项

1. **文件命名**：确保NIfTI文件命名符合预期（`sa_ES.nii.gz` 等）
2. **数据质量**：MMCL要求z轴切片数量 > 7，否则会被跳过
3. **T1 Map可用性**：如果 `include_t1=True` 但T1 Map文件不存在，会抛出错误（除非使用 `fallback_to_cine=True`）
4. **值范围**：所有数据都会归一化到 [0, 1]，符合 MedSAM 要求
5. **空间尺寸**：Cine 和 T1 Map 切片会自动对齐到相同尺寸

## 示例：完整训练流程

```python
from datasets import RetinaCardiacDataset, create_dataloaders

# 创建数据加载器
train_loader, val_loader, test_loader = create_dataloaders(
    train_csv='data/splits/train.csv',
    val_csv='data/splits/val.csv',
    test_csv='data/splits/test.csv',
    batch_size=32,
    num_workers=4,
    retinal_img_size=224,
    mri_img_size=224,
    # UKB格式配置
    use_ukb_format=True,
    include_t1=True,
    t1_cycle_positions=['t1_ES.nii.gz', 't1_mid.nii.gz', 't1_ED.nii.gz'],
    mri_base_path='data/ukb_mri',
    live_loading=True
)

# 训练循环
for batch in train_loader:
    x_R = batch['x_R']  # (B, 3, 224, 224)
    x_C = batch['x_C']  # (B, 6, 1, 224, 224) - 6张切片
    y = batch['y']      # (B,)
    # ... 训练代码 ...
```

