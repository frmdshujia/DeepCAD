# 数据集模块说明

## RetinaCardiacDataset

`RetinaCardiacDataset` 是 DeepCAD 项目的主要数据集类，用于加载视网膜眼底照片和心脏MRI数据的配对样本。

### CSV 数据格式

数据集需要一个CSV文件，包含以下列：

| 列名 | 类型 | 说明 |
|------|------|------|
| `subject_id` | str | 受试者唯一标识符 |
| `retinal_path` | str | 视网膜图像文件路径（相对或绝对路径） |
| `mri_paths` | str | MRI数据路径，可以是：<br>- 单个文件路径<br>- 逗号分隔的多个路径<br>- JSON格式的路径列表字符串 |
| `label` | int | CAD标签 (0: 无CAD, 1: 有CAD) |

### CSV 示例

```csv
subject_id,retinal_path,mri_paths,label
subj_001,retinal_images/subj_001.jpg,mri_data/subj_001_cine.nii.gz,1
subj_002,retinal_images/subj_002.png,"mri_data/subj_002_cine.nii.gz,mri_data/subj_002_t1.nii.gz",0
subj_003,retinal_images/subj_003.jpg,"[\"mri_data/subj_003_slice1.nii.gz\", \"mri_data/subj_003_slice2.nii.gz\"]",1
```

### 使用示例

```python
from datasets import RetinaCardiacDataset, create_dataloaders

# 方式1: 直接使用数据集
dataset = RetinaCardiacDataset(
    data_csv='data/splits/train.csv',
    retinal_img_size=224,
    mri_img_size=224,
    train=True,
    retinal_augmentation='medium',
    mri_augmentation='medium',
    max_mri_slices=10,
    live_loading=True,
    retinal_base_path='data/retinal',
    mri_base_path='data/mri'
)

# 获取一个样本
sample = dataset[0]
print(sample['x_R'].shape)  # (3, 224, 224) - 视网膜图像
print(sample['x_C'].shape)  # (num_slices, 1, 224, 224) - MRI切片
print(sample['y'])          # tensor(1) - CAD标签
print(sample['subject_id']) # 'subj_001'

# 方式2: 使用便捷函数创建数据加载器
train_loader, val_loader, test_loader = create_dataloaders(
    train_csv='data/splits/train.csv',
    val_csv='data/splits/val.csv',
    test_csv='data/splits/test.csv',
    batch_size=32,
    num_workers=4,
    retinal_img_size=224,
    mri_img_size=224,
    max_mri_slices=10
)

# 在训练循环中使用
for batch in train_loader:
    x_R = batch['x_R']  # (batch_size, 3, 224, 224)
    x_C = batch['x_C']  # (batch_size, num_slices, 1, 224, 224)
    y = batch['y']      # (batch_size,)
    subject_ids = batch['subject_id']  # List[str]
    # ... 训练代码 ...
```

## 数据变换

### 视网膜图像变换

- **训练模式**: 应用数据增强（随机裁剪、翻转、颜色抖动等）
- **验证/测试模式**: 只进行标准化变换

增强强度选项：
- `light`: 轻度增强（随机翻转、轻微颜色抖动）
- `medium`: 中度增强（旋转、颜色抖动、高斯模糊）
- `strong`: 强度增强（弹性变换、更强的颜色抖动）

### MRI图像变换

- **训练模式**: 应用空间增强（随机裁剪、翻转、旋转）
- **验证/测试模式**: 只进行标准化变换

注意：MRI增强通常比视网膜图像更保守，以保持医学图像的解剖结构。

## 数据加载工具

### load_retinal_image

加载视网膜眼底照片（支持JPEG、PNG等格式）

### load_mri_slices

加载心脏MRI切片，支持：
- NIfTI格式文件（.nii, .nii.gz）
- 图像格式文件（PNG、JPEG等）
- 目录（自动加载目录中的所有图像文件）

### load_mri_from_nifti

从NIfTI文件加载代表性切片：
- `slice_type="mid"`: 提取中间切片
- `slice_type="representative"`: 均匀采样多个代表性切片
- `slice_type="all"`: 加载所有切片

## 注意事项

1. **内存使用**: 如果数据集很大，建议使用 `live_loading=True`（默认），这样数据会实时从磁盘加载，而不是预加载到内存。

2. **MRI切片数量**: `max_mri_slices` 参数限制每个样本的最大切片数量。如果MRI数据包含更多切片，会进行均匀采样。

3. **路径处理**: 
   - 如果CSV中的路径是绝对路径，直接使用
   - 如果是相对路径，可以使用 `retinal_base_path` 和 `mri_base_path` 指定基础路径

4. **批次处理**: 由于不同样本的MRI切片数量可能不同，数据加载器会自动处理填充或截断，确保批次大小一致。

