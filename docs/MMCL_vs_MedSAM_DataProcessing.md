# MMCL vs MedSAM 数据处理方式对比

## MMCL-Tabular-Imaging-main 的数据处理方式

### 1. 数据预处理阶段（`create_cardiac_image_dataset.ipynb`）

**输入**：UKB 心脏 MRI NIfTI 文件
- `sa_ES.nii.gz`：end-systolic（收缩末期）
- `sa.nii.gz`：完整心脏周期
- `sa_ED.nii.gz`：end-diastolic（舒张末期）

**处理步骤**：
1. **提取关键切片**：
   - 从每个 NIfTI 文件中提取中间 z 轴切片：`im[:,:,im.shape[2]//2]`
   - 对于 `sa.nii.gz`（4D数据），提取中间心跳的中间 z 轴切片
   - 检查：z 轴切片数量必须 > 7（质量检查）

2. **Padding 成正方形**：
   ```python
   # 如果宽 > 高，上下 padding
   # 如果高 > 宽，左右 padding
   ```

3. **堆叠成 3 通道**：
   ```python
   to_stack = [es_slice, mid_beat_slice, ed_slice]
   ims_stacked = torch.stack(to_stack)  # (3, H, W)
   ```

4. **最终形状**：
   - 形状：`(3, 210, 210)` 或 `(3, 208, 208)`（会 padding 到 210x210）
   - 值范围：原始值（未归一化到 [0, 1]）

5. **保存格式**：
   ```python
   all_subjects = {subject_id: tensor}  # 字典格式
   torch.save(all_subjects, 'preprocessed_cardiac_dict.pt')
   ```

### 2. 训练时的数据处理（`ContrastiveImageDataset`）

**加载**：
```python
self.data = torch.load(data_path)  # 字典或列表
im = self.data[index]  # (3, H, W) Tensor
```

**数据增强**（`grab_image_augmentations`）：
- RandomHorizontalFlip
- RandomRotation(45)
- ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
- RandomResizedCrop(size=img_size, scale=(0.2, 1.0))
- Lambda(lambda x: x.float())  # 转换为 float

**归一化**：
- 如果使用 `read_image()`：`im = im / 255`（归一化到 [0, 1]）
- 如果直接使用 Tensor：值范围保持原始（可能需要归一化）

**输出格式**：
- 形状：`(3, img_size, img_size)` 或 `(C, H, W)`
- 值范围：`[0, 1]`（如果使用了 `/255`）

---

## MedSAM 的数据处理方式

### 1. 数据预处理阶段（`pre_CT_MR.py`）

**输入**：CT/MR 医学图像（单通道）

**处理步骤**：
1. **单通道复制为 3 通道**：
   ```python
   img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)  # (H, W) -> (H, W, 3)
   ```

2. **Resize 到目标尺寸**：
   ```python
   resize_img = transform.resize(img_3c, (image_size, image_size), ...)
   ```

3. **归一化到 [0, 1]**：
   ```python
   resize_img_01 = (resize_img - resize_img.min()) / (resize_img.max() - resize_img.min())
   ```

4. **保存格式**：
   - `.npy` 文件
   - 形状：`(H, W, 3)` 或 `(image_size, image_size, 3)`
   - 值范围：`[0, 1]`

### 2. 训练时的数据处理（`train_one_gpu.py`）

**加载**：
```python
img = np.load(img_path)  # (H, W, 3), [0, 1]
img_1024 = resize(img, (1024, 1024))  # MedSAM 标准输入尺寸
assert img_1024.max() <= 1.0 and img_1024.min() >= 0.0
```

**MedSAM 模型输入要求**：
- **形状**：`(B, 3, 1024, 1024)`
- **值范围**：`[0, 1]`
- **通道**：3 通道 RGB（单通道图像需要复制）

**内部处理**：
- MedSAM 的 `image_encoder` 期望 1024x1024 输入
- 输出：`(B, 256, 64, 64)` 的特征图

---

## 关键区别总结

| 特性 | MMCL | MedSAM |
|------|------|--------|
| **通道数** | 3 通道（3 张不同的切片） | 3 通道（单通道复制 3 次） |
| **切片选择** | 3 个时间点（ES, mid, ED）的中间 z 轴切片 | 单张切片或所有切片 |
| **输入尺寸** | 210x210 或自定义（训练时 resize） | 1024x1024（固定） |
| **值范围** | 原始值或 [0, 1]（取决于处理） | [0, 1]（必须） |
| **数据格式** | `.pt` 文件（预处理的 Tensor） | `.npy` 文件或直接加载 |
| **预处理** | 提取关键切片 + padding | Resize + 归一化 |
| **数据增强** | 空间变换 + ColorJitter | 空间变换（较少） |

---

## 对 DeepCAD 项目的建议

### 当前实现的问题

1. **MMCL 方式**：每个样本是 `(3, H, W)`，3 个通道代表 3 个时间点的切片
2. **MedSAM 要求**：期望 `(B, 3, 1024, 1024)`，3 通道是 RGB（单通道复制）

### 解决方案

**选项 1：使用 MMCL 的 3 通道方式（推荐）**
- 保持 MMCL 的预处理方式：3 张切片作为 3 通道
- MedSAM encoder 可以直接处理（它不关心通道的语义，只是需要 3 通道输入）
- 优势：保留更多时间信息

**选项 2：单通道方式**
- 只使用一张切片（如中间切片）
- 在 MedSAM encoder 中复制为 3 通道
- 优势：更简单，但丢失时间信息

**选项 3：多切片方式（当前实现）**
- 使用多张切片 `(num_slices, H, W)`
- 每个切片单独处理，然后池化
- 优势：保留空间信息，但需要池化层

### 推荐方案

如果使用 UKB 数据，建议：
1. **预处理**：参考 MMCL，提取 3 个时间点的切片，堆叠成 `(3, H, W)`
2. **训练**：直接使用 `(3, H, W)` 作为输入，MedSAM encoder 可以处理
3. **数据增强**：参考 MMCL 的增强策略
4. **值范围**：确保归一化到 `[0, 1]`，符合 MedSAM 要求

