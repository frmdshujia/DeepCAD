# 问题排查指南

本文档提供 DeepCAD 项目常见问题的诊断和解决方案。

## 目录

1. [训练问题](#训练问题)
2. [数据加载问题](#数据加载问题)
3. [模型架构问题](#模型架构问题)
4. [损失计算问题](#损失计算问题)
5. [内存问题](#内存问题)
6. [性能问题](#性能问题)

## 训练问题

### 问题1: 损失值为 NaN 或 Inf

**症状**: 训练过程中损失值变为 NaN 或 Inf

**可能原因**:
1. 学习率过大
2. 梯度爆炸
3. 输入数据包含 NaN 或 Inf
4. 损失计算中的数值不稳定

**解决方案**:
```python
# 1. 降低学习率
--lr 1e-5  # 从 1e-4 降低到 1e-5

# 2. 添加梯度裁剪
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
# 在训练循环中添加
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# 3. 检查输入数据
# 使用 debug_model.py 脚本检查数据

# 4. 增加损失计算中的 eps
criterion = CrossModalContrastiveLoss(tau=0.1, eps=1e-6)
```

### 问题2: 损失不下降

**症状**: 训练多个 epoch 后损失值没有明显下降

**可能原因**:
1. 学习率过小
2. 编码器被冻结但投影头初始化不当
3. 数据标签错误
4. 批次大小过小

**解决方案**:
```python
# 1. 增加学习率
--lr 1e-3

# 2. 检查编码器是否被意外冻结
model.unfreeze_encoders()  # 如果应该训练编码器

# 3. 验证数据标签
# 检查 CSV 文件中的标签分布

# 4. 增加批次大小
--batch_size 64
```

### 问题3: 梯度为零

**症状**: 模型参数梯度为零，无法更新

**可能原因**:
1. 编码器被冻结
2. 损失计算错误
3. 优化器配置错误

**解决方案**:
```python
# 1. 检查参数是否可训练
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: requires_grad=True")

# 2. 使用调试脚本检查梯度
python scripts/debug_model.py --train_csv data/splits/train.csv
```

## 数据加载问题

### 问题4: 数据加载失败

**症状**: `FileNotFoundError` 或数据形状不匹配

**可能原因**:
1. 文件路径错误
2. CSV 格式不正确
3. 图像文件损坏

**解决方案**:
```python
# 1. 检查 CSV 格式
# 确保包含列: subject_id, retinal_path, mri_paths, label

# 2. 验证文件路径
import os
for idx, row in df.iterrows():
    retinal_path = row['retinal_path']
    if not os.path.exists(retinal_path):
        print(f"Missing: {retinal_path}")

# 3. 使用相对路径和 base_path
--retinal_base_path data/retinal
--mri_base_path data/mri
```

### 问题5: MRI 切片数量不一致

**症状**: 不同样本的 MRI 切片数量不同导致批次处理失败

**解决方案**:
```python
# 在数据集中设置最大切片数量
dataset = RetinaCardiacDataset(
    max_mri_slices=10,  # 限制最大切片数
    ...
)

# 或使用自定义 collate_fn 处理变长序列
```

## 模型架构问题

### 问题6: 张量形状不匹配

**症状**: `RuntimeError: Sizes of tensors must match`

**可能原因**:
1. 编码器输出维度与投影头输入维度不匹配
2. 批次中样本的 MRI 切片数量不一致

**诊断**:
```python
# 使用调试脚本检查
python scripts/debug_model.py

# 或手动检查
from utils.debug import check_tensor_shapes
shapes = check_tensor_shapes(model, x_R, x_C)
print(shapes)
```

**解决方案**:
```python
# 1. 检查模型配置
retinal_embed_dim = model.retinal_encoder.get_embed_dim()  # 应该是 1024
mri_embed_dim = model.mri_encoder.get_embed_dim()  # 应该是 768

# 2. 确保投影头输入维度匹配
proj_R = ProjectionHead(input_dim=1024, ...)
proj_C = ProjectionHead(input_dim=768, ...)
```

### 问题7: 预训练权重加载失败

**症状**: 无法加载 RETFound 或 MedSAM 预训练权重

**解决方案**:
```python
# 1. 检查文件路径
# RETFound: 可以是 HuggingFace Hub ID 或本地路径
--retinal_pretrained RETFound_mae  # 或
--retinal_pretrained checkpoints/pretrained/retfound/RETFound_mae.pth

# MedSAM: 必须是本地路径
--mri_pretrained checkpoints/pretrained/medsam/medsam_vit_b.pth

# 2. 如果从 HuggingFace 下载失败，手动下载
# 访问: https://huggingface.co/YukunZhou/RETFound_mae
```

## 损失计算问题

### 问题8: 正样本对不足

**症状**: 警告 "samples_without_positives"

**可能原因**:
1. 批次大小过小
2. 标签分布不均匀

**解决方案**:
```python
# 1. 增加批次大小
--batch_size 64

# 2. 使用加权采样确保每个批次都有正负样本
# 在 DataLoader 中使用 WeightedRandomSampler
```

### 问题9: 损失值异常（负数或过大）

**症状**: 损失值为负数或非常大

**可能原因**:
1. 温度参数设置不当
2. 归一化未正确应用

**解决方案**:
```python
# 1. 检查温度参数
# 默认 0.1，如果损失过大可以尝试 0.2 或 0.5
--temperature 0.1

# 2. 验证归一化
from utils.debug import check_normalization
norm_check = check_normalization(z_R, z_C)
print(norm_check)  # 应该显示 normalized=True
```

## 内存问题

### 问题10: CUDA 内存不足

**症状**: `RuntimeError: CUDA out of memory`

**解决方案**:
```python
# 1. 减小批次大小
--batch_size 16  # 或更小

# 2. 减少 MRI 切片数量
--max_mri_slices 5

# 3. 使用梯度累积
# 在训练循环中累积多个小批次的梯度

# 4. 使用混合精度训练
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()
with autocast():
    outputs = model(x_R, x_C)
    loss = criterion(...)
scaler.scale(loss).backward()
scaler.step(optimizer)
```

## 性能问题

### 问题11: 训练速度慢

**可能原因**:
1. 数据加载是瓶颈
2. 模型太大
3. 未使用 GPU

**解决方案**:
```python
# 1. 增加数据加载器工作进程
--num_workers 8

# 2. 使用预加载（如果内存足够）
dataset = RetinaCardiacDataset(..., live_loading=False)

# 3. 冻结编码器（如果只需要训练投影头）
--freeze_encoders

# 4. 使用更小的模型（如果可能）
# 或使用量化
```

## 调试工具

### 使用调试脚本

```bash
# 完整诊断
python scripts/debug_model.py \
    --train_csv data/splits/train.csv \
    --retinal_pretrained RETFound_mae \
    --mri_pretrained checkpoints/pretrained/medsam/medsam_vit_b.pth

# 仅检查模型架构
python scripts/debug_model.py \
    --retinal_pretrained RETFound_mae
```

### 手动调试

```python
from utils.debug import (
    diagnose_training_issue,
    print_diagnosis,
    validate_model_architecture
)

# 诊断训练问题
diagnosis = diagnose_training_issue(model, train_loader, criterion)
print_diagnosis(diagnosis)

# 验证架构
arch_check = validate_model_architecture(model)
print(arch_check)
```

## 常见错误消息

### "Missing required columns in CSV"
- **原因**: CSV 文件缺少必需的列
- **解决**: 确保 CSV 包含 `subject_id`, `retinal_path`, `mri_paths`, `label`

### "No valid MRI slices found"
- **原因**: MRI 路径无效或文件格式不支持
- **解决**: 检查 MRI 文件路径和格式（支持 NIfTI 和图像格式）

### "RETFound models not available"
- **原因**: RETFound 代码路径不正确
- **解决**: 确保 `RETFound-main/` 在项目父目录中

### "MedSAM not available"
- **原因**: MedSAM 代码路径不正确
- **解决**: 确保 `MedSAM-main/` 在项目父目录中

## 获取帮助

如果以上解决方案无法解决问题：

1. 运行完整诊断脚本并保存输出
2. 检查日志文件 (`logs/stage1/`)
3. 提供以下信息：
   - 错误消息和堆栈跟踪
   - 模型配置
   - 数据格式
   - 系统信息（PyTorch 版本、CUDA 版本等）

