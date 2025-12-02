# 调试指南

本文档提供 DeepCAD 项目的调试方法和最佳实践。

## 快速开始

### 1. 验证项目设置

在开始训练之前，先验证项目设置：

```bash
python scripts/validate_setup.py
```

这会检查：
- Python 包依赖
- 项目结构
- 外部依赖（RETFound, MedSAM）
- PyTorch 配置

### 2. 调试模型配置

在训练之前，先调试模型配置：

```bash
python scripts/debug_model.py \
    --train_csv data/splits/train.csv \
    --retinal_pretrained RETFound_mae \
    --mri_pretrained checkpoints/pretrained/medsam/medsam_vit_b.pth
```

这会检查：
- 模型架构正确性
- 前向传播
- 张量形状匹配
- 损失计算
- 梯度流

## 调试工具

### utils/debug.py

提供以下调试函数：

#### check_tensor_shapes()
检查模型各层的张量形状

```python
from utils.debug import check_tensor_shapes

shapes = check_tensor_shapes(model, x_R, x_C)
print(shapes)
```

#### check_normalization()
检查投影嵌入是否已正确归一化

```python
from utils.debug import check_normalization

norm_check = check_normalization(z_R, z_C)
print(norm_check)
```

#### check_loss_computation()
检查损失计算的正确性

```python
from utils.debug import check_loss_computation

loss_check = check_loss_computation(z_R, z_C, labels)
print(loss_check)
```

#### diagnose_training_issue()
全面诊断训练问题

```python
from utils.debug import diagnose_training_issue, print_diagnosis

diagnosis = diagnose_training_issue(model, train_loader, criterion)
print_diagnosis(diagnosis)
```

#### validate_model_architecture()
验证模型架构的正确性

```python
from utils.debug import validate_model_architecture

arch_check = validate_model_architecture(model)
print(arch_check)
```

## 常见问题诊断流程

### 问题: 训练损失不下降

1. **检查数据加载**
   ```python
   # 验证数据加载器
   for batch in train_loader:
       print(batch['x_R'].shape)
       print(batch['x_C'].shape)
       print(batch['y'])
       break
   ```

2. **检查模型输出**
   ```python
   model.eval()
   with torch.no_grad():
       outputs = model(x_R, x_C)
       print(f"z_R norm: {outputs['z_R'].norm(dim=1)}")
       print(f"z_C norm: {outputs['z_C'].norm(dim=1)}")
   ```

3. **检查损失值**
   ```python
   L, L_C, L_R = criterion(z_R, z_C, labels)
   print(f"Loss: {L.item()}, L_C: {L_C.item()}, L_R: {L_R.item()}")
   ```

4. **检查梯度**
   ```python
   L.backward()
   for name, param in model.named_parameters():
       if param.grad is not None:
           print(f"{name}: grad_norm = {param.grad.norm().item()}")
   ```

### 问题: 内存不足

1. **减小批次大小**
   ```bash
   --batch_size 8  # 从 32 减小到 8
   ```

2. **减少 MRI 切片数量**
   ```bash
   --max_mri_slices 5  # 从 10 减小到 5
   ```

3. **使用梯度检查点**（如果支持）
   ```python
   from torch.utils.checkpoint import checkpoint
   ```

### 问题: 张量形状不匹配

1. **检查编码器输出维度**
   ```python
   retinal_dim = model.retinal_encoder.get_embed_dim()  # 应该是 1024
   mri_dim = model.mri_encoder.get_embed_dim()  # 应该是 768
   ```

2. **检查投影头配置**
   ```python
   proj_R_input = model.projection_R.mlp[0].in_features
   proj_C_input = model.projection_C.mlp[0].in_features
   assert proj_R_input == retinal_dim
   assert proj_C_input == mri_dim
   ```

## 日志分析

### 训练日志

检查训练日志文件：
```bash
cat logs/stage1/training_*.log
```

关注：
- 损失值趋势
- 学习率变化
- 梯度范数

### TensorBoard

如果使用 TensorBoard：
```bash
tensorboard --logdir logs/stage1/tensorboard
```

查看：
- 训练/验证损失曲线
- 学习率曲线
- 梯度分布

## 代码审查清单

在提交代码或开始训练前，检查：

- [ ] 模型架构正确（编码器 + 投影头）
- [ ] 损失函数实现正确（数学定义）
- [ ] 数据加载正确（形状、标签）
- [ ] 归一化正确（L2 归一化）
- [ ] 梯度流正常（无零梯度、无 NaN）
- [ ] 检查点保存/加载正常
- [ ] 日志记录正常

## 性能分析

### 使用 PyTorch Profiler

```python
from torch.profiler import profile, record_function, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    outputs = model(x_R, x_C)
    loss = criterion(...)
    loss.backward()

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

### 检查瓶颈

1. **数据加载瓶颈**: 增加 `num_workers`
2. **模型计算瓶颈**: 使用混合精度训练
3. **内存瓶颈**: 减小批次大小

## 最佳实践

1. **从小开始**: 先用小数据集和少量 epoch 测试
2. **逐步增加**: 确认基本功能正常后再增加复杂度
3. **保存检查点**: 定期保存，避免丢失训练进度
4. **监控指标**: 关注损失、梯度、学习率等关键指标
5. **版本控制**: 记录每次实验的配置和结果

## 获取帮助

如果遇到问题：

1. 运行 `validate_setup.py` 检查环境
2. 运行 `debug_model.py` 诊断问题
3. 查看 `TROUBLESHOOTING.md` 查找解决方案
4. 检查日志文件获取详细错误信息

