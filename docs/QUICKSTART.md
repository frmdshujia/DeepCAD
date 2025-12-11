# DeepCAD 快速开始指南

本指南将帮助您从零开始运行 DeepCAD 项目。

## 目录

1. [环境安装](#1-环境安装)
2. [数据准备](#2-数据准备)
3. [小样本快速测试](#3-小样本快速测试)
4. [正式训练](#4-正式训练)

---

## 1. 环境安装

### 1.1 创建虚拟环境（推荐）

```bash
# 使用 conda
conda create -n deepcad python=3.10
conda activate deepcad

# 或使用 venv（确保系统已安装 Python 3.10）
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

### 1.2 安装基础依赖

```bash
cd DeepCAD
pip install -r requirements.txt
```

### 1.3 安装额外依赖（如果需要）

```bash
# 用于 PCA 可视化（可选）
pip install scikit-learn

# 用于 TensorBoard（可选）
pip install tensorboard
```

### 1.4 验证安装

```bash
python scripts/validate_setup.py
```

应该看到所有必需的包都显示 ✓。

### 1.5 检查外部依赖

确保以下目录存在（在项目父目录中）：
- `../RETFound-main/` - RETFound 代码
- `../MedSAM-main/` - MedSAM 代码

如果不存在，需要克隆这些仓库：
```bash
cd ..
git clone <RETFound-repo-url> RETFound-main
git clone <MedSAM-repo-url> MedSAM-main
```

---

## 2. 数据准备

### 2.1 数据格式要求

数据需要组织为 CSV 格式，包含以下列：

| 列名 | 类型 | 说明 | 示例 |
|------|------|------|------|
| `subject_id` | str | 受试者唯一ID | "subj_001" |
| `retinal_path` | str | 视网膜图像路径 | "retinal/subj_001.jpg" |
| `mri_paths` | str | MRI路径（可以是单个或多个） | "mri/subj_001.nii.gz" 或 "mri/subj_001_cine.nii.gz,mri/subj_001_t1.nii.gz" |
| `label` | int | CAD标签 (0=无CAD, 1=有CAD) | 1 |

### 2.2 创建数据目录结构

```bash
mkdir -p data/retinal
mkdir -p data/mri
mkdir -p data/splits
```

### 2.3 准备 CSV 文件

创建 `data/splits/train.csv`:

```csv
subject_id,retinal_path,mri_paths,label
subj_001,retinal/subj_001.jpg,mri/subj_001_cine.nii.gz,1
subj_002,retinal/subj_002.png,mri/subj_002_cine.nii.gz,0
subj_003,retinal/subj_003.jpg,"mri/subj_003_cine.nii.gz,mri/subj_003_t1.nii.gz",1
```

**路径说明**:
- 如果使用相对路径，需要在训练时指定 `--retinal_base_path` 和 `--mri_base_path`
- 如果使用绝对路径，可以直接在 CSV 中写完整路径

### 2.4 数据划分

建议将数据划分为：
- `train.csv`: 训练集（70-80%）
- `val.csv`: 验证集（10-15%）
- `test.csv`: 测试集（10-15%）

### 2.5 数据格式检查

```python
import pandas as pd

# 检查 CSV 格式
df = pd.read_csv('data/splits/train.csv')
print(df.head())
print(f"样本数量: {len(df)}")
print(f"标签分布: {df['label'].value_counts()}")
```

---

## 3. 小样本快速测试

### 3.1 创建测试数据集

首先创建一个小的测试数据集（例如 10-20 个样本）：

```bash
# 从完整数据集中选择前 20 个样本作为测试
head -n 21 data/splits/train.csv > data/splits/train_mini.csv  # 包含表头
```

### 3.2 下载预训练权重（可选，但推荐）

#### RETFound 权重

```bash
# 方式1: 从 HuggingFace Hub 自动下载（推荐）
# 在训练脚本中使用 --retinal_pretrained RETFound_mae
# 脚本会自动从 HuggingFace 下载

# 方式2: 手动下载
# 访问 https://huggingface.co/YukunZhou/RETFound_mae
# 下载 RETFound_mae.pth 到 checkpoints/pretrained/retfound/
mkdir -p checkpoints/pretrained/retfound
# 然后下载文件到该目录
```

#### MedSAM 权重

```bash
# 从 MedSAM 官方仓库下载
mkdir -p checkpoints/pretrained/medsam
# 下载 medsam_vit_b.pth 到该目录
# 参考: https://github.com/bowang-lab/MedSAM
```

### 3.3 验证模型配置

```bash
python scripts/debug_model.py \
    --train_csv data/splits/train_mini.csv \
    --retinal_pretrained RETFound_mae \
    --mri_pretrained checkpoints/pretrained/medsam/medsam_vit_b.pth \
    --batch_size 2
```

如果看到所有检查都通过 ✓，说明配置正确。

### 3.4 运行小样本训练测试

```bash
python scripts/train_stage1.py \
    --train_csv data/splits/train_mini.csv \
    --val_csv data/splits/train_mini.csv \
    --retinal_pretrained RETFound_mae \
    --mri_pretrained checkpoints/pretrained/medsam/medsam_vit_b.pth \
    --batch_size 2 \
    --num_epochs 2 \
    --lr 1e-4 \
    --retinal_base_path data \
    --mri_base_path data \
    --save_dir checkpoints/test \
    --log_dir logs/test
```

**预期结果**:
- 训练应该能正常开始
- 损失值应该为正数且逐渐变化
- 没有错误或警告

### 3.5 验证训练输出

检查：
```bash
# 检查检查点是否保存
ls checkpoints/test/

# 检查日志
cat logs/test/training_*.log
```

### 3.6 测试可视化（可选）

```bash
python scripts/visualize_gradcam.py \
    --checkpoint checkpoints/test/best_model.pth \
    --data_csv data/splits/train_mini.csv \
    --num_samples 2 \
    --retinal_base_path data \
    --mri_base_path data \
    --save_dir visualizations/test
```

---

## 4. 正式训练

### 4.1 超参数设置建议

根据您的数据和硬件配置调整：

#### 基础配置（适合大多数情况）

```bash
python scripts/train_stage1.py \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --retinal_pretrained RETFound_mae \
    --mri_pretrained checkpoints/pretrained/medsam/medsam_vit_b.pth \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4 \
    --weight_decay 1e-5 \
    --temperature 0.1 \
    --optimizer adam \
    --scheduler cosine \
    --latent_dim 128 \
    --mri_pooling_type attention \
    --retinal_base_path data \
    --mri_base_path data \
    --save_dir checkpoints/stage1 \
    --log_dir logs/stage1 \
    --save_interval 5 \
    --num_workers 4
```

#### 如果 GPU 内存不足

```bash
# 减小批次大小
--batch_size 16

# 减少 MRI 切片数量
--max_mri_slices 5

# 减少数据加载器工作进程
--num_workers 2
```

#### 如果数据量很大

```bash
# 增加批次大小（如果内存允许）
--batch_size 64

# 使用更多工作进程
--num_workers 8
```

#### 如果只想训练投影头（快速实验）

```bash
--freeze_encoders \
--lr 1e-3  # 可以设置更大的学习率
```

### 4.2 训练监控

#### 使用 TensorBoard

```bash
# 在另一个终端运行
tensorboard --logdir logs/stage1/tensorboard

# 然后在浏览器打开 http://localhost:6006
```

#### 查看日志文件

```bash
# 实时查看训练日志
tail -f logs/stage1/training_*.log
```

### 4.3 训练检查点

训练过程中会保存：
- `checkpoints/stage1/checkpoint_epoch_N.pth`: 定期检查点
- `checkpoints/stage1/best_model.pth`: 最佳模型（验证损失最低）
- `checkpoints/stage1/latest_checkpoint.pth`: 最新检查点

### 4.4 恢复训练

如果训练中断，可以从检查点恢复：

```bash
python scripts/train_stage1.py \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --resume_from checkpoints/stage1/latest_checkpoint.pth \
    --num_epochs 100 \
    # ... 其他参数保持不变
```

### 4.5 超参数调优建议

#### 学习率
- 默认: `1e-4`
- 如果损失不下降: 尝试 `1e-3` 或 `5e-4`
- 如果损失震荡: 尝试 `5e-5` 或 `1e-5`

#### 温度参数
- 默认: `0.1`
- 如果损失过大: 尝试 `0.2` 或 `0.5`
- 如果损失过小: 尝试 `0.05`

#### 潜在空间维度
- 默认: `128`
- 可以尝试: `64`, `256`, `512`

#### 批次大小
- 根据 GPU 内存调整
- 建议: 16, 32, 64

### 4.6 训练后验证

训练完成后，验证模型：

```bash
# 使用调试脚本验证
python scripts/debug_model.py \
    --checkpoint checkpoints/stage1/best_model.pth \
    --train_csv data/splits/val.csv
```

### 4.7 生成可视化

```bash
python scripts/visualize_gradcam.py \
    --checkpoint checkpoints/stage1/best_model.pth \
    --data_csv data/splits/val.csv \
    --num_samples 20 \
    --retinal_base_path data \
    --mri_base_path data \
    --save_dir visualizations/final
```

---

## 常见问题快速解决

### 问题1: "RETFound models not available"
**解决**: 确保 `RETFound-main/` 在项目父目录中

### 问题2: "CUDA out of memory"
**解决**: 减小 `--batch_size` 或 `--max_mri_slices`

### 问题3: "FileNotFoundError" 数据文件
**解决**: 检查 CSV 中的路径，使用 `--retinal_base_path` 和 `--mri_base_path`

### 问题4: 损失值为 NaN
**解决**: 降低学习率 `--lr 1e-5`，或检查数据是否包含异常值

---

## 训练流程检查清单

在开始正式训练前，确认：

- [ ] 环境安装完成（`validate_setup.py` 通过）
- [ ] 数据准备完成（CSV 格式正确，文件存在）
- [ ] 小样本测试通过（能正常训练 2-3 个 epoch）
- [ ] 预训练权重已下载（或使用 HuggingFace 自动下载）
- [ ] 超参数已设置（学习率、批次大小等）
- [ ] 保存目录已创建
- [ ] GPU 内存足够（或已调整批次大小）

---

## 下一步

训练完成后，可以：
1. 分析训练曲线（TensorBoard）
2. 生成可视化结果
3. 评估模型性能
4. 进行下游任务（CAD 风险预测）

祝训练顺利！

