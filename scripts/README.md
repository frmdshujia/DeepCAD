# 训练脚本说明

## train_stage1.py

DeepCAD Stage I 的主训练脚本。

### 使用方法

#### 基本用法

```bash
python scripts/train_stage1.py \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --batch_size 32 \
    --num_epochs 100 \
    --lr 1e-4
```

#### 完整参数示例

```bash
python scripts/train_stage1.py \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --retinal_base_path data/retinal \
    --mri_base_path data/mri \
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
    --save_dir checkpoints/stage1 \
    --log_dir logs/stage1 \
    --device cuda
```

### 主要参数说明

#### 数据参数
- `--train_csv`: 训练集CSV路径（必需）
- `--val_csv`: 验证集CSV路径（可选）
- `--retinal_base_path`: 视网膜图像基础路径
- `--mri_base_path`: MRI数据基础路径

#### 模型参数
- `--retinal_pretrained`: RETFound预训练权重（HuggingFace Hub ID或本地路径）
- `--mri_pretrained`: MedSAM预训练权重路径
- `--latent_dim`: 共享潜在空间维度（默认128）
- `--mri_pooling_type`: MRI池化类型（attention/learnable_weighted/mean/max）

#### 训练参数
- `--batch_size`: 批次大小（默认32）
- `--num_epochs`: 训练轮数（默认100）
- `--lr`: 学习率（默认1e-4）
- `--temperature`: 对比损失温度参数（默认0.1）
- `--optimizer`: 优化器类型（adam/adamw/sgd）
- `--scheduler`: 学习率调度器（cosine/step/plateau/none）

#### 其他参数
- `--freeze_encoders`: 冻结编码器，只训练投影头
- `--resume_from`: 恢复训练的检查点路径
- `--device`: 设备（cuda/cpu，默认自动选择）

### 输出

训练过程中会生成：
- **检查点**: `checkpoints/stage1/checkpoint_epoch_N.pth`
- **最佳模型**: `checkpoints/stage1/best_model.pth`
- **最新检查点**: `checkpoints/stage1/latest_checkpoint.pth`
- **日志**: `logs/stage1/training_TIMESTAMP.log`
- **TensorBoard日志**: `logs/stage1/tensorboard/`（如果安装了TensorBoard）

### 恢复训练

```bash
python scripts/train_stage1.py \
    --train_csv data/splits/train.csv \
    --val_csv data/splits/val.csv \
    --resume_from checkpoints/stage1/latest_checkpoint.pth \
    --num_epochs 100
```

### 查看训练日志

如果安装了TensorBoard：

```bash
tensorboard --logdir logs/stage1/tensorboard
```

然后在浏览器中打开 `http://localhost:6006`

