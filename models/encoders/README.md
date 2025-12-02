# 编码器模块说明

## 概述

编码器模块包含两个主要组件：
1. **RetinalEncoder**: 视网膜编码器，基于 RETFound ViT-Large
2. **MRICardioEncoder**: 心脏MRI编码器，基于 MedSAM ViT-Base

## RetinalEncoder

### 功能
- 包装 RETFound ViT-Large MAE 编码器
- 输入：预处理后的视网膜眼底图像
- 输出：全局视网膜嵌入 (1024维)

### 使用示例

```python
from models.encoders import RetinalEncoder

# 初始化编码器（不加载预训练权重）
encoder = RetinalEncoder(
    img_size=224,
    pretrained_path=None,
    global_pool=True,
    freeze_backbone=False
)

# 或者从 HuggingFace Hub 加载预训练权重
encoder = RetinalEncoder(
    img_size=224,
    pretrained_path="RETFound_mae",  # HuggingFace Hub ID
    global_pool=True
)

# 或者从本地路径加载
encoder = RetinalEncoder(
    img_size=224,
    pretrained_path="checkpoints/pretrained/retfound/RETFound_mae.pth",
    global_pool=True
)

# 前向传播
retinal_img = torch.randn(4, 3, 224, 224)  # (batch_size, 3, H, W)
embedding = encoder(retinal_img)  # (4, 1024)
```

### 参数说明

- `img_size`: 输入图像尺寸（默认224）
- `pretrained_path`: 预训练权重路径
  - 可以是 HuggingFace Hub ID（如 "RETFound_mae"）
  - 可以是本地文件路径
  - 如果为 None，则随机初始化
- `global_pool`: 是否使用全局平均池化（True）或 CLS token（False）
- `drop_path_rate`: Drop path rate for stochastic depth
- `freeze_backbone`: 是否冻结骨干网络参数

## MRICardioEncoder

### 功能
- 包装 MedSAM ViT-Base 图像编码器
- 处理多切片MRI输入
- 使用可训练池化层聚合切片级特征为主体级嵌入
- 输出：主体级MRI嵌入 (768维)

### 使用示例

```python
from models.encoders import MRICardioEncoder

# 初始化编码器
encoder = MRICardioEncoder(
    img_size=224,
    pretrained_path="checkpoints/pretrained/medsam/medsam_vit_b.pth",
    pooling_type="attention",  # 或 "learnable_weighted", "mean", "max"
    freeze_backbone=False
)

# 前向传播
# 输入形状: (batch_size, num_slices, 1, H, W) 或 (batch_size, num_slices, H, W)
mri_slices = torch.randn(4, 10, 1, 224, 224)  # 4个样本，每个10个切片
embedding = encoder(mri_slices)  # (4, 768)
```

### 参数说明

- `img_size`: 输入图像尺寸（MedSAM内部会调整到1024x1024）
- `pretrained_path`: MedSAM 预训练权重路径
- `pooling_type`: 池化类型
  - `"attention"`: 注意力池化（推荐）
  - `"learnable_weighted"`: 可学习加权池化
  - `"mean"`: 平均池化
  - `"max"`: 最大池化
- `freeze_backbone`: 是否冻结骨干网络参数
- `num_slices`: 预期的切片数量（可选，用于某些池化层的初始化）

### 输入格式

MRI编码器接受以下输入格式：
- `(batch_size, num_slices, 1, H, W)`: 显式通道维度
- `(batch_size, num_slices, H, W)`: 隐式单通道

编码器会自动：
1. 将单通道复制为3通道（MedSAM需要RGB输入）
2. 将图像上采样到1024x1024（MedSAM的期望输入尺寸）
3. 提取每个切片的特征
4. 使用池化层聚合切片特征

## 池化层

### AttentionPooling
使用多头注意力机制对切片进行加权聚合。

```python
from models.encoders import AttentionPooling

pooling = AttentionPooling(
    embed_dim=768,
    num_heads=8,
    dropout=0.1
)
```

### LearnableWeightedPooling
使用可学习的权重网络对切片进行加权平均。

```python
from models.encoders import LearnableWeightedPooling

pooling = LearnableWeightedPooling(embed_dim=768)
```

### MeanPooling / MaxPooling
简单的平均或最大池化（用于对比实验）。

## 注意事项

1. **路径设置**: 
   - RETFound 和 MedSAM 的代码需要位于项目父目录中
   - 或者修改代码中的路径设置

2. **预训练权重**:
   - RETFound: 可以从 HuggingFace Hub 下载（ID: "RETFound_mae"）
   - MedSAM: 需要从官方仓库下载检查点文件

3. **输入尺寸**:
   - RetinalEncoder: 接受任意尺寸，但建议使用224x224
   - MRICardioEncoder: 内部会将输入调整到1024x1024（MedSAM要求）

4. **内存使用**:
   - MRICardioEncoder 处理多切片时会占用较多内存
   - 建议根据GPU内存调整批次大小和切片数量

## 完整示例

```python
import torch
from models.encoders import RetinalEncoder, MRICardioEncoder

# 初始化两个编码器
retinal_encoder = RetinalEncoder(
    img_size=224,
    pretrained_path="RETFound_mae",
    global_pool=True
)

mri_encoder = MRICardioEncoder(
    img_size=224,
    pretrained_path="checkpoints/pretrained/medsam/medsam_vit_b.pth",
    pooling_type="attention"
)

# 处理数据
retinal_img = torch.randn(2, 3, 224, 224)
mri_slices = torch.randn(2, 8, 1, 224, 224)

# 提取嵌入
with torch.no_grad():
    h_R = retinal_encoder(retinal_img)  # (2, 1024)
    h_C = mri_encoder(mri_slices)  # (2, 768)

print(f"Retinal embedding shape: {h_R.shape}")
print(f"MRI embedding shape: {h_C.shape}")
```

