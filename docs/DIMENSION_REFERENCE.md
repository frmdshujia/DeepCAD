# 维度设置参考文档

本文档详细说明 DeepCAD 项目中所有维度数字的来源和设计理由。

## 核心维度说明

### 1. 视网膜编码器输出维度: **1024**

**来源**: RETFound ViT-Large 架构

**参考代码**:
- `RETFound-main/models_vit.py` 中的 `RETFound_mae()` 函数
- 配置: `embed_dim=1024, depth=24, num_heads=16`

**说明**:
- 这是 Vision Transformer Large (ViT-L) 的标准嵌入维度
- 与标准 ViT-Large 配置一致 (patch_size=16, embed_dim=1024)
- RETFound 使用 ViT-Large 作为骨干网络，因此输出维度为 1024

**在代码中的位置**:
- `models/encoders/retinal_encoder.py`: `self.embed_dim = 1024`
- `models/deepcad_stage1.py`: `retinal_embed_dim = 1024`

---

### 2. MedSAM 原始输出维度: **256**

**来源**: MedSAM/SAM 架构设计

**参考代码**:
- `MedSAM-main/segment_anything/build_sam.py` 中的 `_build_sam()` 函数
- 配置: `prompt_embed_dim = 256`, `out_chans=prompt_embed_dim=256`

**说明**:
- 这是 SAM 架构中的 prompt embedding 维度
- MedSAM image_encoder 输出形状为 `(B, 256, 64, 64)` for 1024x1024 input
- 256 维是 SAM 架构的设计，用于与 prompt encoder 和 mask decoder 对接

**在代码中的位置**:
- `models/encoders/mri_encoder.py`: `self.backbone_out_dim = 256`

---

### 3. MRI编码器最终输出维度: **768**

**来源**: MedSAM ViT-Base 的 encoder_embed_dim

**参考代码**:
- `MedSAM-main/segment_anything/build_sam.py` 中的 `build_sam_vit_b()` 函数
- 配置: `encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12`

**说明**:
- 这是 Vision Transformer Base (ViT-B) 的标准嵌入维度
- 虽然 MedSAM 原始输出是 256 维（用于 SAM 任务），但我们将其投影回 768 维
- **设计选择理由**:
  1. 768 是 ViT-Base 的标准维度，特征更丰富
  2. 与视网膜编码器的 1024 维在数量级上更接近
  3. 便于后续投影头设计（两个编码器输出维度差异不会太大）

**在代码中的位置**:
- `models/encoders/mri_encoder.py`: `self.embed_dim = 768`
- `models/deepcad_stage1.py`: `mri_embed_dim = 768`

---

### 4. 共享潜在空间维度: **128**

**来源**: 设计选择，参考 MMCL-Tabular-Imaging 项目

**参考**:
- MMCL-Tabular-Imaging 项目中的常见设置
- 对比学习中的常见实践（SimCLR、CLIP 等）

**说明**:
- 这是两个模态投影后的统一维度，用于计算跨模态相似度
- **设计理由**:
  1. 较小的维度有助于减少计算量
  2. 提高泛化能力（避免过拟合）
  3. 便于后续下游任务
  4. 128 是常用的平衡点（不会太小丢失信息，也不会太大增加计算）

**可调整值**: 64, 128, 256, 512

**在代码中的位置**:
- `models/deepcad_stage1.py`: `latent_dim: int = 128` (默认)
- `models/projection_heads.py`: `output_dim: int = 128` (默认)
- 可通过 `--latent_dim` 参数修改

---

### 5. 投影头隐藏层维度: **1024** (默认)

**来源**: 自动计算 `max(retinal_embed_dim, mri_embed_dim)`

**说明**:
- 默认使用两个编码器输出维度的最大值: `max(1024, 768) = 1024`
- **设计理由**: 确保有足够的容量进行特征变换
- 可以手动指定其他值

**在代码中的位置**:
- `models/deepcad_stage1.py`: `projection_hidden_dim = max(retinal_embed_dim, mri_embed_dim)`

---

### 6. 图像尺寸: **224** (默认输入)

**来源**: ImageNet 预训练模型的标准输入尺寸

**说明**:
- 224x224 是 ImageNet 预训练模型（包括 ViT）的标准输入尺寸
- RETFound 和大多数 ViT 模型都使用 224x224 作为输入
- **注意**: MedSAM 内部会将输入调整到 1024x1024（这是 SAM 架构的要求）

**在代码中的位置**:
- `models/deepcad_stage1.py`: `retinal_img_size: int = 224`, `mri_img_size: int = 224`
- `datasets/retina_cardiac_dataset.py`: 默认 `img_size=224`

---

### 7. MedSAM 内部图像尺寸: **1024**

**来源**: SAM/MedSAM 架构设计

**参考代码**:
- `MedSAM-main/segment_anything/build_sam.py`: `image_size = 1024`

**说明**:
- 这是 SAM/MedSAM 架构的标准输入尺寸
- 即使我们输入 224x224，MedSAM 编码器内部会将其上采样到 1024x1024
- 这是 SAM 架构的设计要求

**在代码中的位置**:
- `models/encoders/mri_encoder.py`: 自动上采样到 1024x1024

---

### 8. 温度参数: **0.1**

**来源**: DeepCAD_README_with_prompts.md 中的数学定义

**参考文档**:
- `DeepCAD_README_with_prompts.md` 第 66 行: "where cos(·,·) is cosine similarity and τ is a temperature parameter (e.g. τ = 0.1)"

**说明**:
- 温度参数用于缩放相似度分数
- 较小的值（0.1）使模型更关注困难样本
- 这是对比学习中常用的设置，参考了 SimCLR、CLIP 等工作的经验

**可调整范围**: 0.05 (更关注困难样本) 到 0.5 (更平滑的分布)

**在代码中的位置**:
- `losses/cross_modal_contrastive.py`: `tau: float = 0.1` (默认)
- `scripts/train_stage1.py`: `--temperature 0.1` (默认)

---

## 维度流转图

```
输入:
├── 视网膜图像 (3×224×224)
│   └── RETFound ViT-Large
│       └── h_R: (B, 1024)  ← 维度来源: RETFound 架构
│           └── ProjectionHead(1024 → 1024 → 128)
│               └── z_R: (B, 128)  ← 维度来源: 设计选择
│
└── MRI切片 (N×1×224×224)
    └── 上采样到 1024×1024  ← 维度来源: MedSAM 架构要求
        └── MedSAM ViT-Base
            └── slice_features: (B*N, 256, 64, 64)  ← 维度来源: SAM 架构
                └── 投影层 (256 → 768)
                    └── h_C: (B, 768)  ← 维度来源: ViT-Base 标准维度
                        └── ProjectionHead(768 → 1024 → 128)
                            └── z_C: (B, 128)  ← 维度来源: 设计选择（与 z_R 相同）

共享空间:
└── z_R, z_C: (B, 128)  ← 用于计算跨模态相似度
```

## 总结表

| 维度 | 数值 | 来源 | 参考文件/函数 | 可调整性 |
|------|------|------|--------------|----------|
| 视网膜编码器输出 | 1024 | RETFound ViT-Large | `RETFound-main/models_vit.py::RETFound_mae()` | ❌ 固定（架构决定） |
| MedSAM 原始输出 | 256 | SAM 架构 | `MedSAM-main/build_sam.py::_build_sam()` | ❌ 固定（架构决定） |
| MRI编码器最终输出 | 768 | ViT-Base 标准维度 | `MedSAM-main/build_sam.py::build_sam_vit_b()` | ⚠️ 可调整（但需重新设计） |
| 共享潜在空间 | 128 | 设计选择 | 参考 MMCL 项目 | ✅ 可通过参数调整 |
| 投影头隐藏层 | 1024 | 自动计算 | `max(1024, 768)` | ✅ 可通过参数调整 |
| 输入图像尺寸 | 224 | ImageNet 标准 | 通用实践 | ✅ 可通过参数调整 |
| MedSAM 内部尺寸 | 1024 | SAM 架构 | `MedSAM-main/build_sam.py` | ❌ 固定（架构决定） |
| 温度参数 | 0.1 | README 定义 | `DeepCAD_README_with_prompts.md` | ✅ 可通过参数调整 |

## 验证方法

要验证这些维度设置是否正确，可以运行：

```bash
python scripts/debug_model.py --train_csv data/splits/train.csv
```

这会检查所有维度是否匹配。

