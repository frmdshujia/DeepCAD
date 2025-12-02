# DeepCAD: Cross-Modal Contrastive Learning Between Retinal Fundus and Cardiac MRI

DeepCAD 是一个多模态学习项目，通过学习视网膜眼底照片和心脏 MRI 之间的共享潜在空间，使用监督跨模态对比学习来预测阻塞性 CAD，并支持跨模态可解释性（例如，视网膜侧的 Grad-CAM 可视化与心脏 MRI 特征对齐）。

## 项目结构

详细的项目结构说明请参考 [PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 训练 Stage I

```bash
python scripts/train_stage1.py --config configs/config_stage1.yaml
```

## 设计概述

本项目结合了三个 GitHub 项目的思想：

- **MMCL-Tabular-Imaging** – 多模态对比学习框架和训练骨架
- **RETFound** – 视网膜基础模型（ViT-Large MAE 编码器）作为强大的视网膜骨干网络
- **MedSAM** – 医学基础模型（ViT-Base）作为多功能心脏 MRI 编码器

## 开发阶段

本项目按照以下阶段逐步开发：

1. ✅ **项目结构设计** - 设计模块化项目结构
2. ⏳ **数据集和数据加载器** - 实现 RetinaCardiacDataset
3. ⏳ **编码器包装器** - 实现 RETFound 和 MedSAM 编码器
4. ⏳ **投影头和损失函数** - 实现投影头和监督跨模态对比损失
5. ⏳ **训练脚本** - 实现 Stage I 训练循环
6. ⏳ **可解释性** - 实现 Grad-CAM 和跨模态可视化
7. ⏳ **迭代和调试** - 代码审查和错误修复框架

## 引用

如果使用本项目，请引用相关的基础项目：
- MMCL-Tabular-Imaging
- RETFound
- MedSAM

