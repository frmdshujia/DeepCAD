# DeepCAD 项目结构设计

## 目录结构

```
DeepCAD/
├── README.md                          # 项目主文档
├── requirements.txt                   # Python 依赖
├── setup.py                          # 安装脚本（可选）
│
├── configs/                          # 配置文件目录
│   ├── config_stage1.yaml           # Stage I 主配置文件
│   ├── datasets/                     # 数据集配置
│   │   └── retina_cardiac.yaml      # 视网膜+心脏MRI数据集配置
│   └── models/                       # 模型配置
│       ├── retinal_encoder.yaml     # 视网膜编码器配置
│       └── mri_encoder.yaml         # MRI编码器配置
│
├── datasets/                         # 数据集模块
│   ├── __init__.py
│   ├── retina_cardiac_dataset.py    # RetinaCardiacDataset 主类
│   ├── transforms.py                # 数据增强和预处理
│   └── data_utils.py                # 数据加载工具函数
│
├── models/                           # 模型定义
│   ├── __init__.py
│   ├── encoders/                     # 编码器模块
│   │   ├── __init__.py
│   │   ├── retinal_encoder.py        # RETFound 视网膜编码器包装器
│   │   ├── mri_encoder.py            # MedSAM 心脏MRI编码器包装器
│   │   └── pooling.py                # 可训练池化层（用于MRI切片聚合）
│   ├── projection_heads.py          # 投影头（MLP）
│   ├── deepcad_stage1.py            # DeepCAD Stage I 完整模型
│   └── downstream/                   # 下游任务模型（未来扩展）
│       └── cad_classifier.py        # CAD风险预测分类器
│
├── losses/                           # 损失函数
│   ├── __init__.py
│   └── cross_modal_contrastive.py    # 监督跨模态对比损失
│
├── trainers/                         # 训练器模块
│   ├── __init__.py
│   ├── stage1_trainer.py            # Stage I 训练器
│   └── evaluator.py                 # 评估器（用于验证和测试）
│
├── utils/                            # 工具函数
│   ├── __init__.py
│   ├── checkpoint.py                # 模型检查点保存/加载
│   ├── logger.py                     # 日志记录（TensorBoard/WandB）
│   ├── metrics.py                   # 评估指标
│   └── visualization.py             # 可视化工具
│
├── explainability/                   # 可解释性模块
│   ├── __init__.py
│   ├── grad_cam.py                  # Grad-CAM 实现（ViT适配）
│   └── cross_modal_viz.py           # 跨模态可视化脚本
│
├── scripts/                          # 可执行脚本
│   ├── train_stage1.py              # Stage I 训练主脚本
│   ├── evaluate.py                  # 评估脚本
│   ├── visualize_gradcam.py         # Grad-CAM 可视化脚本
│   └── extract_embeddings.py        # 提取嵌入向量脚本
│
├── data/                             # 数据目录（不提交到git）
│   ├── raw/                         # 原始数据
│   ├── processed/                   # 处理后的数据
│   └── splits/                      # 训练/验证/测试集划分
│
├── checkpoints/                      # 模型检查点（不提交到git）
│   ├── pretrained/                  # 预训练权重
│   │   ├── retfound/                # RETFound 预训练权重
│   │   └── medsam/                  # MedSAM 预训练权重
│   └── stage1/                      # Stage I 训练检查点
│
├── logs/                             # 训练日志（不提交到git）
│   └── tensorboard/                 # TensorBoard 日志
│
└── tests/                            # 单元测试
    ├── __init__.py
    ├── test_dataset.py
    ├── test_encoders.py
    ├── test_loss.py
    └── test_trainer.py
```

## 模块职责说明

### 1. `configs/` - 配置文件
- **职责**: 存储所有配置参数（超参数、数据路径、模型架构等）
- **设计原则**: 使用 YAML 格式，便于修改和版本控制
- **主要文件**:
  - `config_stage1.yaml`: Stage I 训练的主配置
  - `datasets/retina_cardiac.yaml`: 数据集路径、预处理参数
  - `models/*.yaml`: 编码器和投影头的架构参数

### 2. `datasets/` - 数据集模块
- **职责**: 定义 PyTorch Dataset 和 DataLoader
- **主要类**:
  - `RetinaCardiacDataset`: 加载视网膜图像和心脏MRI数据，返回配对样本和CAD标签
- **关键功能**:
  - 分别处理视网膜和MRI的变换/增强
  - 支持多切片MRI数据的加载和堆叠
  - 返回字典格式：`{'x_R': retinal_tensor, 'x_C': mri_tensor, 'y': label, 'subject_id': id}`

### 3. `models/encoders/` - 编码器模块
- **职责**: 封装 RETFound 和 MedSAM 作为特征提取器
- **主要类**:
  - `RetinalEncoder`: 
    - 包装 RETFound ViT-Large MAE 编码器
    - 加载预训练权重
    - 输出全局视网膜嵌入 `h_j^R`
  - `MRICardioEncoder`:
    - 包装 MedSAM ViT-Base 骨干网络
    - 处理多切片输入（cine + T1）
    - 使用可训练池化层聚合切片级特征
    - 输出主体级MRI嵌入 `h_j^C`
  - `AttentionPooling` / `LearnablePooling`: 可训练池化机制

### 4. `models/projection_heads.py` - 投影头
- **职责**: 将编码器输出映射到共享潜在空间
- **主要类**:
  - `ProjectionHead`: 简单的 MLP（2-3层）
  - 输出 L2 归一化的嵌入 `z_j^R` 和 `z_j^C`
  - 支持可配置的隐藏层维度和激活函数

### 5. `models/deepcad_stage1.py` - 完整模型
- **职责**: 整合所有组件
- **主要类**:
  - `DeepCADStageI`: 
    - 包含视网膜编码器、MRI编码器、两个投影头
    - 前向传播返回 `z_R` 和 `z_C`
    - 支持冻结/解冻编码器参数

### 6. `losses/` - 损失函数
- **职责**: 实现监督跨模态对比损失
- **主要函数**:
  - `cross_modal_contrastive_loss(z_R, z_C, labels, tau=0.1)`:
    - 计算心脏→视网膜损失 `L_C`
    - 计算视网膜→心脏损失 `L_R`
    - 返回总损失 `L = L_C + L_R` 和单独的分量（用于日志）

### 7. `trainers/` - 训练器
- **职责**: 训练循环和验证逻辑
- **主要类**:
  - `Stage1Trainer`:
    - 管理优化器、学习率调度器
    - 实现 `train_one_epoch()` 和 `validate_one_epoch()`
    - 处理检查点保存/加载
    - 集成日志记录
  - `Evaluator`: 评估模型性能

### 8. `explainability/` - 可解释性
- **职责**: Grad-CAM 和跨模态可视化
- **主要类/函数**:
  - `VitGradCAM`: ViT 适配的 Grad-CAM 实现
  - `visualize_cross_modal`: 生成跨模态对齐可视化

### 9. `scripts/` - 可执行脚本
- **职责**: 命令行入口点
- **主要脚本**:
  - `train_stage1.py`: 主训练脚本，解析配置并启动训练
  - `evaluate.py`: 评估训练好的模型
  - `visualize_gradcam.py`: 生成 Grad-CAM 热图
  - `extract_embeddings.py`: 提取并保存嵌入向量（用于下游任务）

### 10. `utils/` - 工具函数
- **职责**: 通用工具和辅助函数
- **主要模块**:
  - `checkpoint.py`: 模型保存/加载
  - `logger.py`: TensorBoard/WandB 集成
  - `metrics.py`: 评估指标（准确率、AUC等）
  - `visualization.py`: 通用可视化工具

## 设计原则

1. **模块化**: 每个组件独立，便于测试和替换
2. **可扩展性**: 易于添加下游任务（分类器、回归器等）
3. **可配置性**: 通过 YAML 配置文件管理所有超参数
4. **可解释性**: 内置 Grad-CAM 和可视化支持
5. **兼容性**: 复用 MMCL、RETFound、MedSAM 的代码结构，便于集成

## 未来扩展点

- `models/downstream/`: 下游任务模型（CAD风险预测）
- `scripts/finetune_downstream.py`: 下游任务微调脚本
- `explainability/attention_visualization.py`: 注意力机制可视化
- `data/preprocessing/`: 数据预处理管道

