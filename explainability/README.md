# 可解释性模块说明

## 概述

本模块提供了 Grad-CAM 和跨模态可视化功能，用于理解 DeepCAD 模型的决策过程。

## Grad-CAM

### VitGradCAM 类

用于 Vision Transformer 的 Grad-CAM 实现。

#### 基本使用

```python
from models.deepcad_stage1 import DeepCADStageI
from explainability import VitGradCAM

# 加载模型
model = DeepCADStageI(...)
model.load_state_dict(torch.load('checkpoint.pth'))

# 创建 Grad-CAM
gradcam = VitGradCAM(model, use_cuda=True)

# 生成热图
retinal_image = ...  # (1, 3, H, W)
cam = gradcam.generate_cam(retinal_image)

# 叠加到原始图像
original_image = ...  # numpy array, (H, W, 3)
overlay = gradcam.overlay_heatmap(original_image, cam)
```

#### 基于相似度的 Grad-CAM

```python
# 使用跨模态相似度作为目标
retinal_image = ...  # (1, 3, H, W)
cardiac_embedding = ...  # (1, D)

cam = gradcam.generate_cam_for_similarity(
    retinal_image,
    cardiac_embedding
)
```

### 方法说明

- `generate_cam()`: 生成标准 Grad-CAM 热图
- `generate_cam_for_similarity()`: 基于跨模态相似度生成 Grad-CAM
- `overlay_heatmap()`: 将热图叠加到原始图像
- `visualize_patch_attention()`: 将 patch 级别的注意力映射回图像空间

## 跨模态可视化

### visualize_cross_modal()

可视化单个样本的跨模态关联。

```python
from explainability.cross_modal_viz import visualize_cross_modal

visualize_cross_modal(
    model=model,
    retinal_image=retinal_image,
    mri_slices=mri_slices,
    retinal_gradcam=cam,
    device="cuda",
    save_path="visualization.png"
)
```

输出包含：
1. 原始视网膜图像
2. 带 Grad-CAM 叠加的视网膜图像
3. 代表性 MRI 切片
4. 嵌入空间的 2D 投影（显示两个模态的关联）

### visualize_batch()

批量可视化多个样本。

```python
from explainability.cross_modal_viz import visualize_batch

visualize_batch(
    model=model,
    dataloader=val_loader,
    num_samples=10,
    device="cuda",
    save_dir="visualizations"
)
```

## 命令行工具

### visualize_gradcam.py

使用命令行脚本进行可视化：

```bash
# 从数据集批量可视化
python scripts/visualize_gradcam.py \
    --checkpoint checkpoints/stage1/best_model.pth \
    --data_csv data/splits/val.csv \
    --num_samples 10 \
    --save_dir visualizations

# 单个图像可视化
python scripts/visualize_gradcam.py \
    --checkpoint checkpoints/stage1/best_model.pth \
    --retinal_image path/to/retinal.jpg \
    --mri_paths path/to/mri1.nii.gz path/to/mri2.nii.gz \
    --save_dir visualizations
```

## 技术细节

### ViT Grad-CAM 实现

对于 Vision Transformer：
1. 选择目标层（通常是最后一个注意力块）
2. 注册前向和反向钩子捕获激活和梯度
3. 计算梯度加权激活图
4. 将 patch 级别的注意力映射回图像空间

### 跨模态相似度

相似度计算：
```python
similarity = torch.sum(z_R * z_C, dim=1)
```
其中 `z_R` 和 `z_C` 是 L2 归一化的投影嵌入。

## 注意事项

1. **目标层选择**: 默认选择最后一个注意力块，可以通过 `target_layer` 参数自定义
2. **内存使用**: Grad-CAM 需要存储激活和梯度，大模型可能占用较多内存
3. **图像预处理**: 确保输入图像与训练时使用相同的预处理
4. **设备**: 建议使用 GPU 以加速计算

## 依赖

- PyTorch
- NumPy
- OpenCV (cv2)
- Matplotlib
- PIL/Pillow
- scikit-learn (可选，用于 PCA 可视化)

