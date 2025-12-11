# DeepCAD 环境安装指南

## 概述

DeepCAD 项目整合了三个外部项目：
- **RETFound**: 视网膜基础模型（需要 Python 3.11, PyTorch 2.5.1, CUDA 12.1）
- **MedSAM**: 医学图像分割模型（需要 Python >=3.9, PyTorch 2.0, numpy >=2.0.0）
- **MMCL-Tabular-Imaging**: 多模态对比学习框架（需要 Python 3.9, PyTorch 1.11.0, CUDA 11.3）

由于这些项目的依赖存在冲突，我们采用**兼容性安装策略**，选择一个折中的版本配置。

## 兼容性策略

### 核心原则
1. **不直接安装外部项目的依赖**：DeepCAD 通过 `sys.path` 导入外部代码，不需要安装它们的包
2. **选择兼容的版本**：在三个项目的需求之间找到平衡点
3. **分步安装**：先安装核心依赖，再处理可能有冲突的包

### 推荐配置

| 组件 | 推荐版本 | 说明 |
|------|---------|------|
| Python | 3.10 | 兼容 MedSAM (>=3.9) 和 RETFound (3.11)，3.10 更稳定 |
| PyTorch | 2.0.1 | 兼容 MedSAM 和 RETFound（虽然 RETFound 推荐 2.5.1，但 2.0.1 也能工作） |
| CUDA | 11.8 或 12.1 | 根据您的 GPU 驱动选择 |
| numpy | 1.26.4 | RETFound 需要，MedSAM 的 numpy>=2.0.0 要求可能过于严格 |
| timm | 0.9.2 | RETFound 需要 |

## 安装步骤

### 方法一：使用安装脚本（推荐）

```bash
cd DeepCAD
bash scripts/install_environment.sh
```

### 方法二：手动安装

#### 步骤 1: 创建虚拟环境

```bash
# 使用 conda（推荐）
conda create -n deepcad python=3.10 -y
conda activate deepcad

# 或使用 venv
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

#### 步骤 2: 安装 PyTorch（根据您的 CUDA 版本）

**CUDA 11.8:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
```

**CUDA 12.1:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu121
```

**CPU only（不推荐，训练会很慢）:**
```bash
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cpu
```

#### 步骤 3: 安装核心依赖（兼容版本）

```bash
cd DeepCAD

# 安装 numpy（固定版本以避免冲突）
pip install "numpy==1.26.4"

# 安装其他核心依赖
pip install \
    "pillow>=9.0.0,<11.0.0" \
    "pandas>=1.3.0" \
    "scipy>=1.7.0" \
    "pyyaml>=6.0" \
    "matplotlib>=3.5.0" \
    "seaborn>=0.11.0" \
    "tensorboard>=2.10.0" \
    "nibabel>=3.2.0" \
    "scikit-image>=0.19.0" \
    "tqdm>=4.64.0" \
    "omegaconf>=2.2.0"
```

#### 步骤 4: 安装 RETFound 相关依赖

```bash
# timm（RETFound 需要）
pip install "timm==0.9.2"

# opencv-python（RETFound 需要特定版本）
pip install "opencv-python==4.9.0.80"

# huggingface-hub（用于下载 RETFound 权重）
pip install "huggingface-hub>=0.23.4"
```

#### 步骤 5: 安装 MedSAM 相关依赖

```bash
# SimpleITK（MedSAM 需要）
pip install "SimpleITK>=2.2.1"

# 注意：MedSAM 的 setup.py 要求 numpy>=2.0.0，但实际测试中 numpy 1.26.4 也能工作
# 如果遇到问题，可以尝试：
# pip install "numpy>=2.0.0"  # 但这可能与 RETFound 冲突
```

#### 步骤 6: 安装可选依赖

```bash
# 用于可视化和分析
pip install scikit-learn

# 用于实验跟踪（可选）
pip install wandb
```

#### 步骤 7: 验证安装

```bash
python scripts/validate_setup.py
```

## 外部项目准备

DeepCAD 需要外部项目的代码（但不安装它们的包）：

### 1. RETFound

```bash
cd ..  # 回到项目父目录
git clone https://github.com/rmaphoh/RETFound RETFound-main
# 注意：不需要安装 RETFound 的依赖，只需要代码存在
```

### 2. MedSAM

```bash
cd ..  # 回到项目父目录
git clone https://github.com/bowang-lab/MedSAM MedSAM-main
# 注意：不需要运行 pip install -e .，只需要代码存在
```

### 3. MMCL-Tabular-Imaging（可选）

MMCL 主要用于参考训练框架，DeepCAD 已经实现了自己的训练器，所以这个项目是可选的。

## 版本冲突处理

### 问题 1: numpy 版本冲突

**症状**: MedSAM 要求 numpy>=2.0.0，但 RETFound 需要 numpy~=1.26.4

**解决方案**: 
- 使用 numpy 1.26.4（推荐）
- MedSAM 的 numpy>=2.0.0 要求可能过于严格，实际测试中 1.26.4 也能工作
- 如果确实遇到问题，可以考虑：
  1. 修改 MedSAM 的 setup.py（不推荐）
  2. 使用两个不同的环境（不推荐，太复杂）

### 问题 2: PyTorch 版本冲突

**症状**: RETFound 推荐 PyTorch 2.5.1，MedSAM 推荐 PyTorch 2.0

**解决方案**:
- 使用 PyTorch 2.0.1（兼容两者）
- RETFound 的代码在 PyTorch 2.0.1 上也能正常工作
- 如果遇到特定 API 不兼容，可以：
  1. 升级到 PyTorch 2.5.1（需要确保其他依赖兼容）
  2. 修改 RETFound 代码中的 API 调用

### 问题 3: CUDA 版本冲突

**症状**: RETFound 需要 CUDA 12.1，MMCL 需要 CUDA 11.3

**解决方案**:
- 使用 CUDA 11.8 或 12.1（根据您的 GPU 驱动）
- MMCL 主要用于参考，DeepCAD 不直接依赖它
- 确保 PyTorch 的 CUDA 版本与系统 CUDA 版本匹配

### 问题 4: Python 版本冲突

**症状**: RETFound 需要 Python 3.11，MedSAM 需要 >=3.9

**解决方案**:
- 使用 Python 3.10（推荐，最稳定）
- Python 3.11 也可以，但某些包可能还没有完全支持

## 验证清单

安装完成后，运行以下检查：

```bash
# 1. 检查 Python 版本
python --version  # 应该是 3.10.x

# 2. 检查 PyTorch
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# 3. 检查 numpy
python -c "import numpy; print(f'numpy: {numpy.__version__}')"  # 应该是 1.26.4

# 4. 检查外部项目
python scripts/validate_setup.py

# 5. 测试导入
python -c "
import sys
sys.path.insert(0, '../RETFound-main')
sys.path.insert(0, '../MedSAM-main')
try:
    import models_vit as retfound_models
    print('✓ RETFound 导入成功')
except ImportError as e:
    print(f'✗ RETFound 导入失败: {e}')

try:
    from segment_anything import build_sam_vit_b
    print('✓ MedSAM 导入成功')
except ImportError as e:
    print(f'✗ MedSAM 导入失败: {e}')
"
```

## 常见问题

### Q1: 安装时出现 "numpy 版本冲突" 错误

**A**: 先卸载所有 numpy，然后安装固定版本：
```bash
pip uninstall numpy -y
pip install "numpy==1.26.4"
```

### Q2: 导入 RETFound 时出错

**A**: 确保：
1. RETFound-main 在项目父目录中
2. 已安装 timm==0.9.2
3. 检查 RETFound 的代码路径是否正确

### Q3: 导入 MedSAM 时出错

**A**: 确保：
1. MedSAM-main 在项目父目录中
2. 已安装 SimpleITK
3. 检查 MedSAM 的代码路径是否正确

### Q4: CUDA out of memory

**A**: 这不是环境问题，是 GPU 内存不足。减小 batch_size 或使用梯度累积。

## 下一步

环境安装完成后，请参考：
- [QUICKSTART.md](QUICKSTART.md) - 快速开始指南
- [DEPLOYMENT_PREPARATION.md](DEPLOYMENT_PREPARATION.md) - 部署准备清单


