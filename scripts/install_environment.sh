#!/bin/bash
# DeepCAD 环境安装脚本（兼容三个外部项目）

set -e  # 遇到错误立即退出

echo "=========================================="
echo "DeepCAD 环境安装（兼容 RETFound/MedSAM/MMCL）"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python
echo "1. 检查 Python 版本..."
if command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
elif command -v python &> /dev/null; then
    PYTHON_CMD=python
else
    echo -e "${RED}错误: 未找到 Python${NC}"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
echo "   Python 版本: $PYTHON_VERSION"

# 检查 Python 版本是否符合要求
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 9 ]); then
    echo -e "${YELLOW}警告: 推荐使用 Python 3.10${NC}"
fi

# 创建虚拟环境
echo ""
read -p "是否创建新的虚拟环境? (y/n, 默认: y): " create_venv
create_venv=${create_venv:-y}

if [ "$create_venv" = "y" ]; then
    read -p "虚拟环境名称 (默认: deepcad): " venv_name
    venv_name=${venv_name:-deepcad}
    
    # 检查是否使用 conda
    if command -v conda &> /dev/null; then
        echo ""
        echo "2. 使用 conda 创建环境..."
        conda create -n $venv_name python=3.10 -y
        echo -e "${GREEN}   ✓ 环境创建成功${NC}"
        echo ""
        echo -e "${YELLOW}   请运行以下命令激活环境:${NC}"
        echo "   conda activate $venv_name"
        echo ""
        read -p "   激活环境后按 Enter 继续..."
    else
        echo ""
        echo "2. 使用 venv 创建环境..."
        $PYTHON_CMD -m venv $venv_name
        echo -e "${GREEN}   ✓ 环境创建成功${NC}"
        echo ""
        echo -e "${YELLOW}   请运行以下命令激活环境:${NC}"
        echo "   source $venv_name/bin/activate  # Linux/Mac"
        echo "   或"
        echo "   $venv_name\\Scripts\\activate  # Windows"
        echo ""
        read -p "   激活环境后按 Enter 继续..."
    fi
fi

# 检查是否在虚拟环境中
if [ -z "$VIRTUAL_ENV" ] && [ -z "$CONDA_DEFAULT_ENV" ]; then
    echo -e "${YELLOW}警告: 未检测到虚拟环境，建议在虚拟环境中安装${NC}"
    read -p "是否继续? (y/n): " continue_install
    if [ "$continue_install" != "y" ]; then
        exit 0
    fi
fi

# 检测 CUDA 版本
echo ""
echo "3. 检测 CUDA 版本..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d, -f1)
    echo "   检测到 CUDA: $CUDA_VERSION"
    
    # 根据 CUDA 版本选择 PyTorch
    CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d. -f1)
    CUDA_MINOR=$(echo $CUDA_VERSION | cut -d. -f2)
    
    if [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
        CUDA_INDEX="cu121"
        echo "   使用 CUDA 12.1 版本的 PyTorch"
    elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        CUDA_INDEX="cu118"
        echo "   使用 CUDA 11.8 版本的 PyTorch"
    else
        CUDA_INDEX="cu118"
        echo -e "${YELLOW}   未检测到支持的 CUDA 版本，默认使用 CUDA 11.8${NC}"
    fi
else
    echo "   未检测到 CUDA，将安装 CPU 版本的 PyTorch"
    CUDA_INDEX="cpu"
fi

# 安装 PyTorch
echo ""
echo "4. 安装 PyTorch 2.0.1..."
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/$CUDA_INDEX

# 验证 PyTorch 安装
echo ""
echo "   验证 PyTorch 安装..."
python -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   CUDA 可用: {torch.cuda.is_available()}')" || {
    echo -e "${RED}   PyTorch 安装失败${NC}"
    exit 1
}

# 安装 numpy（固定版本以避免冲突）
echo ""
echo "5. 安装 numpy 1.26.4（兼容版本）..."
pip uninstall numpy -y 2>/dev/null || true
pip install "numpy==1.26.4"

# 安装核心依赖
echo ""
echo "6. 安装核心依赖..."
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

# 安装 RETFound 相关依赖
echo ""
echo "7. 安装 RETFound 相关依赖..."
pip install \
    "timm==0.9.2" \
    "opencv-python==4.9.0.80" \
    "huggingface-hub>=0.23.4"

# 安装 MedSAM 相关依赖
echo ""
echo "8. 安装 MedSAM 相关依赖..."
pip install "SimpleITK>=2.2.1"

# 安装可选依赖
echo ""
read -p "是否安装可选依赖 (scikit-learn, wandb)? (y/n, 默认: y): " install_optional
install_optional=${install_optional:-y}
if [ "$install_optional" = "y" ]; then
    pip install scikit-learn wandb
fi

# 检查外部依赖
echo ""
echo "9. 检查外部依赖..."

# 检查 RETFound
if [ -d "../RETFound-main" ]; then
    echo -e "   ${GREEN}✓ RETFound 找到: ../RETFound-main${NC}"
else
    echo -e "   ${YELLOW}⚠️  RETFound 未找到: ../RETFound-main${NC}"
    echo "      请克隆 RETFound 仓库到项目父目录:"
    echo "      cd .. && git clone https://github.com/rmaphoh/RETFound RETFound-main"
fi

# 检查 MedSAM
if [ -d "../MedSAM-main" ]; then
    echo -e "   ${GREEN}✓ MedSAM 找到: ../MedSAM-main${NC}"
else
    echo -e "   ${YELLOW}⚠️  MedSAM 未找到: ../MedSAM-main${NC}"
    echo "      请克隆 MedSAM 仓库到项目父目录:"
    echo "      cd .. && git clone https://github.com/bowang-lab/MedSAM MedSAM-main"
fi

# 验证安装
echo ""
echo "10. 验证安装..."
if [ -f "scripts/validate_setup.py" ]; then
    python scripts/validate_setup.py
else
    echo "   验证脚本未找到，跳过验证"
fi

# 测试导入
echo ""
echo "11. 测试外部项目导入..."
python -c "
import sys
import os

# 测试 RETFound
retfound_path = os.path.join('..', 'RETFound-main')
if os.path.exists(retfound_path):
    sys.path.insert(0, retfound_path)
    try:
        import models_vit as retfound_models
        print('   ✓ RETFound 导入成功')
    except ImportError as e:
        print(f'   ✗ RETFound 导入失败: {e}')
else:
    print('   ⚠️  RETFound 路径不存在，跳过测试')

# 测试 MedSAM
medsam_path = os.path.join('..', 'MedSAM-main')
if os.path.exists(medsam_path):
    sys.path.insert(0, medsam_path)
    try:
        from segment_anything import build_sam_vit_b
        print('   ✓ MedSAM 导入成功')
    except ImportError as e:
        print(f'   ✗ MedSAM 导入失败: {e}')
else:
    print('   ⚠️  MedSAM 路径不存在，跳过测试')
" || echo "   导入测试失败（可能外部项目未克隆）"

echo ""
echo "=========================================="
echo -e "${GREEN}环境安装完成！${NC}"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 确保外部项目已克隆（RETFound-main 和 MedSAM-main）"
echo "2. 准备数据（参考 docs/QUICKSTART.md）"
echo "3. 运行小样本测试"
echo "4. 开始正式训练"
echo ""
echo "如遇问题，请参考:"
echo "- docs/ENVIRONMENT_SETUP.md - 环境安装详细说明"
echo "- docs/TROUBLESHOOTING.md - 问题排查指南"


