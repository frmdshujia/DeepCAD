#!/bin/bash
# DeepCAD 环境安装脚本

echo "=========================================="
echo "DeepCAD 环境安装"
echo "=========================================="

# 检查 Python
echo "1. 检查 Python 版本..."
python_version=$(python --version 2>&1 | awk '{print $2}')
echo "   Python 版本: $python_version"

# 创建虚拟环境（可选）
read -p "是否创建新的虚拟环境? (y/n): " create_venv
if [ "$create_venv" = "y" ]; then
    read -p "虚拟环境名称 (默认: deepcad): " venv_name
    venv_name=${venv_name:-deepcad}
    
    # 检查是否使用 conda
    if command -v conda &> /dev/null; then
        echo "2. 使用 conda 创建环境..."
        conda create -n $venv_name python=3.8 -y
        echo "   激活环境: conda activate $venv_name"
    else
        echo "2. 使用 venv 创建环境..."
        python -m venv $venv_name
        echo "   激活环境: source $venv_name/bin/activate"
    fi
fi

# 安装依赖
echo ""
echo "3. 安装 Python 依赖..."
pip install -r requirements.txt

# 安装可选依赖
echo ""
read -p "是否安装可选依赖 (scikit-learn, tensorboard)? (y/n): " install_optional
if [ "$install_optional" = "y" ]; then
    pip install scikit-learn tensorboard
fi

# 检查外部依赖
echo ""
echo "4. 检查外部依赖..."

# 检查 RETFound
if [ -d "../RETFound-main" ]; then
    echo "   ✓ RETFound 找到: ../RETFound-main"
else
    echo "   ⚠️  RETFound 未找到: ../RETFound-main"
    echo "      请克隆 RETFound 仓库到项目父目录"
fi

# 检查 MedSAM
if [ -d "../MedSAM-main" ]; then
    echo "   ✓ MedSAM 找到: ../MedSAM-main"
else
    echo "   ⚠️  MedSAM 未找到: ../MedSAM-main"
    echo "      请克隆 MedSAM 仓库到项目父目录"
fi

# 验证安装
echo ""
echo "5. 验证安装..."
python scripts/validate_setup.py

echo ""
echo "=========================================="
echo "环境安装完成！"
echo "=========================================="
echo ""
echo "下一步:"
echo "1. 准备数据 (参考 docs/QUICKSTART.md)"
echo "2. 运行小样本测试"
echo "3. 开始正式训练"

