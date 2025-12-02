#!/usr/bin/env python3
"""
验证项目设置
检查所有依赖和配置是否正确
"""

import sys
import os
import importlib

def check_imports():
    """检查必需的 Python 包"""
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'matplotlib': 'Matplotlib',
        'yaml': 'PyYAML',
        'nibabel': 'NiBabel (用于 NIfTI 文件)',
        'timm': 'timm (用于 Vision Transformer)',
    }
    
    optional_packages = {
        'sklearn': 'scikit-learn (用于 PCA 可视化)',
        'tensorboard': 'TensorBoard (用于日志)',
    }
    
    print("检查必需的 Python 包...")
    all_ok = True
    
    for module_name, display_name in required_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ❌ {display_name} 未安装")
            all_ok = False
    
    print("\n检查可选的 Python 包...")
    for module_name, display_name in optional_packages.items():
        try:
            importlib.import_module(module_name)
            print(f"  ✓ {display_name}")
        except ImportError:
            print(f"  ⚠️  {display_name} 未安装（可选）")
    
    return all_ok


def check_project_structure():
    """检查项目结构"""
    print("\n检查项目结构...")
    
    required_dirs = [
        'datasets',
        'models',
        'models/encoders',
        'losses',
        'trainers',
        'utils',
        'explainability',
        'scripts',
        'configs',
    ]
    
    all_ok = True
    for dir_path in required_dirs:
        if os.path.exists(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ❌ {dir_path}/ 不存在")
            all_ok = False
    
    return all_ok


def check_external_dependencies():
    """检查外部依赖（RETFound, MedSAM）"""
    print("\n检查外部依赖...")
    
    # 检查 RETFound
    retfound_path = os.path.join('..', 'RETFound-main')
    if os.path.exists(retfound_path):
        print(f"  ✓ RETFound 找到: {retfound_path}")
    else:
        print(f"  ⚠️  RETFound 未找到: {retfound_path}")
        print("     提示: RETFound-main 应该在项目父目录中")
    
    # 检查 MedSAM
    medsam_path = os.path.join('..', 'MedSAM-main')
    if os.path.exists(medsam_path):
        print(f"  ✓ MedSAM 找到: {medsam_path}")
    else:
        print(f"  ⚠️  MedSAM 未找到: {medsam_path}")
        print("     提示: MedSAM-main 应该在项目父目录中")


def check_pytorch():
    """检查 PyTorch 配置"""
    print("\n检查 PyTorch 配置...")
    
    try:
        import torch
        print(f"  PyTorch 版本: {torch.__version__}")
        print(f"  CUDA 可用: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  GPU 数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"    GPU {i}: {torch.cuda.get_device_name(i)}")
    except ImportError:
        print("  ❌ PyTorch 未安装")


def main():
    """主函数"""
    print("=" * 60)
    print("DeepCAD 项目设置验证")
    print("=" * 60)
    
    # 切换到项目根目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    os.chdir(project_root)
    
    # 运行检查
    imports_ok = check_imports()
    structure_ok = check_project_structure()
    check_external_dependencies()
    check_pytorch()
    
    # 总结
    print("\n" + "=" * 60)
    if imports_ok and structure_ok:
        print("✓ 基本设置验证通过")
    else:
        print("❌ 发现问题，请根据上述提示修复")
    print("=" * 60)


if __name__ == "__main__":
    main()

