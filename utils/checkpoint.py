"""
检查点保存和加载工具
"""

import torch
import os
from typing import Dict, Optional


def save_checkpoint(checkpoint: Dict, filepath: str):
    """
    保存检查点
    
    Args:
        checkpoint: 检查点字典
        filepath: 保存路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint, filepath)
    print(f"检查点已保存: {filepath}")


def load_checkpoint(filepath: str, map_location: Optional[str] = None) -> Dict:
    """
    加载检查点
    
    Args:
        filepath: 检查点路径
        map_location: 设备映射（如 'cpu' 或 'cuda:0'）
    
    Returns:
        检查点字典
    """
    if map_location is None:
        map_location = 'cpu' if not torch.cuda.is_available() else 'cuda'
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"检查点文件不存在: {filepath}")
    
    checkpoint = torch.load(filepath, map_location=map_location)
    print(f"检查点已加载: {filepath}")
    return checkpoint

