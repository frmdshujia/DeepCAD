"""
日志记录工具
支持TensorBoard和简单的文本日志
"""

import os
from typing import Optional
from datetime import datetime


class SimpleLogger:
    """
    简单的日志记录器
    可以扩展为支持TensorBoard或WandB
    """
    
    def __init__(self, log_dir: str):
        """
        初始化日志记录器
        
        Args:
            log_dir: 日志目录
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志文件
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"training_{timestamp}.log")
        
        # 尝试导入TensorBoard
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.tensorboard_writer = SummaryWriter(log_dir=os.path.join(log_dir, "tensorboard"))
            self.use_tensorboard = True
        except ImportError:
            self.tensorboard_writer = None
            self.use_tensorboard = False
            print("TensorBoard未安装，将只使用文本日志")
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        记录标量值
        
        Args:
            tag: 标签（如 'train/loss'）
            value: 值
            step: 步数
        """
        # TensorBoard
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.add_scalar(tag, value, step)
        
        # 文本日志
        with open(self.log_file, 'a') as f:
            f.write(f"Step {step}: {tag} = {value:.6f}\n")
    
    def log_text(self, text: str):
        """
        记录文本
        
        Args:
            text: 文本内容
        """
        with open(self.log_file, 'a') as f:
            f.write(f"{text}\n")
    
    def close(self):
        """关闭日志记录器"""
        if self.use_tensorboard and self.tensorboard_writer:
            self.tensorboard_writer.close()


def setup_logger(log_dir: str) -> SimpleLogger:
    """
    设置日志记录器
    
    Args:
        log_dir: 日志目录
    
    Returns:
        日志记录器实例
    """
    return SimpleLogger(log_dir)

