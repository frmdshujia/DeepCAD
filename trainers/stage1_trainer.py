"""
DeepCAD Stage I 训练器
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from typing import Dict, Optional
try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的进度条
    def tqdm(iterable, desc=""):
        return iterable

from models.deepcad_stage1 import DeepCADStageI
from losses import CrossModalContrastiveLoss
from utils.checkpoint import save_checkpoint, load_checkpoint
from utils.logger import setup_logger
from utils import FeatureQueue


class Stage1Trainer:
    """
    DeepCAD Stage I 训练器
    """
    
    def __init__(
        self,
        model: DeepCADStageI,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        criterion: Optional[CrossModalContrastiveLoss] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        device: str = "cuda",
        log_dir: str = "logs",
        save_dir: str = "checkpoints/stage1",
        log_interval: int = 10,
        save_interval: int = 5,
        training_mode: str = "grade",
        use_amp: bool = True,
        grad_accum_steps: int = 1,
        use_queue: bool = False,
        queue_size: int = 16384,
    ):
        """
        初始化训练器
        
        Args:
            model: DeepCAD Stage I 模型
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器（可选）
            criterion: 损失函数（如果为None，则使用默认的CrossModalContrastiveLoss）
            optimizer: 优化器（如果为None，需要后续设置）
            scheduler: 学习率调度器（可选）
            device: 设备（"cuda" 或 "cpu"）
            log_dir: 日志目录
            save_dir: 检查点保存目录
            log_interval: 日志记录间隔（每N个批次）
            save_interval: 检查点保存间隔（每N个epoch）
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # 损失函数
        if criterion is None:
            self.criterion = CrossModalContrastiveLoss(tau=0.1)
        else:
            self.criterion = criterion
        
        self.optimizer = optimizer
        self.scheduler = scheduler
        
        # 日志和保存
        self.log_dir = log_dir
        self.save_dir = save_dir
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        self.logger = setup_logger(log_dir)
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.training_mode = training_mode
        self.use_amp = use_amp and device.startswith("cuda") and torch.cuda.is_available()
        # 梯度累积步数（至少为 1）
        self.grad_accum_steps = max(1, int(grad_accum_steps))
        # 统一的正样本 key 编码映射：
        # - 对于 subject 模式：根据 subject_id 动态分配一个全局整数 key
        # - 对于 label/grade/mixed 模式：直接使用标签作为 key
        self._pos_key_map = {}   # 仅在 training_mode == "subject" 时使用
        self._next_pos_key = 0
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        # 混合精度缩放器
        self.scaler = GradScaler(enabled=self.use_amp)

        # memory bank / 队列负样本
        self.use_queue = use_queue
        self.queue_size = int(queue_size)
        if self.use_queue:
            latent_dim = model.get_latent_dim()
            self.queue_R = FeatureQueue(dim=latent_dim, K=self.queue_size, device=device)
            self.queue_C = FeatureQueue(dim=latent_dim, K=self.queue_size, device=device)
        else:
            self.queue_R = None
            self.queue_C = None
    
    def _build_pos_keys(self, labels: torch.Tensor, subject_ids) -> torch.Tensor:
        """
        根据当前训练模式构造统一的“正样本 key”向量。
        
        - subject 模式：key = subject_id（通过全局字典编码为 int，保证同一 subject 始终 key 相同）
        - 其它模式 (grade / mixed / mixed_clinical)：key = labels
        
        返回:
            pos_keys: (B,) long tensor，放在 self.device 上
        """
        if self.training_mode == "subject":
            # 对任意类型的 subject_id（str/int 等）进行全局编码
            keys = []
            for sid in subject_ids:
                sid_str = str(sid)
                if sid_str not in self._pos_key_map:
                    self._pos_key_map[sid_str] = self._next_pos_key
                    self._next_pos_key += 1
                keys.append(self._pos_key_map[sid_str])
            pos_keys = torch.tensor(keys, dtype=torch.long, device=self.device)
        else:
            # 非 subject 模式：直接使用标签作为 key
            pos_keys = labels.to(self.device).long()
        return pos_keys
        
    def train_one_epoch(self) -> Dict[str, float]:
        """
        训练一个epoch
        
        Returns:
            包含训练指标的字典
        """
        self.model.train()
        
        total_loss = 0.0
        total_loss_C = 0.0
        total_loss_R = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        accum_steps = self.grad_accum_steps

        if self.optimizer is None:
            raise ValueError("Optimizer 未设置，无法训练。")

        # 初始化梯度
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(pbar):
            # 获取数据
            x_R = batch['x_R'].to(self.device)  # (B, 3, H, W)
            x_C = batch['x_C'].to(self.device)  # (B, num_slices, 1, H, W)
            labels = batch['y'].to(self.device)  # (B,)
            subject_ids = batch['subject_id']
            # 构造统一的正样本 key（无论 subject / grade / mixed 模式）
            pos_keys = self._build_pos_keys(labels, subject_ids)  # (B,)
            
            # memory bank: 从队列中取额外负样本（如果启用）
            if self.use_queue and self.queue_R is not None and self.queue_C is not None:
                queue_R_feats, queue_R_keys = self.queue_R.get()
                queue_C_feats, queue_C_keys = self.queue_C.get()
            else:
                queue_R_feats = queue_R_keys = None
                queue_C_feats = queue_C_keys = None

            # 前向传播 + 计算损失（正样本策略在 CrossModalContrastiveLoss 内部根据 training_mode 决定）
            if self.use_amp:
                # 兼容较老版本的 PyTorch：autocast 不支持 device_type 参数
                with autocast():
                    outputs = self.model(x_R, x_C)
                    z_R = outputs['z_R']
                    z_C = outputs['z_C']
                    L, L_C, L_R = self.criterion(
                        z_R, z_C,
                        labels,
                        subject_ids=subject_ids,
                        # 统一的正样本 key（当前 batch）
                        pos_keys=pos_keys,
                        # 队列中的历史特征和 key（如果启用）
                        queue_R_feats=queue_R_feats,
                        queue_R_keys=queue_R_keys,
                        queue_C_feats=queue_C_feats,
                        queue_C_keys=queue_C_keys,
                    )
                # 为了梯度累积，缩放 loss
                loss_for_backward = L / accum_steps
                self.scaler.scale(loss_for_backward).backward()

                # 每 accum_steps 个小批次执行一次参数更新
                if ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(self.train_loader)):
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                outputs = self.model(x_R, x_C)
                z_R = outputs['z_R']
                z_C = outputs['z_C']
                L, L_C, L_R = self.criterion(
                    z_R, z_C,
                    labels,
                    subject_ids=subject_ids,
                    pos_keys=pos_keys,
                    queue_R_feats=queue_R_feats,
                    queue_R_keys=queue_R_keys,
                    queue_C_feats=queue_C_feats,
                    queue_C_keys=queue_C_keys,
                )
                loss_for_backward = L / accum_steps
                loss_for_backward.backward()

                if ((batch_idx + 1) % accum_steps == 0) or (batch_idx + 1 == len(self.train_loader)):
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # 反向传播和参数更新之后，再用当前 batch 的特征 + key 更新队列（不参与本次梯度）
            if self.use_queue and self.queue_R is not None and self.queue_C is not None:
                with torch.no_grad():
                    # 这里使用统一的 pos_keys 作为两侧模态的 key，保证跨模态共享同一“语义身份”
                    self.queue_R.enqueue(z_R.detach(), pos_keys.detach())
                    self.queue_C.enqueue(z_C.detach(), pos_keys.detach())
            
            # 累计统计
            total_loss += L.item()
            total_loss_C += L_C.item()
            total_loss_R += L_R.item()
            num_batches += 1
            self.global_step += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{L.item():.4f}',
                'L_C': f'{L_C.item():.4f}',
                'L_R': f'{L_R.item():.4f}'
            })
            
            # 记录日志
            if batch_idx % self.log_interval == 0:
                self.logger.log_scalar('train/loss', L.item(), self.global_step)
                self.logger.log_scalar('train/loss_C', L_C.item(), self.global_step)
                self.logger.log_scalar('train/loss_R', L_R.item(), self.global_step)
                if self.optimizer is not None:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.logger.log_scalar('train/lr', current_lr, self.global_step)
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_loss_C = total_loss_C / num_batches
        avg_loss_R = total_loss_R / num_batches
        
        return {
            'loss': avg_loss,
            'loss_C': avg_loss_C,
            'loss_R': avg_loss_R
        }
    
    @torch.no_grad()
    def validate_one_epoch(self) -> Dict[str, float]:
        """
        验证一个epoch
        
        Returns:
            包含验证指标的字典
        """
        self.model.eval()
        
        total_loss = 0.0
        total_loss_C = 0.0
        total_loss_R = 0.0
        num_batches = 0
        
        if self.val_loader is None:
            return {}
        
        pbar = tqdm(self.val_loader, desc="Validation")
        
        for batch in pbar:
            # 获取数据
            x_R = batch['x_R'].to(self.device)
            x_C = batch['x_C'].to(self.device)
            labels = batch['y'].to(self.device)
            subject_ids = batch['subject_id']
            pos_keys = self._build_pos_keys(labels, subject_ids)

            # 验证时同样可以使用队列扩展负样本，但不更新队列
            if self.use_queue and self.queue_R is not None and self.queue_C is not None:
                queue_R_feats, queue_R_keys = self.queue_R.get()
                queue_C_feats, queue_C_keys = self.queue_C.get()
            else:
                queue_R_feats = queue_R_keys = None
                queue_C_feats = queue_C_keys = None

            # 前向传播
            outputs = self.model(x_R, x_C)
            z_R = outputs['z_R']
            z_C = outputs['z_C']
            
            # 计算损失（正样本策略在 CrossModalContrastiveLoss 内部根据 training_mode 决定）
            L, L_C, L_R = self.criterion(
                z_R, z_C,
                labels,
                subject_ids=subject_ids,
                pos_keys=pos_keys,
                queue_R_feats=queue_R_feats,
                queue_R_keys=queue_R_keys,
                queue_C_feats=queue_C_feats,
                queue_C_keys=queue_C_keys,
            )
            
            # 累计统计
            total_loss += L.item()
            total_loss_C += L_C.item()
            total_loss_R += L_R.item()
            num_batches += 1
            
            # 更新进度条
            pbar.set_postfix({
                'loss': f'{L.item():.4f}',
                'L_C': f'{L_C.item():.4f}',
                'L_R': f'{L_R.item():.4f}'
            })
        
        # 计算平均损失
        avg_loss = total_loss / num_batches
        avg_loss_C = total_loss_C / num_batches
        avg_loss_R = total_loss_R / num_batches
        
        # 记录日志
        self.logger.log_scalar('val/loss', avg_loss, self.current_epoch)
        self.logger.log_scalar('val/loss_C', avg_loss_C, self.current_epoch)
        self.logger.log_scalar('val/loss_R', avg_loss_R, self.current_epoch)
        
        return {
            'loss': avg_loss,
            'loss_C': avg_loss_C,
            'loss_R': avg_loss_R
        }
    
    def train(self, num_epochs: int, resume_from: Optional[str] = None):
        """
        训练模型
        
        Args:
            num_epochs: 训练轮数
            resume_from: 恢复训练的检查点路径（可选）
        """
        # 恢复训练
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        print(f"开始训练，共 {num_epochs} 个epoch")
        print(f"设备: {self.device}")
        print(f"训练集大小: {len(self.train_loader.dataset)}")
        if self.val_loader is not None:
            print(f"验证集大小: {len(self.val_loader.dataset)}")
        print("-" * 50)
        
        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            
            # 训练
            train_metrics = self.train_one_epoch()
            
            # 验证
            val_metrics = {}
            if self.val_loader is not None:
                val_metrics = self.validate_one_epoch()
            
            # 打印epoch总结
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  训练损失: {train_metrics['loss']:.4f} "
                  f"(L_C: {train_metrics['loss_C']:.4f}, "
                  f"L_R: {train_metrics['loss_R']:.4f})")
            if val_metrics:
                print(f"  验证损失: {val_metrics['loss']:.4f} "
                      f"(L_C: {val_metrics['loss_C']:.4f}, "
                      f"L_R: {val_metrics['loss_R']:.4f})")
            
            # 更新学习率调度器（按epoch）
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    metric = val_metrics.get('loss') if val_metrics else train_metrics['loss']
                    self.scheduler.step(metric)
                else:
                    self.scheduler.step()
            
            # 保存检查点
            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # 保存最佳模型
            if val_metrics and val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, is_best=True)
                print(f"  ✓ 保存最佳模型 (验证损失: {self.best_val_loss:.4f})")
            
            print("-" * 50)
        
        print("\n训练完成！")
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """
        保存检查点
        
        Args:
            epoch: 当前epoch
            is_best: 是否为最佳模型
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'global_step': self.global_step
        }
        
        # 保存常规检查点
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch+1}.pth')
        save_checkpoint(checkpoint, checkpoint_path)
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.save_dir, 'best_model.pth')
            save_checkpoint(checkpoint, best_path)
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest_checkpoint.pth')
        save_checkpoint(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        加载检查点
        
        Args:
            checkpoint_path: 检查点路径
        """
        checkpoint = load_checkpoint(checkpoint_path)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.optimizer and checkpoint['optimizer_state_dict']:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch'] + 1
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        
        print(f"从检查点恢复训练: {checkpoint_path}")
        print(f"  Epoch: {self.current_epoch}")
        print(f"  最佳验证损失: {self.best_val_loss:.4f}")

