"""
调试工具
用于诊断和修复常见问题
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import traceback


def check_tensor_shapes(model: nn.Module, x_R: torch.Tensor, x_C: torch.Tensor) -> Dict[str, Tuple]:
    """
    检查模型各层的张量形状
    
    Args:
        model: DeepCAD Stage I 模型
        x_R: 视网膜图像
        x_C: MRI切片
    
    Returns:
        包含各层输出形状的字典
    """
    shapes = {}
    
    try:
        # 检查输入
        shapes['input_retinal'] = x_R.shape
        shapes['input_mri'] = x_C.shape
        
        # 编码器输出
        with torch.no_grad():
            h_R = model.retinal_encoder(x_R)
            h_C = model.mri_encoder(x_C)
            shapes['h_R'] = h_R.shape
            shapes['h_C'] = h_C.shape
            
            # 投影输出
            z_R = model.projection_R(h_R)
            z_C = model.projection_C(h_C)
            shapes['z_R'] = z_R.shape
            shapes['z_C'] = z_C.shape
        
        return shapes
    except Exception as e:
        return {'error': str(e), 'traceback': traceback.format_exc()}


def check_normalization(z_R: torch.Tensor, z_C: torch.Tensor, tol: float = 1e-5) -> Dict[str, bool]:
    """
    检查投影嵌入是否已正确归一化
    
    Args:
        z_R: 视网膜投影
        z_C: MRI投影
        tol: 容差
    
    Returns:
        检查结果字典
    """
    results = {}
    
    # 检查 L2 范数
    z_R_norm = torch.norm(z_R, p=2, dim=1)
    z_C_norm = torch.norm(z_C, p=2, dim=1)
    
    results['z_R_normalized'] = torch.allclose(z_R_norm, torch.ones_like(z_R_norm), atol=tol)
    results['z_C_normalized'] = torch.allclose(z_C_norm, torch.ones_like(z_C_norm), atol=tol)
    
    results['z_R_norm_mean'] = z_R_norm.mean().item()
    results['z_R_norm_std'] = z_R_norm.std().item()
    results['z_C_norm_mean'] = z_C_norm.mean().item()
    results['z_C_norm_std'] = z_C_norm.std().item()
    
    return results


def check_loss_computation(
    z_R: torch.Tensor,
    z_C: torch.Tensor,
    labels: torch.Tensor,
    tau: float = 0.1
) -> Dict[str, any]:
    """
    检查损失计算的正确性
    
    Args:
        z_R: 视网膜投影
        z_C: MRI投影
        labels: 标签
        tau: 温度参数
    
    Returns:
        检查结果字典
    """
    results = {}
    
    # 检查输入形状
    results['z_R_shape'] = list(z_R.shape)
    results['z_C_shape'] = list(z_C.shape)
    results['labels_shape'] = list(labels.shape)
    results['batch_size_match'] = (z_R.shape[0] == z_C.shape[0] == labels.shape[0])
    
    # 检查潜在空间维度匹配
    results['latent_dim_match'] = (z_R.shape[1] == z_C.shape[1])
    
    # 检查归一化
    norm_check = check_normalization(z_R, z_C)
    results['normalization'] = norm_check
    
    # 计算余弦相似度矩阵
    cosine_sim = torch.matmul(z_R, z_C.T)
    results['cosine_sim_shape'] = list(cosine_sim.shape)
    results['cosine_sim_range'] = [cosine_sim.min().item(), cosine_sim.max().item()]
    
    # 检查正样本掩码
    labels_expanded = labels.unsqueeze(0)
    positive_mask = (labels_expanded == labels_expanded.T).float()
    results['positive_mask_shape'] = list(positive_mask.shape)
    results['positive_pairs_per_sample'] = positive_mask.sum(dim=1).cpu().numpy().tolist()
    
    # 检查是否有样本没有正样本对
    num_positives = positive_mask.sum(dim=1)
    results['samples_without_positives'] = (num_positives == 1).sum().item()  # 只有自己
    
    return results


def diagnose_training_issue(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: str = "cuda"
) -> Dict[str, any]:
    """
    诊断训练问题
    
    Args:
        model: 模型
        train_loader: 训练数据加载器
        criterion: 损失函数
        device: 设备
    
    Returns:
        诊断结果字典
    """
    diagnosis = {
        'errors': [],
        'warnings': [],
        'info': []
    }
    
    model.eval()
    model.to(device)
    
    try:
        # 获取一个批次
        batch = next(iter(train_loader))
        x_R = batch['x_R'].to(device)
        x_C = batch['x_C'].to(device)
        labels = batch['y'].to(device)
        
        # 检查数据加载
        diagnosis['info'].append(f"批次大小: {x_R.shape[0]}")
        diagnosis['info'].append(f"视网膜图像形状: {x_R.shape}")
        diagnosis['info'].append(f"MRI形状: {x_C.shape}")
        diagnosis['info'].append(f"标签形状: {labels.shape}")
        
        # 检查张量形状
        shapes = check_tensor_shapes(model, x_R, x_C)
        if 'error' in shapes:
            diagnosis['errors'].append(f"张量形状检查失败: {shapes['error']}")
        else:
            diagnosis['info'].append(f"张量形状: {shapes}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(x_R, x_C)
            z_R = outputs['z_R']
            z_C = outputs['z_C']
        
        # 检查归一化
        norm_check = check_normalization(z_R, z_C)
        if not norm_check['z_R_normalized']:
            diagnosis['warnings'].append("z_R 未正确归一化")
        if not norm_check['z_C_normalized']:
            diagnosis['warnings'].append("z_C 未正确归一化")
        
        # 检查损失计算
        loss_check = check_loss_computation(z_R, z_C, labels)
        if not loss_check['batch_size_match']:
            diagnosis['errors'].append("批次大小不匹配")
        if not loss_check['latent_dim_match']:
            diagnosis['errors'].append("潜在空间维度不匹配")
        if loss_check['samples_without_positives'] > 0:
            diagnosis['warnings'].append(
                f"{loss_check['samples_without_positives']} 个样本在批次中没有正样本对"
            )
        
        # 计算损失
        try:
            L, L_C, L_R = criterion(z_R, z_C, labels)
            diagnosis['info'].append(f"损失值: L={L.item():.4f}, L_C={L_C.item():.4f}, L_R={L_R.item():.4f}")
            
            if torch.isnan(L) or torch.isinf(L):
                diagnosis['errors'].append("损失值为 NaN 或 Inf")
            if L.item() < 0:
                diagnosis['warnings'].append("损失值为负数（可能正常，取决于实现）")
        except Exception as e:
            diagnosis['errors'].append(f"损失计算失败: {str(e)}")
        
        # 检查梯度
        model.train()
        L, _, _ = criterion(z_R, z_C, labels)
        L.backward()
        
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms[name] = grad_norm
                if grad_norm == 0:
                    diagnosis['warnings'].append(f"{name} 的梯度为零")
                if np.isnan(grad_norm) or np.isinf(grad_norm):
                    diagnosis['errors'].append(f"{name} 的梯度为 NaN 或 Inf")
            else:
                diagnosis['warnings'].append(f"{name} 没有梯度")
        
        diagnosis['info'].append(f"梯度范数: {grad_norms}")
        
    except Exception as e:
        diagnosis['errors'].append(f"诊断过程出错: {str(e)}")
        diagnosis['errors'].append(f"堆栈跟踪: {traceback.format_exc()}")
    
    return diagnosis


def print_diagnosis(diagnosis: Dict[str, List]):
    """打印诊断结果"""
    print("=" * 60)
    print("训练问题诊断结果")
    print("=" * 60)
    
    if diagnosis['errors']:
        print("\n❌ 错误:")
        for error in diagnosis['errors']:
            print(f"  - {error}")
    
    if diagnosis['warnings']:
        print("\n⚠️  警告:")
        for warning in diagnosis['warnings']:
            print(f"  - {warning}")
    
    if diagnosis['info']:
        print("\nℹ️  信息:")
        for info in diagnosis['info']:
            print(f"  - {info}")
    
    print("=" * 60)


def validate_model_architecture(model: nn.Module) -> Dict[str, bool]:
    """
    验证模型架构的正确性
    
    Args:
        model: DeepCAD Stage I 模型
    
    Returns:
        验证结果字典
    """
    results = {}
    
    # 检查必需的组件
    results['has_retinal_encoder'] = hasattr(model, 'retinal_encoder')
    results['has_mri_encoder'] = hasattr(model, 'mri_encoder')
    results['has_projection_R'] = hasattr(model, 'projection_R')
    results['has_projection_C'] = hasattr(model, 'projection_C')
    
    # 检查编码器输出维度
    if results['has_retinal_encoder']:
        try:
            retinal_dim = model.retinal_encoder.get_embed_dim()
            results['retinal_embed_dim'] = retinal_dim
        except:
            results['retinal_embed_dim'] = None
    
    if results['has_mri_encoder']:
        try:
            mri_dim = model.mri_encoder.get_embed_dim()
            results['mri_embed_dim'] = mri_dim
        except:
            results['mri_embed_dim'] = None
    
    # 检查投影头输入维度匹配
    if results['has_projection_R'] and results['has_retinal_encoder']:
        try:
            proj_input_dim = model.projection_R.mlp[0].in_features
            results['projection_R_input_match'] = (proj_input_dim == retinal_dim)
        except:
            results['projection_R_input_match'] = False
    
    if results['has_projection_C'] and results['has_mri_encoder']:
        try:
            proj_input_dim = model.projection_C.mlp[0].in_features
            results['projection_C_input_match'] = (proj_input_dim == mri_dim)
        except:
            results['projection_C_input_match'] = False
    
    # 检查投影头输出维度匹配
    if results['has_projection_R'] and results['has_projection_C']:
        try:
            proj_R_output = model.projection_R.get_output_dim()
            proj_C_output = model.projection_C.get_output_dim()
            results['projection_outputs_match'] = (proj_R_output == proj_C_output)
        except:
            results['projection_outputs_match'] = False
    
    return results

