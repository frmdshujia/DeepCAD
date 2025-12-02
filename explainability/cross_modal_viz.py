"""
跨模态可视化工具
用于可视化视网膜和心脏MRI之间的关联
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple, List
from PIL import Image

from .grad_cam import VitGradCAM


def visualize_cross_modal(
    model: torch.nn.Module,
    retinal_image: torch.Tensor,
    mri_slices: torch.Tensor,
    retinal_gradcam: Optional[np.ndarray] = None,
    device: str = "cuda",
    save_path: Optional[str] = None
):
    """
    可视化跨模态关联
    
    Args:
        model: DeepCAD Stage I 模型
        retinal_image: 视网膜图像，形状为 (1, 3, H, W)
        mri_slices: MRI切片，形状为 (1, num_slices, 1, H, W)
        retinal_gradcam: 视网膜 Grad-CAM 热图（可选）
        device: 设备
        save_path: 保存路径（可选）
    """
    model.eval()
    model.to(device)
    
    retinal_image = retinal_image.to(device)
    mri_slices = mri_slices.to(device)
    
    # 前向传播
    with torch.no_grad():
        outputs = model(retinal_image, mri_slices)
        z_R = outputs['z_R']
        z_C = outputs['z_C']
        
        # 计算相似度
        similarity = torch.sum(z_R * z_C, dim=1).item()
    
    # 准备可视化
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # 1. 视网膜图像（带或不带 Grad-CAM）
    retinal_np = retinal_image[0].cpu().permute(1, 2, 0).numpy()
    if retinal_np.max() <= 1.0:
        retinal_np = (retinal_np * 255).astype(np.uint8)
    else:
        retinal_np = retinal_np.astype(np.uint8)
    
    axes[0, 0].imshow(retinal_np)
    axes[0, 0].set_title(f'Retinal Fundus\nSimilarity: {similarity:.4f}')
    axes[0, 0].axis('off')
    
    # 2. 视网膜 Grad-CAM（如果有）
    if retinal_gradcam is not None:
        from .grad_cam import VitGradCAM
        gradcam_viz = VitGradCAM(model, use_cuda=(device == "cuda"))
        overlay = gradcam_viz.overlay_heatmap(retinal_np, retinal_gradcam)
        axes[0, 1].imshow(overlay)
        axes[0, 1].set_title('Retinal Grad-CAM')
        axes[0, 1].axis('off')
    else:
        axes[0, 1].axis('off')
    
    # 3. MRI切片（显示代表性切片）
    num_slices = mri_slices.shape[1]
    mid_slice_idx = num_slices // 2
    mri_slice = mri_slices[0, mid_slice_idx, 0].cpu().numpy()
    
    # 归一化到 [0, 255]
    mri_slice = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
    mri_slice = (mri_slice * 255).astype(np.uint8)
    
    axes[1, 0].imshow(mri_slice, cmap='gray')
    axes[1, 0].set_title(f'Cardiac MRI (Slice {mid_slice_idx+1}/{num_slices})')
    axes[1, 0].axis('off')
    
    # 4. 嵌入空间可视化（2D投影）
    z_R_np = z_R[0].cpu().numpy()
    z_C_np = z_C[0].cpu().numpy()
    
    # 使用PCA或简单的2D投影
    try:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        embeddings = np.vstack([z_R_np, z_C_np])
        embeddings_2d = pca.fit_transform(embeddings)
        
        axes[1, 1].scatter(embeddings_2d[0, 0], embeddings_2d[0, 1], 
                          c='red', s=100, label='Retinal', marker='o')
        axes[1, 1].scatter(embeddings_2d[1, 0], embeddings_2d[1, 1], 
                          c='blue', s=100, label='Cardiac MRI', marker='s')
        axes[1, 1].plot([embeddings_2d[0, 0], embeddings_2d[1, 0]],
                       [embeddings_2d[0, 1], embeddings_2d[1, 1]],
                       'k--', alpha=0.5, linewidth=2)
        axes[1, 1].set_title('Embedding Space (2D Projection)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    except ImportError:
        # 如果没有sklearn，使用简单的可视化
        axes[1, 1].text(0.5, 0.5, f'Similarity: {similarity:.4f}\n'
                                  f'z_R norm: {np.linalg.norm(z_R_np):.4f}\n'
                                  f'z_C norm: {np.linalg.norm(z_C_np):.4f}',
                       ha='center', va='center', fontsize=12)
        axes[1, 1].set_title('Embedding Statistics')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"可视化已保存: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_batch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    num_samples: int = 4,
    device: str = "cuda",
    save_dir: str = "visualizations"
):
    """
    批量可视化多个样本
    
    Args:
        model: DeepCAD Stage I 模型
        dataloader: 数据加载器
        num_samples: 要可视化的样本数量
        device: 设备
        save_dir: 保存目录
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    model.eval()
    model.to(device)
    
    gradcam = VitGradCAM(model, use_cuda=(device == "cuda"))
    
    count = 0
    for batch in dataloader:
        if count >= num_samples:
            break
        
        x_R = batch['x_R'].to(device)
        x_C = batch['x_C'].to(device)
        labels = batch['y']
        subject_ids = batch['subject_id']
        
        batch_size = x_R.shape[0]
        
        for i in range(batch_size):
            if count >= num_samples:
                break
            
            # 生成 Grad-CAM
            try:
                cam = gradcam.generate_cam(x_R[i:i+1])
                cam_resized = gradcam.visualize_patch_attention(x_R[i:i+1], cam)
            except Exception as e:
                print(f"生成 Grad-CAM 失败: {e}")
                cam_resized = None
            
            # 可视化
            save_path = os.path.join(save_dir, f"sample_{subject_ids[i]}_label_{labels[i].item()}.png")
            visualize_cross_modal(
                model=model,
                retinal_image=x_R[i:i+1],
                mri_slices=x_C[i:i+1],
                retinal_gradcam=cam_resized,
                device=device,
                save_path=save_path
            )
            
            count += 1

