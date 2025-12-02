"""
Grad-CAM 实现（适配 Vision Transformer）
用于可视化视网膜编码器的注意力区域
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import cv2


class VitGradCAM:
    """
    Vision Transformer 的 Grad-CAM 实现
    
    用于可视化 ViT 编码器中哪些 patch 对最终输出贡献最大
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
        use_cuda: bool = True
    ):
        """
        初始化 Grad-CAM
        
        Args:
            model: 模型（应该是 DeepCADStageI 或 RetinalEncoder）
            target_layer: 目标层（如果为None，则自动选择最后一个注意力块）
            use_cuda: 是否使用CUDA
        """
        self.model = model
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 存储梯度和激活
        self.gradients = []
        self.activations = []
        
        # 注册钩子
        self.target_layer = self._find_target_layer(target_layer)
        self._register_hooks()
    
    def _find_target_layer(self, target_layer: Optional[nn.Module]) -> nn.Module:
        """
        查找目标层
        
        对于 ViT，通常选择最后一个注意力块或最后一个 block
        """
        if target_layer is not None:
            return target_layer
        
        # 尝试找到视网膜编码器的最后一个 block
        if hasattr(self.model, 'retinal_encoder'):
            encoder = self.model.retinal_encoder
            if hasattr(encoder, 'backbone'):
                backbone = encoder.backbone
                if hasattr(backbone, 'blocks'):
                    # 返回最后一个 block
                    return backbone.blocks[-1]
                elif hasattr(backbone, 'layers'):
                    return backbone.layers[-1]
        
        # 如果找不到，返回模型本身
        return self.model
    
    def _register_hooks(self):
        """注册前向和反向钩子"""
        def backward_hook(module, grad_input, grad_output):
            self.gradients.append(grad_output[0])
        
        def forward_hook(module, input, output):
            self.activations.append(output)
        
        self.target_layer.register_full_backward_hook(backward_hook)
        self.target_layer.register_forward_hook(forward_hook)
    
    def _generate_gradcam(
        self,
        activations: torch.Tensor,
        gradients: torch.Tensor
    ) -> np.ndarray:
        """
        生成 Grad-CAM 热图
        
        Args:
            activations: 激活值，形状为 (B, N, D) 或 (B, D, H, W)
            gradients: 梯度值，形状与 activations 相同
        
        Returns:
            Grad-CAM 热图，形状为 (H, W)
        """
        # 对于 ViT，activations 通常是 (B, N, D)，其中 N 是 patch 数量
        if len(activations.shape) == 3:
            # (B, N, D) -> 对通道维度求平均梯度
            weights = torch.mean(gradients, dim=2, keepdim=True)  # (B, N, 1)
            cam = torch.sum(weights * activations, dim=2)  # (B, N)
            cam = F.relu(cam)  # ReLU 激活
            # 归一化
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            # 取第一个样本
            cam = cam[0].cpu().numpy()
        elif len(activations.shape) == 4:
            # (B, D, H, W) -> 标准卷积层格式
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # (B, D, 1, 1)
            cam = torch.sum(weights * activations, dim=1)  # (B, H, W)
            cam = F.relu(cam)
            cam = cam - cam.min()
            cam = cam / (cam.max() + 1e-8)
            cam = cam[0].cpu().numpy()
        else:
            raise ValueError(f"不支持的激活形状: {activations.shape}")
        
        return cam
    
    def generate_cam(
        self,
        input_image: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        target_type: str = "similarity"
    ) -> np.ndarray:
        """
        生成 Grad-CAM 热图
        
        Args:
            input_image: 输入图像，形状为 (1, 3, H, W)
            target: 目标值（用于计算梯度）
                  - 如果为None，则使用模型输出
                  - 如果 target_type="similarity"，应该是与另一个模态的相似度
                  - 如果 target_type="embedding"，应该是嵌入向量
            target_type: 目标类型 ("similarity", "embedding", "auto")
        
        Returns:
            Grad-CAM 热图，形状为 (H, W) 或 (patch_h, patch_w)
        """
        # 清空之前的激活和梯度
        self.activations = []
        self.gradients = []
        
        # 移动到设备
        input_image = input_image.to(self.device)
        if input_image.dim() == 3:
            input_image = input_image.unsqueeze(0)
        
        # 前向传播
        if hasattr(self.model, 'retinal_encoder'):
            # DeepCADStageI 模型
            if target is None:
                # 使用模型输出作为目标
                outputs = self.model.retinal_encoder(input_image)
                if target_type == "similarity":
                    # 需要另一个模态的嵌入来计算相似度
                    # 这里使用嵌入的 L2 范数作为目标
                    target = torch.norm(outputs, dim=1).mean()
                else:
                    target = outputs.mean()
            else:
                outputs = self.model.retinal_encoder(input_image)
        else:
            # 直接是编码器
            outputs = self.model(input_image)
            if target is None:
                target = outputs.mean()
        
        # 反向传播
        self.model.zero_grad()
        if isinstance(target, torch.Tensor):
            target.backward()
        else:
            outputs.mean().backward()
        
        # 获取激活和梯度
        if len(self.activations) == 0 or len(self.gradients) == 0:
            raise RuntimeError("未能捕获激活或梯度，请检查钩子注册")
        
        activations = self.activations[0]
        gradients = self.gradients[0]
        
        # 生成 Grad-CAM
        cam = self._generate_gradcam(activations, gradients)
        
        return cam
    
    def generate_cam_for_similarity(
        self,
        retinal_image: torch.Tensor,
        cardiac_embedding: torch.Tensor
    ) -> np.ndarray:
        """
        生成基于跨模态相似度的 Grad-CAM
        
        Args:
            retinal_image: 视网膜图像，形状为 (1, 3, H, W)
            cardiac_embedding: 心脏MRI嵌入，形状为 (1, D)
        
        Returns:
            Grad-CAM 热图
        """
        # 获取视网膜嵌入
        if hasattr(self.model, 'retinal_encoder'):
            retinal_embedding = self.model.retinal_encoder(retinal_image.to(self.device))
        else:
            retinal_embedding = self.model(retinal_image.to(self.device))
        
        # 投影到共享空间
        if hasattr(self.model, 'projection_R'):
            z_R = self.model.projection_R(retinal_embedding)
        else:
            z_R = F.normalize(retinal_embedding, p=2, dim=1)
        
        # 计算相似度
        cardiac_embedding = cardiac_embedding.to(self.device)
        if hasattr(self.model, 'projection_C'):
            z_C = self.model.projection_C(cardiac_embedding)
        else:
            z_C = F.normalize(cardiac_embedding, p=2, dim=1)
        
        similarity = torch.sum(z_R * z_C, dim=1)  # (1,)
        
        # 使用相似度作为目标生成 Grad-CAM
        return self.generate_cam(retinal_image, similarity, target_type="similarity")
    
    def overlay_heatmap(
        self,
        image: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.4,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        将热图叠加到原始图像上
        
        Args:
            image: 原始图像，形状为 (H, W, 3) 或 (H, W)，值范围 [0, 255]
            heatmap: 热图，形状为 (H, W)，值范围 [0, 1]
            alpha: 叠加透明度
            colormap: OpenCV 颜色映射
        
        Returns:
            叠加后的图像，形状为 (H, W, 3)
        """
        # 确保图像是 RGB 格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 1:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # 调整热图大小以匹配图像
        if heatmap.shape != image.shape[:2]:
            heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        
        # 应用颜色映射
        heatmap_colored = cv2.applyColorMap(
            (heatmap * 255).astype(np.uint8),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # 叠加
        overlay = (alpha * heatmap_colored + (1 - alpha) * image).astype(np.uint8)
        
        return overlay
    
    def visualize_patch_attention(
        self,
        image: torch.Tensor,
        cam: np.ndarray,
        patch_size: int = 16,
        img_size: int = 224
    ) -> np.ndarray:
        """
        将 patch 级别的注意力映射回原始图像空间
        
        Args:
            image: 原始图像，形状为 (1, 3, H, W)
            cam: patch 级别的 Grad-CAM，形状为 (num_patches,)
            patch_size: patch 大小
            img_size: 图像尺寸
        
        Returns:
            映射到图像空间的热图，形状为 (H, W)
        """
        # 计算 patch 网格大小
        num_patches_per_side = img_size // patch_size
        num_patches = num_patches_per_side * num_patches_per_side
        
        # 重塑为 2D 网格
        if len(cam) == num_patches:
            cam_2d = cam.reshape(num_patches_per_side, num_patches_per_side)
        else:
            # 如果 patch 数量不匹配，尝试调整
            cam_2d = cam[:num_patches].reshape(num_patches_per_side, num_patches_per_side)
        
        # 上采样到原始图像尺寸
        if isinstance(image, torch.Tensor):
            h, w = image.shape[-2:]
        else:
            h, w = image.shape[:2]
        
        cam_resized = cv2.resize(
            cam_2d,
            (w, h),
            interpolation=cv2.INTER_LINEAR
        )
        
        return cam_resized

