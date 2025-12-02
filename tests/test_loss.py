"""
测试监督跨模态对比损失函数
"""

import torch
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from losses import cross_modal_contrastive_loss, CrossModalContrastiveLoss


def test_loss_basic():
    """测试基本损失计算"""
    print("测试1: 基本损失计算")
    
    batch_size = 4
    latent_dim = 128
    tau = 0.1
    
    # 创建随机嵌入（已归一化）
    z_R = torch.randn(batch_size, latent_dim)
    z_R = torch.nn.functional.normalize(z_R, p=2, dim=1)
    
    z_C = torch.randn(batch_size, latent_dim)
    z_C = torch.nn.functional.normalize(z_C, p=2, dim=1)
    
    # 创建标签（2个正样本，2个负样本）
    labels = torch.tensor([1, 1, 0, 0])
    
    # 计算损失
    L, L_C, L_R = cross_modal_contrastive_loss(z_R, z_C, labels, tau=tau)
    
    print(f"  总损失 L: {L.item():.4f}")
    print(f"  心脏→视网膜损失 L_C: {L_C.item():.4f}")
    print(f"  视网膜→心脏损失 L_R: {L_R.item():.4f}")
    print(f"  验证: L = L_C + L_R? {torch.allclose(L, L_C + L_R)}")
    assert torch.allclose(L, L_C + L_R), "总损失应该等于L_C + L_R"
    print("  ✓ 通过\n")


def test_loss_module():
    """测试Module版本的损失"""
    print("测试2: Module版本损失")
    
    batch_size = 4
    latent_dim = 128
    
    z_R = torch.randn(batch_size, latent_dim)
    z_R = torch.nn.functional.normalize(z_R, p=2, dim=1)
    
    z_C = torch.randn(batch_size, latent_dim)
    z_C = torch.nn.functional.normalize(z_C, p=2, dim=1)
    
    labels = torch.tensor([1, 1, 0, 0])
    
    criterion = CrossModalContrastiveLoss(tau=0.1)
    L, L_C, L_R = criterion(z_R, z_C, labels)
    
    print(f"  总损失 L: {L.item():.4f}")
    print(f"  心脏→视网膜损失 L_C: {L_C.item():.4f}")
    print(f"  视网膜→心脏损失 L_R: {L_R.item():.4f}")
    assert L.item() > 0, "损失应该为正数"
    print("  ✓ 通过\n")


def test_loss_all_positive():
    """测试所有样本都是正样本的情况"""
    print("测试3: 所有样本都是正样本")
    
    batch_size = 4
    latent_dim = 128
    
    z_R = torch.randn(batch_size, latent_dim)
    z_R = torch.nn.functional.normalize(z_R, p=2, dim=1)
    
    z_C = torch.randn(batch_size, latent_dim)
    z_C = torch.nn.functional.normalize(z_C, p=2, dim=1)
    
    # 所有标签相同
    labels = torch.ones(batch_size)
    
    L, L_C, L_R = cross_modal_contrastive_loss(z_R, z_C, labels, tau=0.1)
    
    print(f"  总损失 L: {L.item():.4f}")
    print(f"  心脏→视网膜损失 L_C: {L_C.item():.4f}")
    print(f"  视网膜→心脏损失 L_R: {L_R.item():.4f}")
    assert L.item() >= 0, "损失应该非负"
    print("  ✓ 通过\n")


def test_loss_all_negative():
    """测试所有样本都是负样本的情况"""
    print("测试4: 所有样本都是负样本（每个样本标签不同）")
    
    batch_size = 4
    latent_dim = 128
    
    z_R = torch.randn(batch_size, latent_dim)
    z_R = torch.nn.functional.normalize(z_R, p=2, dim=1)
    
    z_C = torch.randn(batch_size, latent_dim)
    z_C = torch.nn.functional.normalize(z_C, p=2, dim=1)
    
    # 每个样本标签不同（实际上每个样本自己还是正样本）
    labels = torch.tensor([0, 1, 2, 3])
    
    L, L_C, L_R = cross_modal_contrastive_loss(z_R, z_C, labels, tau=0.1)
    
    print(f"  总损失 L: {L.item():.4f}")
    print(f"  心脏→视网膜损失 L_C: {L_C.item():.4f}")
    print(f"  视网膜→心脏损失 L_R: {L_R.item():.4f}")
    assert L.item() >= 0, "损失应该非负"
    print("  ✓ 通过\n")


def test_loss_gradient():
    """测试梯度计算"""
    print("测试5: 梯度计算")
    
    batch_size = 4
    latent_dim = 128
    
    # 创建叶子节点
    z_R_raw = torch.randn(batch_size, latent_dim, requires_grad=True)
    z_C_raw = torch.randn(batch_size, latent_dim, requires_grad=True)
    
    # 归一化（保留梯度）
    z_R = torch.nn.functional.normalize(z_R_raw, p=2, dim=1)
    z_R.retain_grad()
    
    z_C = torch.nn.functional.normalize(z_C_raw, p=2, dim=1)
    z_C.retain_grad()
    
    labels = torch.tensor([1, 1, 0, 0])
    
    L, _, _ = cross_modal_contrastive_loss(z_R, z_C, labels, tau=0.1)
    
    # 反向传播
    L.backward()
    
    # 检查归一化后的张量的梯度
    if z_R.grad is not None:
        print(f"  z_R梯度范数: {z_R.grad.norm().item():.4f}")
    if z_C.grad is not None:
        print(f"  z_C梯度范数: {z_C.grad.norm().item():.4f}")
    
    # 检查原始张量的梯度（应该存在）
    assert z_R_raw.grad is not None, "z_R_raw应该有梯度"
    assert z_C_raw.grad is not None, "z_C_raw应该有梯度"
    assert z_R_raw.grad.norm() > 0, "z_R_raw梯度应该非零"
    assert z_C_raw.grad.norm() > 0, "z_C_raw梯度应该非零"
    
    print(f"  z_R_raw梯度范数: {z_R_raw.grad.norm().item():.4f}")
    print(f"  z_C_raw梯度范数: {z_C_raw.grad.norm().item():.4f}")
    print("  ✓ 通过\n")


def test_loss_temperature():
    """测试不同温度参数的影响"""
    print("测试6: 不同温度参数")
    
    batch_size = 4
    latent_dim = 128
    
    z_R = torch.randn(batch_size, latent_dim)
    z_R = torch.nn.functional.normalize(z_R, p=2, dim=1)
    
    z_C = torch.randn(batch_size, latent_dim)
    z_C = torch.nn.functional.normalize(z_C, p=2, dim=1)
    
    labels = torch.tensor([1, 1, 0, 0])
    
    for tau in [0.05, 0.1, 0.2, 0.5]:
        L, _, _ = cross_modal_contrastive_loss(z_R, z_C, labels, tau=tau)
        print(f"  tau={tau:.2f}: L={L.item():.4f}")
    
    print("  ✓ 通过\n")


if __name__ == "__main__":
    print("=" * 50)
    print("监督跨模态对比损失函数测试")
    print("=" * 50 + "\n")
    
    try:
        test_loss_basic()
        test_loss_module()
        test_loss_all_positive()
        test_loss_all_negative()
        test_loss_gradient()
        test_loss_temperature()
        
        print("=" * 50)
        print("所有测试通过！✓")
        print("=" * 50)
    except Exception as e:
        print(f"\n测试失败: {e}")
        import traceback
        traceback.print_exc()

