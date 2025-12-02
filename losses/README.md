# 损失函数模块说明

## 监督跨模态对比损失

本模块实现了 DeepCAD Stage I 的监督跨模态对比学习损失函数，严格按照数学定义实现。

### 数学定义

对于批次 B 中的 N 个样本，每个样本 j 有：
- 视网膜投影：z_j^R
- 心脏MRI投影：z_j^C
- CAD标签：y_j ∈ {0, 1}

定义正样本集合：P(j) = {k ∈ B : y_k = y_j}

**心脏→视网膜损失**：
```
L_C = -1/N * sum_{j∈B} log(
  (sum_{k∈P(j)} exp(cos(z_j^C, z_k^R)/τ)) /
  (sum_{k∈B} exp(cos(z_j^C, z_k^R)/τ))
)
```

**视网膜→心脏损失**（对称）：
```
L_R = -1/N * sum_{j∈B} log(
  (sum_{k∈P(j)} exp(cos(z_j^R, z_k^C)/τ)) /
  (sum_{k∈B} exp(cos(z_j^R, z_k^C)/τ))
)
```

**总损失**：
```
L = L_C + L_R
```

其中：
- `cos(·,·)` 是余弦相似度
- `τ` 是温度参数（默认 0.1）

### 使用示例

#### 方式1: 函数调用

```python
import torch
from losses import cross_modal_contrastive_loss

# 准备数据
batch_size = 32
latent_dim = 128

z_R = torch.randn(batch_size, latent_dim)  # 视网膜投影
z_R = torch.nn.functional.normalize(z_R, p=2, dim=1)

z_C = torch.randn(batch_size, latent_dim)  # 心脏MRI投影
z_C = torch.nn.functional.normalize(z_C, p=2, dim=1)

labels = torch.randint(0, 2, (batch_size,))  # CAD标签

# 计算损失
L, L_C, L_R = cross_modal_contrastive_loss(
    z_R=z_R,
    z_C=z_C,
    labels=labels,
    tau=0.1
)

print(f"总损失: {L.item():.4f}")
print(f"心脏→视网膜损失: {L_C.item():.4f}")
print(f"视网膜→心脏损失: {L_R.item():.4f}")
```

#### 方式2: Module类

```python
import torch
from losses import CrossModalContrastiveLoss

# 创建损失函数
criterion = CrossModalContrastiveLoss(tau=0.1)

# 计算损失
L, L_C, L_R = criterion(z_R, z_C, labels)
```

### 参数说明

- `z_R`: 视网膜投影，形状为 `(batch_size, latent_dim)`，应该已L2归一化
- `z_C`: 心脏MRI投影，形状为 `(batch_size, latent_dim)`，应该已L2归一化
- `labels`: CAD标签，形状为 `(batch_size,)`，值为0或1
- `tau`: 温度参数（默认0.1），较小的值会使损失更关注困难样本
- `eps`: 数值稳定性的小常数（默认1e-8）

### 返回值

返回一个元组 `(L, L_C, L_R)`：
- `L`: 总损失（标量）
- `L_C`: 心脏→视网膜损失（标量）
- `L_R`: 视网膜→心脏损失（标量）

### 辅助函数

#### compute_positive_mask

计算正样本掩码：

```python
from losses import compute_positive_mask

labels = torch.tensor([1, 1, 0, 0])
mask = compute_positive_mask(labels)
# mask[i, j] = 1 如果 labels[i] == labels[j]，否则为 0
```

#### cosine_similarity_matrix

计算两个嵌入集合之间的余弦相似度矩阵：

```python
from losses import cosine_similarity_matrix

z1 = torch.randn(10, 128)
z2 = torch.randn(5, 128)
sim_matrix = cosine_similarity_matrix(z1, z2)  # (10, 5)
```

### 注意事项

1. **输入归一化**: 虽然函数内部会再次归一化，但建议输入已经L2归一化
2. **批次大小**: 损失函数需要批次中有多个样本才能计算对比损失
3. **正样本数量**: 如果某个样本在批次中没有正样本（相同标签），损失计算仍然有效（只考虑自己作为正样本）
4. **温度参数**: 
   - 较小的τ（如0.05）会使损失更关注困难样本
   - 较大的τ（如0.5）会使损失更平滑
   - 默认0.1是经过验证的良好选择

### 测试

运行测试脚本验证实现：

```bash
python tests/test_loss.py
```

### 数学验证

损失函数严格遵循文档中的数学定义：
- 使用余弦相似度计算样本间的相似性
- 通过标签构建正样本集合 P(j)
- 分别计算两个方向的损失并求和
- 使用温度缩放和数值稳定性处理

