# 模型模块说明

## 投影头 (Projection Heads)

投影头将编码器输出映射到共享潜在空间，并进行L2归一化。

### ProjectionHead

可配置的多层MLP投影头。

```python
from models import ProjectionHead

# 创建投影头
projection_head = ProjectionHead(
    input_dim=1024,      # 视网膜编码器输出维度
    hidden_dim=512,     # 隐藏层维度
    output_dim=128,     # 共享潜在空间维度
    num_layers=2,       # MLP层数
    use_bn=True,        # 是否使用批归一化
    dropout=0.1         # Dropout比率
)

# 使用
h_R = retinal_encoder(retinal_img)  # (B, 1024)
z_R = projection_head(h_R)         # (B, 128), L2归一化
```

### SimpleProjectionHead

简单的2层投影头（用于快速实验）。

```python
from models import SimpleProjectionHead

projection_head = SimpleProjectionHead(
    input_dim=768,      # MRI编码器输出维度
    output_dim=128,     # 共享潜在空间维度
    use_bn=True
)
```

### 特性

- **自动L2归一化**: 输出自动归一化到单位超球面
- **灵活配置**: 支持自定义层数、隐藏维度、批归一化等
- **数值稳定**: 使用批归一化和Dropout提高训练稳定性

### 使用建议

1. **输出维度**: 通常设置为64-256之间，128是常用的选择
2. **隐藏维度**: 可以等于输入维度或稍小
3. **层数**: 2-3层通常足够，更多层可能导致过拟合
4. **批归一化**: 建议启用，有助于训练稳定性

