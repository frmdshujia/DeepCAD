# 自动诊断报告

生成时间: 2026-04-12T14:11:04.104748

## 1. 数据与 S_GT（diagnostics_contrast）

```
[CMRBank] split=train, n_cmr=54408, shape=torch.Size([54408, 14]), device=cuda, unique_eid_in_bank=51703, paired_eid_table=51703

========== [A] EID 覆盖（Fundus train ↔ CMR 全表配对）==========
  Fundus train 行数: 3091，唯一 EID: 1588
  CMR 配对表 unique EID: 51703
  Fundus 中无法在 CMR 配对表查到的 EID 数: 0
  覆盖率: 1.0000
  → 判定: EID 覆盖良好。

========== [B] S_GT 统计（单 batch，含正样本在前 B 维）==========
[FundusContrastDataset] train 子采样: 64/3091 行 (ratio=1.0, max_samples=64, seed=42)
[FundusContrastDataset] split=train: 可用样本=64, 唯一EID=64
  batch_size B=16, K=1024, σ=6.5893
  S_GT 对角（正样本槽）: mean=0.999943, min=0.999808, max=1.000000
  每行非对角均值（粗看「背景」）: mean=0.611570
  对角 − 非对角（越大越好）: mean=0.388374
  softmax(S_GT) 行熵: mean=6.9222（均匀分布约 log(K)=6.9315）
  → 判定: 目标分布接近均匀，soft 标签很「平」，loss 会贴近 log(K)、下降慢。

========== [B2] σ 敏感度（同 batch，σ × {0.5, 1, 2}）==========
[FundusContrastDataset] train 子采样: 64/3091 行 (ratio=1.0, max_samples=64, seed=42)
[FundusContrastDataset] split=train: 可用样本=64, 唯一EID=64
  σ=3.294650 (×0.5): diag_mean=0.999878, off_mean=0.159285, diag−off=0.840594
  σ=6.589300 (×1.0): diag_mean=0.999970, off_mean=0.564792, diag−off=0.435178
  σ=13.178600 (×2.0): diag_mean=0.999992, off_mean=0.856512, diag−off=0.143481
  → 若随 σ 变化对角优势消失很快，说明 PC 距离相对 σ 的尺度需复查。

========== 汇总阅读顺序建议 ==========
  1) EID 覆盖差 → 先修表/拆分，再谈模型。
  2) S_GT 熵≈log(K) → 目标过平：σ、K、或任务本身信息弱。
  3) 梯度≈0 → 实现/冻结/数值；梯度正常但 loss 不降 → 看 2) 与 LR。
  4) 单 batch 过拟合仍不降 → 实现或目标定义；能降 → 正式训练慢多半来自 LR/数据量/K。
完成。


```

## 2. 冻结特征 MLP 回归 PC（Pearson）

```
Train: torch.Size([3091, 1024]), Val: torch.Size([386, 1024]), PC dim: 14

--- hidden=256, wd=0.1 ---
  [ep  19] loss=0.9347  train_R=0.3387  val_R=0.0142  val_R_max=0.1438
  [ep  39] loss=0.8542  train_R=0.4655  val_R=0.0004  val_R_max=0.1357
  [ep  59] loss=0.7590  train_R=0.5747  val_R=0.0016  val_R_max=0.1911
  [ep  79] loss=0.6688  train_R=0.6532  val_R=0.0155  val_R_max=0.2049
  [ep  99] loss=0.5971  train_R=0.7173  val_R=0.0178  val_R_max=0.1809
  [ep 119] loss=0.5319  train_R=0.7755  val_R=0.0192  val_R_max=0.1769
  [ep 139] loss=0.4854  train_R=0.8113  val_R=0.0164  val_R_max=0.1663
  [ep 159] loss=0.4555  train_R=0.8295  val_R=0.0171  val_R_max=0.1627
  [ep 179] loss=0.4364  train_R=0.8409  val_R=0.0188  val_R_max=0.1689
  [ep 199] loss=0.4288  train_R=0.8437  val_R=0.0179  val_R_max=0.1635

--- hidden=64, wd=0.5 ---
  [ep  19] loss=0.9646  train_R=0.2381  val_R=0.0088  val_R_max=0.1525
  [ep  39] loss=0.9264  train_R=0.3068  val_R=0.0176  val_R_max=0.1894
  [ep  59] loss=0.8837  train_R=0.4028  val_R=0.0097  val_R_max=0.1904
  [ep  79] loss=0.8453  train_R=0.4786  val_R=0.0125  val_R_max=0.1460
  [ep  99] loss=0.7942  train_R=0.5254  val_R=0.0102  val_R_max=0.1520
  [ep 119] loss=0.7585  train_R=0.5750  val_R=0.0011  val_R_max=0.1272
  [ep 139] loss=0.7147  train_R=0.6167  val_R=0.0064  val_R_max=0.1431
  [ep 159] loss=0.6841  train_R=0.6461  val_R=0.0043  val_R_max=0.1164
  [ep 179] loss=0.6738  train_R=0.6618  val_R=0.0070  val_R_max=0.1307
  [ep 199] loss=0.6658  train_R=0.6655  val_R=0.0075  val_R_max=0.1303

Done.

```

## 3. 随机初始化头（未训练）验证集 gt_pred_spearman / Pearson

```json
{
  "R@1": 0.0,
  "R@5": 0.0,
  "R@10": 0.0,
  "paired_cosine": -0.006147295830124709,
  "n_matched": 386,
  "alignment": 2.0122947692871094,
  "gt_pred_spearman": -0.023271519416725648,
  "gt_pred_pearson": -0.013597721139300722,
  "n_pairs": 8000
}
```

## 4. 学习曲线（fast_contrast_train，linear_proj，100 epoch）

### train_eid_frac = 0.25

- 用时: 17.6s, exit=0
- 末 epoch 指标:
```json
{
  "epoch": 99,
  "loss": 3.002885580062866,
  "R@1": 0.0,
  "R@5": 0.0,
  "R@10": 0.0,
  "paired_cosine": 0.004358108311051918,
  "n_matched": 386,
  "alignment": 1.9912837743759155,
  "gt_pred_spearman": -0.026327976422660843,
  "gt_pred_pearson": -0.01631478683245548,
  "n_pairs": 8000
}
```

### train_eid_frac = 0.5

- 用时: 20.6s, exit=0
- 末 epoch 指标:
```json
{
  "epoch": 99,
  "loss": 3.528805414835612,
  "R@1": 0.0,
  "R@5": 0.0,
  "R@10": 0.0,
  "paired_cosine": 0.016268337553202953,
  "n_matched": 386,
  "alignment": 1.9674633741378784,
  "gt_pred_spearman": 0.11082523300466636,
  "gt_pred_pearson": 0.11274422354881812,
  "n_pairs": 8000
}
```

### train_eid_frac = 1.0

- 用时: 25.9s, exit=0
- 末 epoch 指标:
```json
{
  "epoch": 99,
  "loss": 4.1172086000442505,
  "R@1": 0.0,
  "R@5": 0.0,
  "R@10": 0.0025906735751295338,
  "paired_cosine": -0.005910535690427753,
  "n_matched": 386,
  "alignment": 2.0118212699890137,
  "gt_pred_spearman": -0.01072304493902486,
  "gt_pred_pearson": -0.006775053224617049,
  "n_pairs": 8000
}
```

## 5. 温度对比（linear_proj，train_eid_frac=1，120 epoch）

### temperature = 0.07

```json
{
  "epoch": 119,
  "loss": 3.2496219277381897,
  "R@1": 0.0,
  "R@5": 0.0,
  "R@10": 0.0,
  "paired_cosine": 0.0012355160023631055,
  "n_matched": 386,
  "alignment": 1.997529149055481,
  "gt_pred_spearman": -0.049289990640570956,
  "gt_pred_pearson": -0.04769797666799101,
  "n_pairs": 8000
}
```

### temperature = 0.2

```json
{
  "epoch": 119,
  "loss": 4.253470579783122,
  "R@1": 0.0,
  "R@5": 0.0025906735751295338,
  "R@10": 0.0025906735751295338,
  "paired_cosine": -0.007458363914225308,
  "n_matched": 386,
  "alignment": 2.0149166584014893,
  "gt_pred_spearman": -0.06355497543466011,
  "gt_pred_pearson": -0.0631331403702,
  "n_pairs": 8000
}
```

## 6. 总结（自动归纳）

### 学习曲线末轮 gt_pred_spearman / Pearson（越高越好）

- **learning_curve_frac_0.25**: Spearman=-0.026327976422660843, Pearson=-0.01631478683245548, paired_cosine=0.004358108311051918, R@5=0.0
- **learning_curve_frac_0.5**: Spearman=0.11082523300466636, Pearson=0.11274422354881812, paired_cosine=0.016268337553202953, R@5=0.0
- **learning_curve_frac_1.0**: Spearman=-0.01072304493902486, Pearson=-0.006775053224617049, paired_cosine=-0.005910535690427753, R@5=0.0

### 解读提示

- **gt_pred_spearman**：跨人「fundus·CMR 余弦」与 PC 高斯 GT 的秩相关；比 R@k 更贴近 soft 目标。
- 若随 **train_eid_frac** 增大而 Spearman **单调升**，扩大样本大概率有效。
- 若随机初始化 Spearman 已接近训练后，说明 **嵌入空间尚未学到表型结构** 或信号极弱。
- 冻结特征回归 val Pearson≈0 且 gt_pred 仍低 → **瓶颈多在表征/信号**，非仅调参。

## 7. 执行摘要（给课题决策）

### 已排除或确认的原因

| 假设 | 结果 |
|------|------|
| EID 未对齐 / 训练缺正样本 | **已排除**：Fundus train 1588 EID 在 CMR 配对表中覆盖率 **100%** |
| 仅实现 bug 导致「训不好」 | **基本排除**：管道可跑通；新指标 **gt_pred_spearman** 与随机初始化同量级波动，说明不是单一工程错误 |
| Soft 目标过平 | **确认存在**：`softmax(S_GT)` 行熵≈log(K)，与此前分析一致，会拖慢有效学习 |

### 仍占主导的原因（证据）

| 假设 | 证据 |
|------|------|
| **冻结 CLS 可泛化预测 PC 的信号极弱** | 回归基线：val 上 **平均 Pearson R≈0.02**，与「信号弱」一致 |
| **跨模态表型几何对齐难** | 多数设置下 **gt_pred_spearman≈0 或小幅正负**；仅 `train_eid_frac=0.5` 一轮出现 **Spearman≈0.11**（需 **固定多种子重复** 判断是否稳定） |
| **R@k 作为指标过严** | 多数 run 的 R@5=0，但 **gt_pred** 有时略正，更宜作为主监控 |

### 学习曲线（本次快速实验）

- **未呈现单调关系**：0.25 → 负相关；0.5 → Spearman **+0.11**；1.0 → 又接近 0。  
- **解读**：100 epoch / 线性头 / 单次种子下 **方差大**，不能据此断言「数据越多越差」；需要 **多 seed + 更长训练** 再下结论。

### 建议的下一步（未在本次自动跑完）

- `diagnostics_contrast.py --checks grad overfit`（单 batch 过拟合，验证梯度管道）。  
- **ViT 微调**若干组（只训顶层 / 小 LR），与 fast 线性头对照。  
- **辅助 PC 回归损失** + 对比损失。  
- **多种子**重复 `train_eid_frac` 与温度扫描。

完整机器可读结果：`output_dir/diagnostic_suite_results.json`。
