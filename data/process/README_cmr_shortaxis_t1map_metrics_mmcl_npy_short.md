## 任务介绍

### **任务：MRI embedding 预测传统 CMR 表型**  
  - **目的**：  
    验证 MedSAM 输出的表征是否与已知心脏表型（包括 LV ejection fraction、LV end diastolic volume、LV end systolic volume、LV stroke volume）存在相关性。  
  - **Encoder 使用方式**：  
    - 仅保留 **MedSAM 的 encoder 部分** 来编码 MRI，后面接简单的回归头；  
    - **数据预处理必须与 MedSAM 官方实现保持一致**（尺度、归一化、插值方式等，参考 `https://github.com/bowang-lab/MedSAM`）；  
    - 当前 `.npy` 中的三张切片（ED / mid / ES）在时间上是有严格顺序的：`[ED, mid-beat, ES]`，**不要在构造输入时打乱顺序**。  
    - MedSAM 原始设计是：**单张 2D 切片复制三遍作为 3 通道输入**；  
      - 因此我们这三张时间相位的切片**不能直接当作 RGB 三通道一起喂给 encoder**，否则破坏了 MedSAM 预训练时的输入假设；  
      - 可以的做法包括不限于：  
        - 分别对 ED / mid / ES 三张切片单独构造 3 通道输入（每张复制三遍），再在特征层面做拼接或聚合；  
  - **实验设计（回归任务）**：在有 CMR 结构/功能参数的 UKB 子队列上（已知包含：LV ejection fraction, LV end diastolic volume, LV end systolic volume, LV stroke volume，均为连续指标），针对这些指标训练回归模型：  
    - 使用 MRI embedding 预测上述连续指标，采用线性回归或简单 MLP，报告 **R²** 和 **MAE**；  
    - 可以采用 **一个共享 encoder + 多个回归头的多任务学习方式**：  
      - 同一个 MedSAM encoder 输出一个 embedding；  
      - 在其上接 4 个独立的回归头，分别预测 LVEF / LVEDV / LVESV / LV stroke volume，避免为每个表型单独训练一套模型；  
    - 训练策略上依次尝试：  
      1）**完全冻结** MedSAM encoder，仅训练回归头；  
      2）**解冻 encoder 的最后几层** 做轻量微调；  
      3）**解冻 encoder 全部层** 做端到端微调。  
  - **数据与结果呈现**：  
    - 训练/验证数据均来自 UK Biobank；  
    - 对每个表型（LVEF、LVEDV、LVESV、LV stroke volume）分别给出：R²、MAE，以及真实值 vs 预测值散点图，共 4 组结果图（4 张 figure）。  
  - **预期结论**：若在这些表型上取得合理的 R² / MAE，则可说明当前 MRI embedding **确实编码了心脏结构与功能信息**，而非仅仅提供一个无关的初始化。

## 数据介绍

### 1. 心脏 MRI（CMR）& 短轴 CINE 是啥？

- **CMR 是什么**：  
  心脏 MRI（Cardiac MRI, CMR）是一种给心脏做动态成像的 MRI 检查，可以看到心脏一个心动周期内的收缩和舒张。

- **短轴 CINE**：  
  - 把左心室从心底（base）到心尖（apex）切成多层，每一层在一个心动周期上拍一段小电影；  
  - 每一层通常有大约 **50 帧**，从舒张到收缩再回到舒张，这就是我们说的 **cine 序列**。

- **b1–b10 是啥**：  
  - `CINE_segmented_SAX_b1` ~ `b10` 是 UK Biobank 里短轴 cine 的不同采集组（不同屏气 / slab / acquisition 等）；  
  - 有多层短轴切面（不同高度），每一层都有约 50 帧时间序列。

---

### 2. 表 `cmr_shortaxis_t1map_metrics_mmcl_npy.csv` 是啥？

- **每一行代表**：一个受试者的一次 CMR 检查（`subject_id` + `instance`）。
- 主要字段（只列和图像相关的）：
  - **`subject_id`**：受试者 ID；  
  - **`instance`**：这次检查的实例编号；  
  - **`short_axis_seq`**：原始短轴 CINE 压缩包（zip）的路径（在 UKB 下载目录下）；  
  - **`short_axis_ED_mid_ES_npy`**：我们预处理生成的 `.npy` 文件路径（可能为空字符串，表示这一例失败或缺失）。
- 其他列是上游整理的各种 CMR 指标（T1、体积、功能参数等），也就是回归任务的gt。
- 已经按subject_id去重，确保训练集和测试集不会有overlap。
- 数据划分按照训练集：验证集：测试集=8:1:1

---

### 3. 这些 `.npy` 文件是什么？怎么用？

- **文件放在哪里**：  
  白玉兰`/home/bml/storage/mnt/v-044d0fb740b04ad3/org/UK_Bio/cmr_shortaxis_ED_mid_ES/` 目录下，每个样本一个文件，例如：  
  `1001044_inst2_shortaxis_ED_mid_ES.npy`
  该目录下一共30656个npy文件。

- **每个 `.npy` 文件的内容**：
  - 是一个 `numpy` 数组；
  - 形状：`(3, H, W)`，数据类型：`float32`；
  - 维度 0 的三张切片依次是：
    1. **ED**：舒张早期（心动周期最早的帧）；  
    2. **mid-beat**：ED 和 ES 之间的中间时间点；  
    3. **ES**：收缩末期（大约 0.35×RR 附近）。

- **这些三帧是怎么选出来的（概念理解即可）**：
  1. 先用 DICOM 里的 `SliceLocation` / `ImagePositionPatient[2]` 找到**中层短轴**（在从心底到心尖的层面里选一个“中间层”）；  
  2. 再在这一层的 50 帧时间序列里，用 `TriggerTime` / `NominalInterval` 选出 ED、ES 和它们中间的 mid-beat。

- **如何在代码里读取**（示例）：

```python
import numpy as np

path = "<某一行的 short_axis_ED_mid_ES_npy 字段值>"  # 例如来自 pandas 里的某个单元格
arr = np.load(path)        # arr.shape == (3, H, W), dtype=float32
ed, mid, es = arr[0], arr[1], arr[2]
```

- **像素值说明**：
  - 当前只是从 DICOM 里读出的原始像素（转为 float32），**还没有做强度归一化或 MedSAM 特定预处理**；  
  - 下游任务（如分割、表型预测、多模态配准等）可以在此基础上自行做归一化、resize、标准化等。

---


