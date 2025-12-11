## 文件概览

`cmr_shortaxis_t1map_metrics_mmcl_npy.csv` 是在 UK Biobank（UKB）心脏 MRI（cardiac MRI, CMR）原始 DICOM 基础上整理得到的 **短轴中层三相位切片索引表**。  
它基于原始表 `cmr_shortaxis_t1map_metrics.csv`，新增了一列指向 `.npy` 文件的路径，每个 `.npy` 文件对应某个受试者在一次检查中的 **中层短轴 ED / mid-beat / ES 三个时间点的 2D 图像堆叠**。

- **原始表**：`cmr_shortaxis_t1map_metrics.csv`
- **新增表**：`cmr_shortaxis_t1map_metrics_mmcl_npy.csv`
- **生成的图像文件目录**：`cmr_mmcl_slices/`
- **单个样本的 npy 文件命名**：`{subject_id}_inst{instance}_shortaxis_ED_mid_ES.npy`
  - 数组形状：`(3, H, W)`，数据类型 `float32`
  - 维度 0 顺序：`[ED, mid-beat, ES]`

该 README 主要说明：

- **UK Biobank 心脏 MRI 基本情况**
- **短轴 CINE 序列及 b1–b10 的含义**
- **“每个 b 里大约 50 张切片”具体指什么**
- **`cmr_shortaxis_t1map_metrics_mmcl_npy.csv` 与 `.npy` 的生成流程与字段说明**

---

## UK Biobank 心脏 MRI 简介

- **数据来源**：  
  UK Biobank CMR 数据。单次检查通常包含多套 cine 序列（短轴 SAX、长轴 2CH/3CH/4CH 等）、T1/T2 mapping 等。

- **cine 序列特点**：  
  - 常见序列命名：`CINE_segmented_SAX_b1` ~ `CINE_segmented_SAX_b10` 等。  
  - 这是典型 SSFP 心脏 cine：一个心动周期被离散为约 **50 个时间帧（frames）**，从舒张早期到收缩末期，再回到舒张。
  - 在 DICOM 中：
    - **时间维度**通常由 `TriggerTime` 和/或 `InstanceNumber` 描述；
    - **空间维度（切片高度）**由 `SliceLocation` 或 `ImagePositionPatient[2]` 描述（从心底 base 到心尖 apex）。

简单理解：UKB CMR 的短轴 cine 是一个 4D 数据：  
\( \text{time} \times \text{slice (base→apex)} \times H \times W \)。

---

## 短轴 CINE 及 b1–b10 的含义

- **SAX（short-axis）短轴序列**：
  - 短轴是垂直于左心室长轴的切面，一般从心底（base）到心尖（apex）分成多个层面。

- **`CINE_segmented_SAX_b1` ~ `b10` 是什么？**
  - 在 UKB 的命名习惯中，`CINE_segmented_SAX_b1`、`..._b2`、…、`..._b10` 通常表示：
    - 同一次检查中，不同呼吸屏气 / 不同 slab / 不同 acquisition 设置的 **短轴 cine 序列组**。
  - 每个 `bX` 序列内部又可以细分为若干 `seriesid`：
    - 这些 series 的 `SliceLocation` 不同，对应从心底到心尖的多个空间切面。

- **空间维度 vs 时间维度**：
  - **空间维度**：短轴层面（不同 z，高度），通常 2–10 个不等；
  - **时间维度**：每个层面在 **一个心动周期上约 50 帧**。

因此可以这样记：

- **b1–b10**：一组短轴 cine 序列（多次采集 or 多 slab）；
- **每个 b 里面的约 50 张切片**：对某个给定短轴层面来说，这 50 张图像本质上是 **同一空间位置在一个心动周期上的 50 个时间帧**。

---

## 原始表 `cmr_shortaxis_t1map_metrics.csv` 概述

`cmr_shortaxis_t1map_metrics.csv` 是在上游处理阶段整理的 CMR 结构信息表，主要包含：

- **受试者相关字段**（示例）：
  - `subject_id`：UKB participant / subject ID
  - `instance`：检查实例编号（同一受试者多次检查可区分）

- **短轴原始序列字段**：
  - `short_axis_seq`：指向短轴 CINE 压缩包（zip）的相对路径  
    - 压缩包内部包括若干 DICOM 文件及一个 manifest 描述表。

- **T1 / 体积功能等 CMR 指标**（视上游整理而定）：
  - 与 T1-mapping、心腔容积、射血分数等相关的量化特征。

在此基础上，我们新增了图像 stack 的索引列，形成 `cmr_shortaxis_t1map_metrics_mmcl_npy.csv`。

---

## 从 DICOM 到中层短轴 ED/mid/ES 三帧的处理流程
 
**“基于 DICOM header 从短轴 b1–b10 中提取中层短轴的 ED / mid-beat / ES 三个时间点”**。

### 1. 输入 / 输出路径

- **输入表**：  
  - `cmr_shortaxis_t1map_metrics.csv`（位于 `output_dir`）

- **原始 CMR DICOM zip 根目录**：
  - `cmr_base_dir = /data/home/shujia/UKB/CMRI/downloaded`

- **输出目录**：
  - `out_dir = output_dir / "cmr_mmcl_slices"`
  - 单个 `.npy` 文件命名：`{subject_id}_inst{instance}_shortaxis_ED_mid_ES.npy`

- **最终索引表**：
  - `out_csv = output_dir / "cmr_shortaxis_t1map_metrics_mmcl_npy.csv"`

### 2. 逐行遍历样本

对于原始表中的每一行（一个 subject-instance）：

- 读取：
  - `subject_id`
  - `instance`
  - `short_axis_seq`（短轴 zip 路径）
- 在 `cmr_base_dir` 下定位对应的 zip 文件：
  - `sa_zip = cmr_base_dir / short_axis_seq`

### 3. 读取 manifest 并筛选短轴 CINE 序列

- 在 zip 内查找名为 `*manifest.csv` 或 `*manifest.cvs` 的文件，读入为 `pandas.DataFrame`。
- 在 manifest 中筛选：
  - 列 `series discription` 存在；
  - 文本满足正则：`CINE_segmented_SAX_b[0-9]`（包含短轴 CINE）；
  - 排除描述中包含 `InlineVF` 等后处理或派生序列。

筛选结果记为 `sax_b`，代表 **候选短轴 CINE 原始序列**。

### 4. 根据 SliceLocation 选择“中层”短轴序列

- 按 `seriesid` 分组，针对每个 `seriesid`：
  - 抽取前若干个 DICOM 文件（例如前 5 个），只读取 header（`stop_before_pixels=True`）；
  - 优先读取 `SliceLocation`；
  - 若缺失，则使用 `ImagePositionPatient[2]` 作为 z 坐标；
  - 计算该 `seriesid` 的平均 z 值 `z_mean`。
- 收集所有 `seriesid` 的 `{seriesid, z_mean, series discription}`：
  - 按 `z_mean` 排序，即从心底到心尖的顺序；
  - 取中间位置的 `seriesid` 作为 **“中层短轴序列”**。

### 5. 在中层序列中读取所有时间帧

对于选中的 `mid_seriesid`：

- 在 manifest 中找到所有属于该 `seriesid` 的 DICOM 文件；
- 逐个读取：
  - `TriggerTime`（如果存在）；
  - `InstanceNumber`；
  - `NominalInterval`（RR 间期，如存在）；
  - `pixel_array`（转为 float32）。
- 对所有帧进行时间排序：
  - 若所有帧均有 `TriggerTime`：按 `TriggerTime` 升序；
  - 否则：按 `InstanceNumber` 升序；
- 得到按心动周期顺序排列的帧列表。

### 6. 从时间序列中选出 ED / mid-beat / ES 三个相位

这一部分由函数 `_select_ed_es_mid_from_frames` 完成，核心规则如下：

- **无 TriggerTime 情况（退化方案）**：
  - 使用帧索引近似：
    - **ED**：第 1 帧（索引 0）；
    - **ES**：中间帧（索引 `n // 2`）；
    - **mid-beat**：ED 与 ES 索引的中点（`(0 + n // 2) // 2`）。

- **有 TriggerTime 情况**：
  - 将所有帧的 `TriggerTime` 记为数组 `tts`：
    - **ED**：
      - 取 `TriggerTime` 最小的帧，代表心动周期最早的时间点（接近舒张早期 / ED）。
    - **ES**：
      - 若存在有效的 `NominalInterval`（RR 间期）：
        - 在时间点 `0.35 * RR` 附近寻找最近的帧，近似收缩末期（ES）。
      - 若无 `NominalInterval`：使用中位帧近似 ES。
    - **mid-beat**：
      - 取 ED 与 ES 时间的中点：
        - `mid_target = 0.5 * (tts[ed_idx] + tts[es_idx])`；
      - 在所有帧中找到 `TriggerTime` 最接近 `mid_target` 的帧。

最终输出三幅 2D 图像：`ed_img`, `mid_img`, `es_img`。

### 7. 生成 (3, H, W) 的 npy 图像 stack

- 若三幅图像均成功获得：
  - 将其按顺序堆叠为一个 `numpy` 数组：
    - 形状：`(3, H, W)`；
    - 数据类型：`float32`；
    - 维度 0 顺序：`[ED, mid-beat, ES]`。
  - 保存路径：
    - `out_dir / f"{subject_id}_inst{instance}_shortaxis_ED_mid_ES.npy"`。

- 若中途任一步失败（例如：zip 缺失、manifest 缺失、没有满足条件的短轴序列、DICOM 读失败等）：
  - 不写入 `.npy` 文件；
  - 在最终表中对应路径字段置为空字符串 `""`。

---

## `cmr_shortaxis_t1map_metrics_mmcl_npy.csv` 字段说明

在 `cmr_shortaxis_t1map_metrics.csv` 所有原有字段基础上，**新增一列**：

- **`short_axis_ED_mid_ES_npy`**：
  - **类型**：字符串；
  - **含义**：该 subject-instance 对应的中层短轴三相位 stack `.npy` 文件路径；
  - **取值约定**：
    - 预处理成功：为有效文件路径（如 `.../cmr_mmcl_slices/1001044_inst2_shortaxis_ED_mid_ES.npy`）；
    - 预处理失败或 zip 缺失：为空字符串 `""`。

对应的 `.npy` 文件内容：

- `numpy` 数组；
- 形状：`(3, H, W)`；
- `dtype=float32`；
- 维度 0 的含义：
  - 第 0 帧：ED（舒张早期 / 心动周期最早帧）；
  - 第 1 帧：mid-beat（ED 与 ES 之间的中间时间点）；
  - 第 2 帧：ES（收缩末期，约 0.35×RR 附近）。

当前版本中，图像未做强度标准化或 MedSAM 特定预处理，仅对原始像素做了类型转换（转为 float32）。

---

## 如何复现实验 / 重新生成该表

- **依赖环境**：
  - Python 包：`pydicom`, `numpy`, `pandas`, `tqdm`, `scikit-image` 等；
  - 建议安装加速 DICOM 解码的依赖（如 `pylibjpeg` / `pylibjpeg-libjpeg` / `pylibjpeg-openjpeg` 或 `gdcm`），以提高 JPEG/JPEG2000 解码速度。

- **数据准备**：
  - 原始 CMR zip：`/data/home/shujia/UKB/CMRI/downloaded`；
  - 预先整理好的结构化表：`output_dir/cmr_shortaxis_t1map_metrics.csv`。

- **复现步骤**：
  1. 打开 `dataprocess.ipynb`；
  2. 找到标题为  
     **“基于 DICOM header 从短轴 b1–b10 中提取中层短轴的 ED / mid-beat / ES 三个时间点（MedSAM 风格预处理）”** 的代码单元；
  3. 根据需要调整多线程参数（如 `ThreadPoolExecutor(max_workers=8~16)`），然后运行该单元；
  4. 程序会：
     - 逐行处理 `cmr_shortaxis_t1map_metrics.csv` 中的样本；
     - 在 `cmr_mmcl_slices/` 目录下写入对应的 `.npy` 文件；
     - 生成 / 覆盖 `cmr_shortaxis_t1map_metrics_mmcl_npy.csv`，并在其中填写 `short_axis_ED_mid_ES_npy` 列。

如需为团队做进一步的下游任务说明（例如：如何加载该 npy 做分割 / 表型预测 / 多模态对齐等），可以在本 README 基础上新增对应章节。


