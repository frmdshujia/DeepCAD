# 数据准备请求 — 扩展 Stage 2 对比学习 Cohort 到 ~25K EID

**请求发起者**：对比学习训练 agent (2026-04-21)
**目标读者**：数据工程 agent
**本次请求不阻塞当前 Tier 1 (1,196 EID) 训练，属于为 Tier 2/3 扩量做的前置数据工程**

---

## 1. 背景一页纸

当前 `contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table.csv`
只索引了 **1,983 个 EID**（全部 instance=2，即 "same-instance" 眼底-CMR 配对）。

但实地扫描磁盘发现原始数据远多于此：

| 资源 | 路径 | 唯一 EID 数 |
|---|---|---|
| Fundus 图 (21015, all instances) | `/data/home/home6/fundus_data/UKB/fundus_images/{eid}_21015_{instance}_{eye}.png` | **96,827** |
| └─ instance 0（baseline visit） | | 67,679 |
| └─ instance 1（first repeat） | | 19,316 |
| └─ instance 2（imaging visit 1） | | 2,957 |
| └─ instance 3（imaging visit 2） | | 12,363 |
| CMR zip LAX (20208) + SAX (20209) inst=2 | `/data/home/shujia/UKB/CMRI/downloaded/{eid}_{field}_2_0.zip` | **74,754** |
| CMR zip LAX + SAX inst=3 | 同目录 | 5,147 |
| 任意 fundus ∩ CMR(LAX+SAX) inst=2 | — | **28,154** ← 目标池 |

分解：
- fundus inst=0 ∩ CMR inst=2 → 9,745 EID（跨 instance，fundus 早于 CMR ~5-7y）
- fundus inst=1 ∩ CMR inst=2 → 9,018 EID（跨 instance，fundus 早于 CMR ~3-5y）
- fundus inst=2 ∩ CMR inst=2 → 1,534 EID（same-instance，当前 cohort 主体）
- fundus inst=3 ∩ CMR inst=2 → 11,754 EID（fundus 晚于 CMR ~2y）

**结论**：handoff 里说的"25K xinst cohort"本来就存在，无需下载任何新数据，只要把 fundus 表
从 "only instance=2" 扩展到 "all instances"，就能解锁 28K EID × avg 1.5 fundus/人 ≈ 30-50K
条 training pair。

---

## 2. 核心交付物（按优先级）

### 任务 1（必须，最高优先）：扩展版 fundus_table

**文件路径**：
`contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table_extended.csv`

**行粒度**：每张 fundus 图一行（同一 EID 可有多行 = 多 instance × 左右眼）

**必需字段**：

| 列名 | 类型 | 说明 |
|---|---|---|
| `eid` | int | UKB EID |
| `fundus_instance` | int ∈ {0,1,2,3} | fundus 访问点 |
| `eye` | str ∈ {'left','right'} | 解析自文件名最后数字：`..._0.png`=left, `..._1.png`=right |
| `fundus_image_path` | str | 绝对路径，确保文件实际存在 |
| `cmr_instance` | int ∈ {2,3} | 配对的 CMR 访问点。优先 2（imaging visit 1，数据量大）|
| `cmr_lax_zip` | str | `/data/home/shujia/UKB/CMRI/downloaded/{eid}_20208_{cmr_instance}_0.zip`，必须存在 |
| `cmr_sax_zip` | str | `/data/home/shujia/UKB/CMRI/downloaded/{eid}_20209_{cmr_instance}_0.zip`，必须存在 |
| `pair_type` | str ∈ {'same_inst','cross_inst'} | `fundus_instance == cmr_instance` → same, 否则 cross |
| `visit_date_fundus` | date (YYYY-MM-DD) | 来自 UKB field 53, instance=fundus_instance。允许为空 |
| `visit_date_cmr` | date | field 53 at instance=cmr_instance。允许为空 |
| `visit_interval_years` | float | `(visit_date_cmr - visit_date_fundus).days / 365.25`；可为负 |
| `split` | str ∈ {'train','val','test'} | 见下面规则 |

**Split 规则（铁律）**：
1. **继承现有 fundus_table.csv 的 split**：原 1983 EID 的 split 不得变动
2. 新增 EID（26,171 个）按 **EID 级** 70/15/15 随机切，**seed=20260421 固定**
3. 同一 EID 的所有行必须在同一 split（若同 EID 有 instance 0+2+3 多条，全部进同一个 split）

**过滤规则**：
- 必须 fundus_image_path + cmr_lax_zip + cmr_sax_zip **三者实际存在**才保留
- 优先 `cmr_instance=2`；仅当同人不存在 CMR inst=2 但存在 inst=3 时用 inst=3（覆盖
  少量 edge case）
- 删除 `fundus_image_path` 文件打不开 / size==0 / 损坏的行（顺手做一次 `PIL.Image.open +
  verify`）

**预期产出规模**：
- 行数：~30-50K（取决于左右眼 + 多 instance 重复）
- 唯一 EID：~28K
- split 分布：train ~19.6K / val ~4.2K / test ~4.2K

---

### 任务 2（次优先）：扩展 master_long 用于 event filtering

**文件路径**：
`contrastive_pretrain/raw/master_long_extended.csv`

**用途**：Tier 3 (25K full training) 需要过滤"fundus 和 CMR 两次 imaging 之间
发生新发 CVD 事件"的 EID（~5-8% 预计会被剔除），避免标签漂移污染对比学习。

**必需字段**：

| 列名 | 说明 |
|---|---|
| `eid` | UKB EID |
| `baseline_date` | field 53, instance=0 |
| `fundus_date_i{0,1,2,3}` | field 53 at each instance (可空) |
| `cmr_date_i{2,3}` | field 53 at CMR instance (可空) |
| `mi_date` | Myocardial infarction first occurrence date (field 42000 或 HES 合并) |
| `stroke_date` | Stroke first occurrence date |
| `hf_date` | Heart failure first occurrence date |
| `af_date` | Atrial fibrillation first occurrence date |
| `cad_date` | Coronary artery disease first occurrence date (I20-I25) |
| `death_date` | field 40000 |
| `censoring_date` | 数据截止日 |

**Filtering logic (下游使用时会这样用，不需要你在这个文件里做过滤)**：
```python
# For each (eid, fundus_instance, cmr_instance) row in fundus_table_extended:
#   event_between = any(event_date is between fundus_date and cmr_date)
# 如果 event_between = True，该 row 被排除出 Stage 2 训练
```

**覆盖要求**：28K cohort **全员**覆盖。当前的 `contrastive_pretrain/raw/master_long.csv`
只有 ~10K EID 有 baseline_date，需要补齐。

---

### 任务 3（可选，Stage 1a 需要才做）：CMR PC score @ instance 3

**背景**：目前 `cmr_table.csv` 只有 64,775 EID 的 PC scores，且没明确标注 instance。需要确认：
1. 这些 PC 分是 CMR @ instance 2 算出来的（多半是）
2. 是否有 instance=3 的 ~5K EID 也需要 PC？如果 Stage 1a (soft-label InfoNCE on PC) 要用
   inst=3 的 fundus ↔ inst=3 的 CMR 做 same-instance 配对，才需要

**结论**：**Stage 2 (image-level) 不需要这项**。Stage 1a 如果启用 cross-instance 且 CMR
也允许 inst=3，再单独做。**先不做**。

---

## 3. 产出后的 Sanity Check 清单

交付前请跑（不要合并到代码仓，放到一份 check log 里即可）：

```python
# check_fundus_extended.py
df = pd.read_csv('.../fundus_table_extended.csv')

assert len(df) > 30000, f'expected >30K rows, got {len(df)}'
assert df['eid'].nunique() > 25000, f'expected >25K unique eids'

# All image paths readable
bad = df[~df['fundus_image_path'].apply(os.path.exists)]
assert len(bad) == 0, f'{len(bad)} fundus paths missing'

# All CMR zips exist
bad = df[~df['cmr_lax_zip'].apply(os.path.exists)]
assert len(bad) == 0
bad = df[~df['cmr_sax_zip'].apply(os.path.exists)]
assert len(bad) == 0

# Same-eid → same-split invariant
by_eid = df.groupby('eid')['split'].nunique()
assert (by_eid == 1).all(), f'{(by_eid > 1).sum()} EIDs span multiple splits'

# Legacy 1,983 EID split unchanged
legacy = pd.read_csv('.../fundus_table.csv')
legacy_split = legacy.drop_duplicates('eid').set_index('eid')['split']
new_split = df.drop_duplicates('eid').set_index('eid')['split']
overlap = legacy_split.index.intersection(new_split.index)
diff = (legacy_split.loc[overlap] != new_split.loc[overlap]).sum()
assert diff == 0, f'{diff} legacy EIDs changed split'

# pair_type distribution
print(df['pair_type'].value_counts())
print(df.groupby(['fundus_instance', 'cmr_instance']).size())

# visit_interval distribution
print(df['visit_interval_years'].describe())
```

---

## 4. 交付时请同时提供

1. 上述 3 个 CSV
2. Sanity check 脚本的 stdout 日志
3. **行数 / 唯一 EID / split 分布 / instance 配对分布** 四张表
4. 任何边角情况（例如某些 EID 在 field 53 里找不到日期）的数量统计

---

## 5. 当前训练侧进度快照（让你了解这事不急）

- Tier 0 (100 EID overfit): ✅ 完成，证明管线有容量 (train R@1=99%)
- Tier 0 train/val (63/16): ✅ 完成，证明 N=16 太小无法给出泛化信号
- Tier 1 (1,196 EID, 959/115/122 split): **正在启动 CMR 预处理**，不依赖你这份交付
- Tier 2 (5K) / Tier 3 (25K)：**阻塞于本次数据工程**。我们这边继续推 Tier 1，
  你这边产出后我们直接切到 25K full training。

---

## 6. 任何疑问优先问这些

- 如果 fundus_image_path 的命名格式不是 `{eid}_21015_{instance}_{eye}.png`，先
  列出其他命名 pattern 给我看一下
- 如果 `eye` 的 0/1 分别代表 left/right 不确定，参考 `contrastive_pretrain/preprocessed_data/modeling_delivery/fundus_table.csv` 已有的映射
- field 53（visit date）的 schema 位置不明确，参考 `contrastive_pretrain/field.tsv`
