# DeepCAD: Cross-Modal Contrastive Learning Between Retinal Fundus and Cardiac MRI

DeepCAD is a cross-modal contrastive learning project that learns a **shared latent space** between retinal fundus photographs and cardiac MRI.  
Stage I focuses on **supervised cross-modal contrastive pretraining** to align retina and CMR at the **subject / label / grade** level, and to support cross-modal interpretability (e.g., Grad-CAM on retina aligned with cardiac MRI features).

---
## Online Demo
A web-based demo for testing the model is available at: http://1.13.171.98:8080/


## Project Structure

The full project layout is described in detail in [`PROJECT_STRUCTURE.md`](PROJECT_STRUCTURE.md).  
At a high level:

- `datasets/`: paired retina–CMR dataset and transforms
- `models/`: retina encoder (RETFound), CMR encoder (MedSAM), projection heads, and fusion logic
- `losses/`: supervised cross-modal contrastive loss
- `trainers/`: Stage I trainer (training loop, logging, checkpointing)
- `scripts/`: entry scripts (e.g. `train_stage1.py`)
- `utils/`: checkpointing, logging, and memory bank (feature queue)

---

## Key Features (Current Implementation)

- **Dual Encoders**
  - Retina: **RETFound** ViT-Large MAE backbone.
  - Cardiac MRI: **MedSAM** ViT-Base backbone with multi-slice aggregation.

- **Flexible, subject-centric Dataset**
  - Configurable CSV column names via CLI (no hard-coded headers):
    - `subject_column`, `retinal_column`, `short_axis_npy_column`, `t1map_npy_column`, `label_column`, `grade_column`, etc.
  - Supports **multi-view fundus per subject**:
    - Fundus paths can be a list-like string (e.g. `"['f1.png', 'f2.png']"`).
    - Uniform sampling / duplication to a fixed `max_retinal_images` and pooled inside the model.
  - Supports **multi-sequence CMR (short-axis + T1MAP)**:
    - Two npy columns with potentially different original shapes are loaded and concatenated slice-wise.
    - A fixed number of slices is enforced via `max_mri_slices` and internal slicing / duplication.

- **Unified Cross-Modal Contrastive Loss**
  - Supports multiple **training modes**:
    - `subject`: positives = same subject ID.
    - `grade`: positives = same grade / label.
    - `mixed`, `mixed_clinical`: placeholders for more complex clinical keys.
  - Internally uses a **unified positive key (`pos_keys`)**:
    - For `subject` mode: derived from `subject_id` (globally stable integer index).
    - For other modes: derived from `label` (and can be extended to composite keys).
  - Loss is symmetric: **CMR→Retina** and **Retina→CMR** with InfoNCE-style objectives.

- **ID-aware Memory Bank / Feature Queue (optional)**
  - A **FIFO feature queue** stores `(embedding, pos_key)` pairs:
    - Both retina and CMR sides maintain their own queues.
    - Queue size is configurable via `--queue_size`.
  - During loss computation:
    - **Extra negatives** come from the queue (previous batches).
    - Queue entries whose `pos_key` matches the current anchor are **masked out**, so
      historical positives are **not** treated as false negatives.
  - Works uniformly for `subject` and `label/grade` modes via the shared `pos_keys`.

- **Training Stability & Efficiency**
  - **Mixed precision (AMP)** training support via `torch.cuda.amp`.
  - **Gradient accumulation** via `--grad_accum_steps`, to simulate large effective batch size under limited GPU memory.
  - Option to **freeze encoders** (`--freeze_encoders`) and only train projection heads.

---

## Installation

### 1. Create environment and install dependencies

```bash
cd DeepCAD
pip install -r requirements.txt
```

You need a GPU environment with a recent version of PyTorch and CUDA.  
Pretrained weights for RETFound and MedSAM should be downloaded or placed under your own checkpoint directory.

---

## Data Preparation (Stage I)

DeepCAD Stage I expects a **subject-level CSV** describing paired retina and CMR samples.  
Typical columns (actual names are configurable via CLI):

- `subject_id` : subject identifier
- `fundus_images` : retinal image paths (single path or Python-list-like string)
- `short_axis_ED_mid_ES_npy` : npy file path for short-axis CMR sequence
- `T1MAP_B1B2B3_npy` : npy file path for T1MAP sequence
- `label` (optional): CAD / CHD label (0/1 or multi-grade)
- `grade` (optional): disease severity

You can preprocess and split your CMR table into train/val/test CSVs under e.g.:

- `data/processed/train_val_test/cmr_T1MAP_shortaxis_npy_train.csv`
- `data/processed/train_val_test/cmr_T1MAP_shortaxis_npy_val.csv`
- `data/processed/train_val_test/cmr_T1MAP_shortaxis_npy_test.csv`

Retinal and CMR base paths (for resolving relative image / npy paths) are passed via:

- `--retinal_base_path`
- `--mri_base_path`

---

## Training Stage I (CLI)

The main script is `scripts/train_stage1.py`, which directly takes CLI arguments instead of a YAML config.

### Basic subject-level contrastive training example

```bash
export ROOT=/path/to/DeepCAD

python $ROOT/scripts/train_stage1.py \
  --train_csv  $ROOT/data/processed/train_val_test/cmr_T1MAP_shortaxis_npy_train.csv \
  --val_csv    $ROOT/data/processed/train_val_test/cmr_T1MAP_shortaxis_npy_val.csv \
  --test_csv   $ROOT/data/processed/train_val_test/cmr_T1MAP_shortaxis_npy_test.csv \
  --subject_column subject_id \
  --retinal_column fundus_images \
  --short_axis_npy_column short_axis_ED_mid_ES_npy \
  --t1map_npy_column      T1MAP_B1B2B3_npy \
  --retinal_base_path $ROOT/data/processed/contrastive_learning/fundus \
  --mri_base_path     $ROOT/data/processed/contrastive_learning/cmr \
  --retinal_pretrained $ROOT/checkpoints/pretrained/retfound/RETFound_cfp_weights.pth \
  --mri_pretrained     $ROOT/checkpoints/pretrained/medsam/medsam_vit_b.pth \
  --training_mode subject \
  --freeze_encoders \
  --batch_size 4 \
  --grad_accum_steps 4 \
  --max_mri_slices 6 \
  --mri_pooling_type attention \
  --retinal_augmentation medium \
  --mri_augmentation medium \
  --use_queue \
  --queue_size 1024 \
  --num_epochs 50 \
  --save_dir $ROOT/checkpoints/stage1_subject \
  --log_dir  $ROOT/logs/stage1_subject
```

### Important arguments (partial list)

- **Data & columns**
  - `--train_csv`, `--val_csv`, `--test_csv` : CSV paths.
  - `--subject_column` : subject ID column name.
  - `--retinal_column` : fundus image paths column (single or list-like).
  - `--short_axis_npy_column`, `--t1map_npy_column` : CMR npy column names.
  - `--label_column`, `--grade_column` : label / grade column names (optional).

- **Model & encoders**
  - `--retinal_pretrained` : RETFound checkpoint path.
  - `--mri_pretrained` : MedSAM checkpoint path.
  - `--retinal_img_size`, `--mri_img_size` : input sizes (MedSAM internally upsamples to 1024×1024).
  - `--max_mri_slices` : max number of MRI slices per subject.
  - `--mri_pooling_type` : MRI slice aggregation strategy (`attention`, `learnable_weighted`, `mean`, `max`).
  - `--latent_dim` : shared embedding dimension (default 128).

- **Training**
  - `--training_mode` : `subject` / `grade` / `mixed` / `mixed_clinical`.
  - `--batch_size` : per-step batch size (before gradient accumulation).
  - `--grad_accum_steps` : gradient accumulation steps; effective batch size ≈ `batch_size × grad_accum_steps`.
  - `--lr`, `--weight_decay`, `--optimizer`, `--scheduler` : standard optimizer/scheduler options.
  - `--freeze_encoders` : freeze RETFound & MedSAM, only train projection heads.
  - `--retinal_augmentation`, `--mri_augmentation` : augmentation strength (`light`, `medium`, `strong`).

- **Memory bank / queue (optional)**
  - `--use_queue` : enable ID-aware memory bank (feature queue).
  - `--queue_size` : queue capacity (typical range 512–2048 for small cohorts).

- **Logging & checkpoints**
  - `--save_dir` : checkpoint directory.
  - `--log_dir` : logging directory (supports scalar logging via custom logger).
  - `--save_interval` : save checkpoint every N epochs.
  - `--resume_from` : path to resume from an existing checkpoint.

---

## Design Notes

- **Unified positive key (`pos_keys`)**
  - All training modes eventually define a **1D key vector** per batch (e.g. subject ID index, label, grade).
  - Positive pairs are defined by equality of keys: `pos_keys[i] == pos_keys[j]`.
  - This unifies subject-level and label-level contrastive learning in a single loss implementation.

- **ID-aware queue for stable contrast**
  - Each queue entry stores `(embedding, pos_key)`.
  - When using the queue to extend negatives:
    - For each anchor, queue entries with the **same key** are **masked out** from the denominator.
    - This avoids “past self” or same-label samples being treated as false negatives.

- **Multi-view retina & multi-sequence CMR**
  - Retina: multiple fundus images per subject are encoded independently and pooled (mean) to a subject-level embedding.
  - CMR: short-axis and T1MAP sequences are loaded from separate npy columns, normalized, concatenated, and pooled by the MRI encoder.

---

## Citing & Related Work

DeepCAD builds on ideas from the following projects:

- **MMCL-Tabular-Imaging** – multimodal contrastive learning framework and training skeleton.
- **RETFound** – foundation model for retinal images (ViT-Large MAE encoder).
- **MedSAM** – foundation model for medical image segmentation and representation (ViT-Base encoder).

If you use this repository in your research, please also cite the corresponding original works.  
A dedicated citation section for DeepCAD will be added once a manuscript is available.
