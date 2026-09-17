# Methods-to-code alignment

This file identifies the public implementation corresponding to each model-
development statement. The YAML files under `configs/` are canonical; command-
line overrides are recorded in each run's `run_config.json`.

## Cardiac MRI teacher

- Input and frame order: `deepcad/data.py` and the constants at the top of
  `deepcad/models/cmr_encoder.py` implement six LAX cine frames, nine SAX cine
  frames, and one native-T1 map.
- Architecture: `CMREncoderV4` uses the shared MedSAM2-US-Heart-compatible
  Hiera-tiny backbone. `BidirectionalCrossAttention` performs two explicit
  Q/K/V calls; `HierarchicalCineT1Fusion` applies LAX↔SAX before fused-cine↔T1.
- Missing T1: `CMREncoderV4._tokenise` routes only observed T1 maps through the
  backbone, skips cine–T1 attention for missing-T1 participants, zeros the T1
  stream, and masks its 16 global-token positions.
- Targets, loss, optimization, freezing, augmentation, and stopping:
  `configs/cmr_teacher.yaml`, `train_cmr_teacher.py`, and `deepcad/losses.py`.
  The primary protocol is five classification plus 22 regression tasks.
- Selection and evaluation: minimum validation total loss selects `best.pt`;
  `evaluate_cmr_teacher.py` is the separate final evaluator.

## Stage I retinal–CMR alignment

- Frozen teacher: `extract_cmr_embeddings.py` exports the 768-dimensional CMR
  representation before Stage I; `train_stage1_contrastive.py` only loads these
  frozen arrays.
- Retinal encoder: `deepcad/models/fundus_encoder.py` implements RETFound ViT-L
  with an adapter after each of 24 blocks. The first 12 blocks remain frozen and
  the last 12 blocks are trainable under `configs/stage1_alignment.yaml`.
- Projection and objective: separate 1024→512→128 retinal and 768→512→128 CMR
  heads feed symmetric InfoNCE at fixed temperature 0.1.
- Image policy: training samples one photograph per participant per epoch;
  validation and test use one deterministic, outcome-independent photograph
  per participant. No bilateral averaging is performed.
- Selection: minimum validation InfoNCE selects `best.pt`.

## Stage II supervised retinal development

- `configs/stage2_sdpp.yaml` and `configs/stage2_shcc.yaml` define the sequential
  SDPP then SHCC runs. The selected SDPP checkpoint initializes SHCC.
- Both steps use focal loss with alpha 0.25 and gamma 2.0 and keep the last 12
  RETFound blocks plus adapters trainable. Validation AUROC selects checkpoints.
- The SHCC run fits one scalar calibration temperature on its validation split;
  `evaluate_stage2.py` applies that frozen value.
- Stage II uses the same one-image-per-participant protocol as Stage I.

## Stage III clinical fusion

- `configs/stage3_fusion.yaml` and `train_stage3_fusion.py` define the MLP over
  retinal score, age, sex, hypertension, diabetes, smoking, dyslipidemia, and
  family history.
- Imputation and standardization are fitted on training data only. Validation
  AUROC selects the model. Validation data alone determine cutoffs satisfying
  sensitivity ≥0.90 and specificity ≥0.95; `evaluate_stage3.py` freezes and
  applies those cutoffs to test or external cohorts.

## Required preflight condition

The primary CMR YAML requires `history_revasc` in addition to I21, I25, CM/HF,
and I48. A four-classification/22-regression manifest cannot run the five-class
manuscript protocol. Training intentionally fails at schema validation until a
five-classification/22-regression manifest is supplied.
