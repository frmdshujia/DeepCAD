# DeepCAD: reviewer code release

This repository contains the model definitions and training entry points used by
DeepCAD. It is deliberately manifest-driven: private images, participant
identifiers, labels, pretrained weights, and institution-specific paths are not
included.

## Pipeline

1. **CMR teacher development** (`train_cmr_teacher.py`): a shared Hiera encoder
   tokenizes 6 LAX cine frames, 9 SAX cine frames, and one native-T1 frame.
2. **Stage I cross-modal alignment** (`train_stage1_contrastive.py`): RETFound
   retinal representations are aligned with frozen 768-dimensional CMR teacher
   embeddings using symmetric InfoNCE.
3. **Stage II supervised learning** (`train_stage2_supervised.py`): the Stage I
   retinal encoder is fine-tuned for CAD classification. The same entry point is
   used for sequential development cohorts by initializing the second run from
   the selected checkpoint of the first run.
4. **Stage III clinical fusion** (`train_stage3_fusion.py`): retinal risk scores
   are combined with prespecified clinical risk factors using an MLP.

The held-out test split is not used by any training entry point. Model selection
and operating thresholds are based on the validation split. Test performance is
computed separately with `evaluate_cmr_teacher.py`,
`evaluate_stage1_alignment.py`, `evaluate_stage2.py`, and `evaluate_stage3.py`.
Test evaluators require an explicit `--acknowledge-final-test` flag and refuse
to overwrite an existing result file.

## CMR cross-attention implemented here

The primary `hierarchical` fusion mode performs four explicit attention calls:

```text
LAX'  = LAX  + Attention(Q=LAX,  K=SAX,  V=SAX)
SAX'  = SAX  + Attention(Q=SAX,  K=LAX,  V=LAX)
Cine  = Concat(LAX', SAX')
Cine' = Cine + Attention(Q=Cine, K=T1,   V=T1)
T1'   = T1   + Attention(Q=T1,   K=Cine, V=Cine)
```

With the default 4×4 spatial pooling, the streams entering cross-attention have
96 LAX tokens, 144 SAX tokens, and 16 T1 tokens. The final global Transformer
receives `[CLS]`, the 256 frame tokens, and 80 ED-to-ES difference tokens (337
tokens in total). The `sax_t1` and `global_only` modes provide controlled
architecture ablations.

For participants without observed T1, the manifest sets `t1_available=false`.
The model does not send their T1 placeholder through the image backbone,
bypasses cine–T1 cross-attention for those participants, zeros the T1 stream,
and masks the 16 T1 positions in the global Transformer. It does not treat an
all-zero placeholder as observed tissue information.

The primary CMR objective is the equally weighted mean of five task-wise focal
losses and the mean of 22 task-wise MSE losses. The focal focusing parameter is
2.0; task-specific class weights and regression z-score statistics are
estimated from the training split only. The Hiera backbone is frozen for three
epochs, then unfrozen with gradient checkpointing. The manuscript-locked
optimization schedule is recorded in [`configs/cmr_teacher.yaml`](configs/cmr_teacher.yaml).
The selected checkpoint minimizes validation total loss.

## Manuscript-locked configurations

All training entry points accept a flat YAML file through `--config`; explicit
command-line arguments override YAML values. The primary configurations are in
[`configs/`](configs/README.md). In particular, Stage I uses a fixed temperature
of 0.1, fine-tunes the last 12 RETFound blocks, and applies a 1:10 learning-rate
ratio between the pretrained backbone and newly introduced modules. Stage II
uses focal loss (`alpha=0.25`, `gamma=2.0`); the SHCC step additionally fits a
single temperature on the validation split and stores it in the selected
checkpoint.

The five-task CMR configuration requires a `history_revasc` column in addition
to I21, I25, CM/HF, and I48. A four-classification `complete26` manifest is not
compatible with the manuscript's five-classification + 22-regression protocol.
See [`docs/METHODS_CODE_ALIGNMENT.md`](docs/METHODS_CODE_ALIGNMENT.md) for the
claim-by-claim mapping from the Methods section to implementation and YAML.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .[test]
```

The CMR encoder additionally requires the official SAM 2 Python package. Install
it from its upstream repository in the same environment. Pretrained RETFound and
MedSAM2-US-Heart weights are supplied by their respective distributors and are
passed explicitly on the command line; they are not redistributed here.

## Model weights

The repository reserves stable filenames under [`weights/`](weights/README.md)
for the DeepCAD checkpoints that will accompany the final release. The directory
currently contains metadata placeholders only; it does not contain dummy model
files. When the final checkpoints are available, publish them as GitHub Release
assets or through Git LFS, then replace the `PENDING` entries in
`weights/manifest.example.json` with immutable download URLs, file sizes, and
SHA256 checksums. Third-party RETFound and MedSAM2-US-Heart checkpoints must be
obtained from their original distributors.

## Minimal commands

The following examples use versioned output directories to prevent accidental
overwriting. Replace the placeholder paths with files matching
[`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

```bash
python train_cmr_teacher.py \
  --config configs/cmr_teacher.yaml \
  --manifest manifests/cmr_teacher_train.csv manifests/cmr_teacher_val.csv manifests/cmr_teacher_test.csv \
  --backbone-checkpoint weights/MedSAM2_US_Heart.pt \
  --output-dir outputs/cmr_teacher_hierarchical_seed42

python extract_cmr_embeddings.py \
  --manifest manifests/stage1_train_participants.csv manifests/stage1_val_participants.csv manifests/stage1_test_participants.csv \
  --checkpoint outputs/cmr_teacher_hierarchical_seed42/best.pt \
  --output-dir outputs/stage1_teacher_embeddings_seed42

python train_stage1_contrastive.py \
  --config configs/stage1_alignment.yaml \
  --manifest manifests/stage1_train_images.csv manifests/stage1_val_images.csv manifests/stage1_test_images.csv \
  --cmr-embeddings outputs/stage1_teacher_embeddings_seed42/cmr_teacher_embeddings.npy \
  --cmr-eids outputs/stage1_teacher_embeddings_seed42/cmr_teacher_eids.npy \
  --retfound-checkpoint weights/RETFound_cfp_weights.pth \
  --external-validation-manifest manifests/external_validation_clean.csv \
  --output-dir outputs/stage1_infonce_seed42

python evaluate_stage1_alignment.py \
  --manifest manifests/stage1_test_images.csv \
  --checkpoint outputs/stage1_infonce_seed42/best.pt \
  --cmr-embeddings outputs/stage1_teacher_embeddings_seed42/cmr_teacher_embeddings.npy \
  --cmr-eids outputs/stage1_teacher_embeddings_seed42/cmr_teacher_eids.npy \
  --split test --acknowledge-final-test \
  --output-json outputs/stage1_infonce_seed42/test_metrics.json

python train_stage2_supervised.py \
  --config configs/stage2_sdpp.yaml \
  --manifest manifests/sdpp.csv \
  --initial-checkpoint outputs/stage1_infonce_seed42/best.pt \
  --output-dir outputs/stage2_sdpp_seed42

python train_stage2_supervised.py \
  --config configs/stage2_shcc.yaml \
  --manifest manifests/shcc.csv \
  --initial-checkpoint outputs/stage2_sdpp_seed42/best.pt \
  --output-dir outputs/stage2_shcc_seed42

python train_stage3_fusion.py \
  --config configs/stage3_fusion.yaml \
  --manifest manifests/stage3.csv \
  --output-dir outputs/stage3_clinical_fusion_seed42
```

## Verification

```bash
python -m compileall -q deepcad *.py
python tests/smoke_test.py
pytest -q
python verify_manifests.py --checksum-file manifests/SHA256SUMS.txt
```

Stage I optimization samples one retinal photograph per participant per epoch.
Validation and test use one deterministically selected photograph per
participant; no bilateral or multi-image averaging is performed. The optional
`--require-t1` switch implements the T1-complete sensitivity analysis, while
the evaluator reports T1-present and T1-missing subgroups separately.

An optional real-backbone smoke test is available at
`tests/real_backbone_smoke.py`; it requires a compatible Hiera checkpoint and
one de-identified `(16, 224, 224)` CMR array. The optional
`tests/fundus_encoder_smoke.py` checks a retinal forward/backward pass.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for split, checkpoint,
and reporting safeguards. This code release does not grant access to restricted
cohort data.
