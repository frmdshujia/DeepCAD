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
computed separately with `evaluate_cmr_teacher.py`, `evaluate_stage2.py`, and
`evaluate_stage3.py`.

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
The model bypasses cine–T1 cross-attention for those participants, zeros the T1
stream, and masks the 16 T1 positions in the global Transformer. It does not
treat an all-zero placeholder as observed tissue information.

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

## Minimal commands

The following examples use versioned output directories to prevent accidental
overwriting. Replace the placeholder paths with files matching
[`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md).

```bash
python train_cmr_teacher.py \
  --manifest manifests/cmr_teacher_train.csv manifests/cmr_teacher_val.csv manifests/cmr_teacher_test.csv \
  --backbone-checkpoint weights/MedSAM2_US_Heart.pt \
  --classification-columns prevalent_I21,prevalent_I25,composite_cardiomyopathy_hf,prevalent_I48 \
  --regression-columns LVEF,LVEDV,LVESV,LVSV,LV_CO,LVM,LV_WT,GLS,GCS,GRS,RVEF,RVEDV,RVESV,RVSV,LAEF,LA_max,LA_min,LASV,RAEF,RA_max,RA_min,RASV \
  --fusion-mode hierarchical \
  --output-dir outputs/cmr_teacher_hierarchical_seed42

python extract_cmr_embeddings.py \
  --manifest manifests/stage1_train_participants.csv manifests/stage1_val_participants.csv manifests/stage1_test_participants.csv \
  --checkpoint outputs/cmr_teacher_hierarchical_seed42/best.pt \
  --output-dir outputs/stage1_teacher_embeddings_seed42

python train_stage1_contrastive.py \
  --manifest manifests/stage1_train_images.csv manifests/stage1_val_images.csv manifests/stage1_test_images.csv \
  --cmr-embeddings outputs/stage1_teacher_embeddings_seed42/cmr_teacher_embeddings.npy \
  --cmr-eids outputs/stage1_teacher_embeddings_seed42/cmr_teacher_eids.npy \
  --retfound-checkpoint weights/RETFound_cfp_weights.pth \
  --external-validation-manifest manifests/external_validation_clean.csv \
  --output-dir outputs/stage1_infonce_seed42

python train_stage2_supervised.py \
  --manifest manifests/sdpp.csv \
  --initial-checkpoint outputs/stage1_infonce_seed42/best.pt \
  --output-dir outputs/stage2_sdpp_seed42

python train_stage2_supervised.py \
  --manifest manifests/shcc.csv \
  --initial-checkpoint outputs/stage2_sdpp_seed42/best.pt \
  --output-dir outputs/stage2_shcc_seed42

python train_stage3_fusion.py \
  --manifest manifests/stage3.csv \
  --output-dir outputs/stage3_clinical_fusion_seed42
```

## Verification

```bash
python -m compileall -q deepcad *.py
python tests/smoke_test.py
pytest -q
```

An optional real-backbone smoke test is available at
`tests/real_backbone_smoke.py`; it requires a compatible Hiera checkpoint and
one de-identified `(16, 224, 224)` CMR array. The optional
`tests/fundus_encoder_smoke.py` checks a retinal forward/backward pass.

See [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for split, checkpoint,
and reporting safeguards. This code release does not grant access to restricted
cohort data.
