# Model-weight release slots

This directory defines the stable filenames used by the public DeepCAD release.
No empty or randomly initialized checkpoint is provided: such a file could be
mistaken for a trained model.

Expected DeepCAD checkpoints:

- `deepcad_cmr_teacher_v2.pt`: selected CMR teacher checkpoint.
- `deepcad_stage1_retinal_encoder_v2.pt`: retinal encoder after Stage I
  retinal–CMR alignment.
- `deepcad_stage2_sdpp_v2.pt`: selected checkpoint after the SDPP fine-tuning
  step.
- `deepcad_stage2_shcc_v2.pt`: selected checkpoint after the SHCC fine-tuning
  step.
- `deepcad_stage3_fusion_v2.pt`: selected clinical-fusion checkpoint.

Large checkpoints should be distributed as GitHub Release assets or with Git
LFS rather than ordinary Git blobs. After publication, copy
`manifest.example.json` to `manifest.json` and replace every `PENDING` value
with the final release URL, byte size, and SHA256 checksum.

The following initialization checkpoints are third-party dependencies and are
not redistributed by this repository:

- `MedSAM2_US_Heart.pt`
- `RETFound_cfp_weights.pth`

Users should download those files from the respective upstream distributors and
pass their local paths with `--backbone-checkpoint` and
`--retfound-checkpoint`.
