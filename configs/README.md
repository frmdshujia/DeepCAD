# Manuscript-locked training configurations

These flat YAML files are consumed by the corresponding training entry points
through `--config`. Command-line options override YAML values. Private paths,
participant identifiers, and checkpoint download locations are intentionally
excluded and must be supplied at runtime.

- `cmr_teacher.yaml`: primary 5-classification + 22-regression cardiac MRI
  teacher with hierarchical bidirectional cross-attention.
- `stage1_alignment.yaml`: retinal-CMR symmetric InfoNCE alignment.
- `stage2_sdpp.yaml`: first supervised retinal fine-tuning step.
- `stage2_shcc.yaml`: angiography-confirmed second fine-tuning step and
  validation-set temperature scaling.
- `stage3_fusion.yaml`: retinal-score and clinical-risk-factor MLP.

These files define the protocol that must be used to produce the final reported
checkpoints. In particular, Stage I uses a fixed temperature of 0.1, rather
than a learnable temperature, and a 1:10 learning-rate ratio between the
trainable RETFound blocks and newly introduced modules. Every training run also
writes the fully resolved values to `run_config.json` beside its checkpoints.
