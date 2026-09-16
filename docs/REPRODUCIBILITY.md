# Reproducibility and leakage safeguards

- Cohort assignment is read from a manifest and is never generated inside a
  training script. Split creation should be participant-level and completed
  before model development.
- `train`, `val`, and `test` have distinct semantics. Optimization uses `train`;
  checkpoint and threshold selection use `val`; final metrics use `test` only.
- Stage I may receive a clean external-validation manifest. Any participant
  overlap causes an immediate error.
- Stage I training uses one retinal image per participant per epoch. This
  prevents a second eye from the same participant being treated as an in-batch
  negative. Validation and test aggregate all eyes within EID before metrics.
- The CMR teacher is frozen before embeddings are extracted for Stage I. The
  released Stage I code never updates the teacher.
- Missing native-T1 is represented by an explicit observation mask. Missing-T1
  placeholders bypass the image backbone and cine–T1 cross-attention, and their
  global token positions are masked.
- CMR classification positive weights and regression normalization statistics
  are fitted on `train` only. The primary CMR checkpoint is selected by minimum
  validation total loss, not by a test metric.
- Final test evaluators require an explicit acknowledgement flag and reject an
  existing output path. Preserve that output together with the manifest and
  checkpoint hashes; do not repeatedly inspect test results during development.
- Every output directory should include the architecture, cohort, seed, and run
  version in its name. Scripts reject non-empty output directories unless
  `--allow-existing` is deliberately supplied.
- Report stochastic experiments over prespecified seeds and retain the complete
  run configuration saved next to each checkpoint.

Before release, public manifests must contain neither real participant IDs nor
private filesystem paths. Restricted images, labels, and weights must not be
committed to the repository.
