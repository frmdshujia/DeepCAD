# Data manifest schemas

All manifests are UTF-8 CSV files. Every `--manifest` option accepts either one
combined CSV or multiple split-specific CSVs separated by spaces. Paths may be
absolute or relative to their source CSV file. Participant identifiers are used
only to enforce grouping and disjointness; they must not encode outcome labels.

## CMR teacher manifest

Required columns:

- `eid`: participant identifier.
- `split`: `train`, `val`, or `test`.
- `cmr_path`: NumPy file with shape `(16, 224, 224)` and values in `[0, 1]`.
- `t1_available`: Boolean observation flag; do not infer it from a label.
- one column per classification or regression target named in the YAML/command
  line. The primary manuscript protocol requires five classification columns
  (`prevalent_I21`, `prevalent_I25`, `history_revasc`,
  `composite_cardiomyopathy_hf`, and `prevalent_I48`) and the 22 regression
  columns listed in `configs/cmr_teacher.yaml`.

The 16-frame order is fixed: six LAX cine frames (2Ch and 4Ch, each at ED,
mid-systole and ES), nine SAX cine frames (base, mid and apex at the same three
phases), followed by native T1. Missing targets may be blank/NaN and are masked.

## Stage I retinal–CMR manifest

Required columns: `eid`, `split`, `t1_available`, and `fundus_path` (the
historical name `fundus_image_path` is accepted as an alias). Multiple retinal
images may belong to one participant. The participant sampler selects one image
per person per training epoch so that another image from the same person cannot
become an InfoNCE negative. Validation and test use one photograph per person,
selected deterministically without reference to the outcome: shortest absolute
`visit_interval_years`, then lowest `fundus_instance`, then resolved image path.
No bilateral or multi-image averaging is performed. CMR embeddings are supplied
separately as an `(N, 768)` `.npy` file and an aligned `(N,)` participant-ID
`.npy` file.

If `--external-validation-manifest` is passed, it must contain an `eid` column.
Training aborts if any identifier occurs in both manifests.

## Stage II manifest

Required columns: `eid`, `split`, `fundus_path`, and binary `label`. Training
samples one photograph per participant per epoch. Validation and test use one
deterministically selected photograph per participant according to the same
label-independent hierarchy used in Stage I; probabilities are not averaged
across eyes or repeated acquisitions.

## Stage III manifest

Required columns: `eid`, `split`, binary `label`, and the columns supplied to
`--feature-columns`. Defaults are `retinal_score`, `age`, `sex`, `hypertension`,
`diabetes`, `smoking`, `dyslipidemia`, and `family_history`. Missing values are
median-imputed using the training split only. Means and standard deviations are
also fitted on the training split only and stored in the checkpoint.
