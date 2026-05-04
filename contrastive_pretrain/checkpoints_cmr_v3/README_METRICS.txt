CMR v3 checkpoints — metrics truth source
=========================================

Authoritative numbers: metrics_verified.json (recomputed 2026-05-03).

Do NOT cite mean_AUC ~0.85 on validation — that came from the deprecated training-time log and did not survive independent recalculation (~0.685 on task1_cmr_val).

Files:
  metrics_verified.json ............ trustworthy recomputed train / val / partial-test + notes
  metrics_history_DEPRECATED_untrusted_20260430.json ... old per-epoch curve (misleading), kept only for archaeology
  last_checkpoint_history_DEPRECATED_untrusted_20260430.json ... exported from last.pth history before purge

metrics_history.json was removed intentionally so tooling does not silently read inflated curves.

Restore training curve only if needed for debugging (see DEPRECATED JSON above).
