# Release checklist

- [ ] Manuscript terminology matches the CMR-teacher step and Stages I–III.
- [ ] Figure S6 Q/K/V arrows match `HierarchicalCineT1Fusion`.
- [ ] The final Stage I and external-validation manifests are disjoint.
- [ ] Exact cohort counts and exclusions are documented in the manuscript.
- [ ] The CMR teacher manifest contains all five classification targets,
      including `history_revasc`, plus all 22 regression targets.
- [ ] The resolved `run_config.json` values match the canonical YAML and the
      final Methods text for every released checkpoint.
- [ ] Validation-derived thresholds and test-derived performance are identified.
- [ ] No data, real IDs, absolute private paths, secrets, or large checkpoints.
- [ ] All retained experimental checkpoints have an immutable provenance record.
- [ ] `python -m compileall -q deepcad *.py` and `pytest -q` pass.
- [ ] Repository visibility is appropriate for the journal's blinding policy.
- [ ] A software license and archival citation are selected before public release.
