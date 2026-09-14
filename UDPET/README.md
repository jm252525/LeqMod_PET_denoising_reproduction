# UDPET baseline integration

This directory versions the UDPET adaptation that was previously stored only
under `/mnt/sdb/jinming.hu/UDPET/step5_env/code`.

## Snapshot scope

- Snapshot date: 2026-09-15 (Asia/Shanghai)
- Upstream base commit: `2ec976a0378e06b6cd115b940f47272cc25c98a5`
- The seven Python files in `code/` are byte-for-byte copies of the server
  adaptation at snapshot time.
- Patient images, manifests, smoke-test outputs, and model artifacts are not
  stored in Git. Their aggregate counts and SHA-256 hashes are recorded in
  `configs/server_snapshot_20260915.json`.

## Current experimental status

- Patient-level train/validation/test manifests exist and their patient sets
  are disjoint.
- Loader-only smoke tests have run for all three cohorts.
- QuMod is enabled by the all-cohort dry-run configuration.
- LeMod is disabled because the current manifests do not provide lesion
  segmentation paths. This snapshot is therefore not a complete LeqMod run.
- No formal training checkpoint or training-loss file existed at snapshot
  time.

## Known launch blocker

The copied training code passes `verbose=True` to PyTorch learning-rate
schedulers. PyTorch 2.7.0 no longer accepts that argument, so formal training
must not be launched from this snapshot until the scheduler compatibility is
fixed and an optimizer-step smoke test passes.

## Layout

- `code/`: unchanged UDPET adaptation snapshot
- `configs/server_snapshot_20260915.json`: sanitized provenance, environment,
  dataset counts, and source hashes
