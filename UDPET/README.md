# UDPET baseline integration

This directory versions the UDPET adaptation that was previously stored only
under `/mnt/sdb/jinming.hu/UDPET/step5_env/code`.

## Snapshot scope

- Snapshot date: 2026-09-15 (Asia/Shanghai)
- Upstream base commit: `2ec976a0378e06b6cd115b940f47272cc25c98a5`
- The initial integration commit preserved the seven server-side Python files
  byte-for-byte. The follow-up loader optimization is versioned in this branch.
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

## Loader optimization

The default loader now combines two bounded optimizations:

- `volume_grouped_weighted` first performs the same weighted, with-replacement
  row draw as the legacy sampler and then reorders the sampled rows so that all
  DRFs sharing one NORMAL reference are contiguous. The sampled row multiset
  and expected cohort/DRF probabilities are unchanged; only order changes.
- Each DataLoader worker keeps an LRU cache of cropped NORMAL data, affine,
  crop coordinates, valid patch boxes, and (when available) segmentation. The
  default cache size is one reference per worker to avoid unbounded RAM use.
  Low-count data is still read once for every sampled row.

Persistent workers are enabled by default when `num_workers > 0`, so the
process-local cache survives epoch boundaries. The legacy behavior remains
available by combining `--sampling-mode csv_weighted`,
`--reference-cache-size 0`, and `--no-persistent-workers`.

For the fixed train manifest, seed `20260910`, 5,020 epoch samples, two workers,
and a one-reference cache, an index-only locality audit estimates 1,824 NORMAL
loads per epoch versus 5,009 with the legacy random order (about 64% fewer).
This is a predicted decompression count, not a wall-clock benchmark; storage
throughput and worker scheduling still need measurement before a long run.

Example loader-only smoke test (no optimizer steps are run without
`--run-training`):

```bash
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/train_LeqModGan_csv.py \
  --train-csv /mnt/sdb/jinming.hu/UDPET/metrics_step4/input/train.csv \
  --output-path /mnt/sdb/jinming.hu/UDPET/runs \
  --experiment-name loader_v2_smoke \
  --sampling-mode volume_grouped_weighted \
  --reference-cache-size 1 \
  --num-workers 2
```

Regression test:

```bash
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/test_dataset_cache.py
```

The two `pydeps` directories are pip target directories joined through
`PYTHONPATH`; they are not two activated virtual environments. The interpreter
is the system `/usr/bin/python3`. The first path supplies PyTorch/CUDA packages;
the second supplies NumPy/SciPy/nibabel/scikit-image and related metrics
packages.

## Known launch blocker

The copied training code passes `verbose=True` to PyTorch learning-rate
schedulers. PyTorch 2.7.0 no longer accepts that argument, so formal training
must not be launched from this snapshot until the scheduler compatibility is
fixed and an optimizer-step smoke test passes.

## Layout

- `code/`: UDPET adaptation plus the versioned loader optimization and tests
- `configs/server_snapshot_20260915.json`: sanitized provenance, environment,
  dataset counts, and source hashes
