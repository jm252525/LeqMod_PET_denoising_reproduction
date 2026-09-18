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
- PyTorch 2.7 scheduler creation, a real eight-patch optimizer step, fixed
  patient-level validation, and an exact interrupted-versus-continuous resume
  comparison have passed. No formal training run has started.
- The complete four-split HDF5 cache has been built and audited. All 1,163
  containers, 7,463 image datasets, and 727,522,124,800 stored voxels decoded
  successfully; cross-center full-volume and model-patch comparisons were
  exactly equal to the NIfTI backend.
- The HDF5 train/validation backend is now bound into the strict resume
  contract by backend identity and complete train/validation index hashes. A
  one-GPU bounded comparison passed exact continuous/resume and HDF5/NIfTI
  state equivalence. HDF5 is the recommended explicit working backend for the
  next baseline experiments; NIfTI remains the archival source of truth.

## Project plan

- [后续计划与课题创新目标差距](docs/PROJECT_ROADMAP_AND_INNOVATION_GAP.md)

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
This remains a predicted decompression count, not evidence of speed. A later
ABBA wall-clock benchmark processed 200 real batches per arm and found the
optimized loader about 0.6% slower in total and 1.2% slower in steady fetch
mean, with essentially unchanged GPU-forward wait. See
`docs/LOADER_BENCHMARK_REPORT_20260917.md`. The grouping/cache behavior is kept
for reproducibility, but is not considered a promoted throughput optimization.

### Experimental HDF5 direct-patch backend

A bounded follow-up stores one full-volume float32 NORMAL array and all
available low-count levels in one chunked HDF5 file per study. The loader reads
only the eight selected low/NORMAL patches instead of materializing complete
gzip NIfTI volumes. Original NIfTI files remain immutable, and the NIfTI backend
remains the default.

On a center-balanced 24-study/135-row train pilot, the `80^3` LZF cache occupied
12.12 GiB versus 6.04 GiB of source NIfTI. Twenty-four augmented deterministic
requests were exactly equal between backends (`max_abs_diff=0`). A four-run
ABBA benchmark with 100 real batches per run found 4.26x median pipeline wall
speedup, 4.95x steady fetch speedup, and HDF5/NIfTI process-tree PSS of 0.357.
One real eight-patch QuMod optimizer step and strict checkpoint round-trip also
passed. This promotes the design to a full-cache candidate, not yet to the
formal default. See `docs/HDF5_PATCH_CACHE_EXPERIMENT_20260917.md`.

The full train/validation/Bern-test/Ruijin-test cache was subsequently built at
`/mnt/sdb/jinming.hu/UDPET/hdf5_cache_v1_full_20260917`. It contains 1,163
study containers and occupies 458.387 GiB versus 228.817 GiB of unique source
NIfTI files. A read-only audit opened every container and dataset, decoded all
727,522,124,800 voxels, and found no non-finite values, negative values, or
coverage/metadata errors. A deterministic two-per-`split x center x DRF`
sample produced 139/139 exact full-volume matches and 72/72 exact augmented
eight-patch matches, with maximum absolute difference 0.0 for low-count,
NORMAL, and weight tensors. See
`docs/FULL_HDF5_CACHE_AUDIT_20260918.md`.

The formal training entry point now accepts `--storage-backend hdf5`, separate
train and validation cache indexes, and a bounded per-worker handle cache. The
backend, both complete index SHA-256 values, and the handle-cache setting are
part of training protocol `udpet_training_v7_storage_bound_resume_20260918`.
Changing backends or index content therefore causes strict resume to fail.

On physical GPU 2, a bounded two-update HDF5 run, an interrupted/resumed HDF5
run, and a matched NIfTI run produced exactly equal model, optimizer, scheduler,
RNG, loss-history, and validation states (apart from the deliberately distinct
HDF5/NIfTI training-contract hash). HDF5 wall time was 12.55 s versus 22.99 s
for NIfTI in this tiny run; this is a training-path check, not a throughput
benchmark. See `docs/HDF5_TRAINING_GATE_20260918.md`.

Re-run the full audit with:

```bash
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/hdf5_cache_env/pydeps:/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/audit_full_hdf5_cache.py \
  --cache-root /mnt/sdb/jinming.hu/UDPET/hdf5_cache_v1_full_20260917 \
  --output-json /mnt/sdb/jinming.hu/UDPET/hdf5_cache_v1_full_20260917/audit/full_audit.json \
  --comparison-per-stratum 2 \
  --patches-per-volume 8 \
  --full-data-scan \
  --scan-workers 4
```

The optional backend requires the isolated dependency in
`configs/hdf5_cache_requirements.txt`. Cache construction is atomic, refuses to
overwrite by default, and supports explicit validated resume with
`--resume-existing`.

Example loader-only smoke test (no optimizer steps are run without
`--run-training`):

```bash
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/train_LeqModGan_csv.py \
  --train-csv /mnt/sdb/jinming.hu/UDPET/metrics_step4/input/train.csv \
  --output-path /mnt/sdb/jinming.hu/UDPET/runs \
  --experiment-name loader_v3_smoke \
  --sampling-mode volume_grouped_weighted \
  --reference-cache-size 1 \
  --num-workers 2
```

HDF5 train/validation runs must name both immutable indexes explicitly:

```bash
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/hdf5_cache_env/pydeps:/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/train_LeqModGan_csv.py \
  --train-csv /mnt/sdb/jinming.hu/UDPET/metrics_step4/input/train.csv \
  --val-csv /mnt/sdb/jinming.hu/UDPET/metrics_step4/input/val.csv \
  --output-path /mnt/sdb/jinming.hu/UDPET/runs \
  --experiment-name hdf5_qumod_example \
  --storage-backend hdf5 \
  --chunk-cache-index /mnt/sdb/jinming.hu/UDPET/hdf5_cache_v1_full_20260917/train/index.json \
  --val-chunk-cache-index /mnt/sdb/jinming.hu/UDPET/hdf5_cache_v1_full_20260917/val/index.json
```

Regression test:

```bash
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/test_dataset_cache.py
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/test_training_state.py
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/test_scheduler_creation.py
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/test_validation_metrics.py
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/hdf5_cache_env/pydeps:/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/test_hdf5_patch_cache.py
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/hdf5_cache_env/pydeps:/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/test_training_storage_contract.py
```

## Training and resume contract

- `--n-epochs N` now executes exactly `N` epochs; the previous inclusive loop
  that executed `N + 1` epochs is removed.
- PyTorch 2.7 schedulers are constructed without the removed `verbose`
  argument.
- Checkpoint version 2 atomically stores raw G/D weights, both optimizers, all
  schedulers, completed epoch, total iterations, Python/NumPy/CPU/CUDA RNG,
  DataLoader generator state, sampler epoch, and a training-contract SHA-256.
- Every sampler emits an explicit per-sample seed. Patch selection and rotation
  therefore do not depend on persistent-worker scheduling and are reproducible
  after an epoch-boundary resume.
- Training uses strict deterministic PyTorch algorithms, the deterministic
  cuBLAS workspace, and disables CUDA matmul/cuDNN TF32. Unsupported
  nondeterministic operators fail instead of silently weakening the contract.
- The checkpointed DataLoader generator is a canonical next-epoch state. It is
  independent of worker-process creation; all scientific sample randomness is
  carried by the sampler request seed.
- Strict resume is the default. It rejects a changed training contract, a
  partial-epoch checkpoint, missing loss history, or a legacy checkpoint.
  `--allow-inexact-resume` is an explicit diagnostic escape hatch and must not
  be used for a formal run.
- `--max-epochs-this-invocation` can deliberately stop a process at an epoch
  boundary without changing `--n-epochs` or the scientific contract. This is
  intended for restart testing and bounded scheduling, not early stopping.

The real-data engineering smoke used one study with eight `80x80x80` patches,
QuMod enabled, and one discriminator plus one generator update on physical
GPU0. It passed with finite losses, 7,166.5 MiB peak allocated and 10,196 MiB
peak reserved GPU memory. A 273.4 MiB temporary checkpoint was reloaded with
identical parameters and restored optimizer, scheduler, CPU/CUDA RNG states;
the temporary checkpoint was deleted. The JSON report is stored outside Git at
`/mnt/sdb/jinming.hu/UDPET/loader_checks/optimizer_step_8patch_20260917/report.json`.

Re-run the bounded smoke test with:

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/mnt/sdb/jinming.hu/UDPET/step5_env/pydeps:/mnt/sdb/jinming.hu/UDPET/metrics_step4/pydeps \
  /usr/bin/python3 UDPET/code/smoke_optimizer_step.py \
  --train-csv /mnt/sdb/jinming.hu/UDPET/metrics_step4/input/train.csv \
  --output-json /mnt/sdb/jinming.hu/UDPET/loader_checks/optimizer_step_8patch_20260917/report.json
```

The two `pydeps` directories are pip target directories joined through
`PYTHONPATH`; they are not two activated virtual environments. The interpreter
is the system `/usr/bin/python3`. The first path supplies PyTorch/CUDA packages;
the second supplies NumPy/SciPy/nibabel/scikit-image and related metrics
packages.

## Fixed quantitative validation

Every optimizer run now requires `--val-csv`. Validation uses a fixed,
non-augmented set of explicit `(row_index, sample_seed)` requests and never
reads a test manifest. Results are written below the run directory as:

- `validation/epoch_NNNN_patient_metrics.csv`: one row per patient and DRF;
- `validation/epoch_NNNN_summary.json`: per-DRF and overall means with a
  deterministic patient bootstrap confidence interval;
- `train_loss.csv`: training losses plus validation body RMSE, body SUVmean
  absolute bias, and reference-hotspot absolute bias.

Patch errors are pooled before each patient/DRF metric is calculated. Overall
results first average available DRFs within a patient and then weight patients
equally. The body is defined from NORMAL PET as SUV `> 0.2`. The high-uptake
target is the top one percent of NORMAL body voxels per sampled patch and is
explicitly named a reference-defined hotspot, not a lesion.

The final engineering comparison used two real eight-patch optimizer updates.
One run was stopped after epoch 1 and strictly resumed; the control ran both
epochs continuously. G, D, both optimizers, both schedulers, training RNG,
canonical loader generator, sampler epoch, contract hash, loss CSV, and all
validation CSV/JSON files were exactly equal. See
`docs/ENGINEERING_GATE_REPORT_20260917.md` for the bounded protocol and artifact
locations. The validation subset contained only two patients, so its metric
values are pipeline checks rather than scientific estimates.

## Remaining launch gates

Scheduler, one-step optimization, eight-patch memory, fixed patient-level
validation, multi-iteration execution, and exact epoch-boundary resume are no
longer blockers. Full-cache construction, integrity, cross-center numerical
equivalence, storage-contract binding, and the bounded HDF5/NIfTI
training/resume gate are also complete. Before a long baseline run, freeze the
baseline configuration and predeclare the full validation set/checkpoint rule.
LeMod remains unavailable until lesion masks are added to a versioned manifest.

## Layout

- `code/`: UDPET adaptation plus the versioned loader optimization and tests
- `configs/server_snapshot_20260915.json`: sanitized provenance, environment,
  dataset counts, and source hashes
- `docs/ENGINEERING_GATE_REPORT_20260917.md`: bounded real-data engineering
  evidence; not a QuMod efficacy result
- `docs/LOADER_BENCHMARK_REPORT_20260917.md`: ABBA wall-clock, memory, and
  GPU-forward starvation comparison
- `docs/HDF5_PATCH_CACHE_EXPERIMENT_20260917.md`: direct-patch cache numerical,
  storage, throughput, memory, and optimizer-step pilot
- `docs/FULL_HDF5_CACHE_AUDIT_20260918.md`: four-split coverage, full HDF5
  decoded-data integrity, and cross-center NIfTI-equivalence audit
- `docs/HDF5_TRAINING_GATE_20260918.md`: one-GPU HDF5/NIfTI training,
  validation, strict-resume, and storage-contract comparison
