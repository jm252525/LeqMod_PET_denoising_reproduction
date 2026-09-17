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
This is a predicted decompression count, not a wall-clock benchmark; storage
throughput and worker scheduling still need measurement before a long run.

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
longer blockers. Before a long baseline run, freeze the baseline configuration,
run the pending loader wall-clock benchmark, and predeclare the full validation
set/checkpoint rule. LeMod remains unavailable until lesion masks are added to
a versioned manifest.

## Layout

- `code/`: UDPET adaptation plus the versioned loader optimization and tests
- `configs/server_snapshot_20260915.json`: sanitized provenance, environment,
  dataset counts, and source hashes
- `docs/ENGINEERING_GATE_REPORT_20260917.md`: bounded real-data engineering
  evidence; not a QuMod efficacy result
