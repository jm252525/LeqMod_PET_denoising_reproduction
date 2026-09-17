# UDPET HDF5 direct-patch cache experiment

**Date:** 2026-09-17
**Scope:** bounded engineering diagnostic on train data only; no test manifest,
no formal training, and no efficacy claim.

## Question

The existing loader materializes a complete gzip-compressed NIfTI volume before
selecting eight `80x80x80` patches. The earlier locality/cache benchmark reduced
predicted NORMAL loads but did not improve wall time, because every sampled row
still decompressed its complete low-count volume. This experiment asks whether a
versioned, chunked representation that reads only selected patches removes that
bottleneck without changing any tensor presented to the model.

## Implementation

- Original `.nii.gz` files are immutable and remain the source of truth.
- A deterministic, center-balanced pilot selected 24 train studies: 12 Bern and
  12 Ruijin, comprising 135 low-count rows.
- Each study is stored in one HDF5 file containing one full-volume NORMAL array
  and all available low-count levels.
- Stored values are full-volume float32 SUVbw after the loader's existing
  `value <= 0 -> 0` transform. No float16 quantization is used.
- Dataset chunks are `80x80x80`, with HDF5 LZF compression and shuffle.
- NORMAL-derived crop boxes and candidate coordinates are stored as metadata;
  the only cached image representation remains full-volume.
- Files are created through same-directory temporary files and atomically
  replaced. The builder refuses an existing file by default; an explicit
  resumable mode validates protocol, configuration, datasets, and source paths
  before reuse.
- The HDF5 backend lazily imports `h5py`, opens files read-only inside each
  worker, bounds open handles with an LRU, and reads the chosen low/NORMAL
  patches directly.
- The default training/data path remains NIfTI. The cache backend is opt-in.

`h5py==3.11.0` was installed into the isolated target directory
`/mnt/sdb/jinming.hu/UDPET/hdf5_cache_env/pydeps`; the existing PyTorch and
scientific dependency directories were not modified.

## Cache construction

Artifacts:

`/mnt/sdb/jinming.hu/UDPET/chunked_cache_experiment/hdf5_lzf_chunk80_24studies_20260917`

| Quantity | Result |
| --- | ---: |
| Studies | 24 |
| Low-count rows | 135 |
| Source `.nii.gz` size | 6.044 GiB |
| HDF5 cache size | 12.121 GiB |
| Cache/source ratio | 2.006 |
| Build time | 530.4 s |

The complete train manifest contains about 181.7 GiB of unique source files.
If the pilot ratio remains representative, a full train cache would require
approximately 364 GiB. This is a projection, not a measured full-cache size.

## Numerical-equivalence gate

Twenty-four deterministic weighted requests were evaluated through both
backends with eight patches per request and the normal training augmentation
path enabled. Low-count, NORMAL, and weight tensors were compared exactly and
hashed.

| Tensor | Maximum absolute difference | Exact failures |
| --- | ---: | ---: |
| Low-count | 0.0 | 0 |
| NORMAL | 0.0 | 0 |
| Weight | 0.0 | 0 |

Result: **passed**. The storage backend did not change a sampled model input.

## ABBA performance benchmark

Protocol:

- order: NIfTI, HDF5, HDF5, NIfTI;
- 100 batches per run, two DataLoader workers, persistent workers;
- identical volume-grouped weighted requests and request order in every arm;
- eight `80x80x80` patches per batch;
- same deterministic generator forward on physical GPU 0;
- five batches excluded from steady-state summaries;
- same-request output-checksum maximum spread: `0.0`.

| Metric | NIfTI median | HDF5 median | Change |
| --- | ---: | ---: | ---: |
| Wall time per 100 batches | 181.08 s | 42.48 s | 4.26x speedup |
| Steady fetch mean | 1.622 s | 0.328 s | 4.95x speedup |
| Steady fetch p95 | 4.711 s | 0.882 s | 5.34x lower |
| GPU-forward pipeline wait fraction | 94.50% | 77.45% | -17.05 percentage points |
| Maximum process-tree PSS | 3390.5 MiB | 1210.3 MiB | HDF5/NIfTI = 0.357 |

Fetch wait is measured against a generator-forward consumer. A complete
adversarial optimizer step is slower, so these wait fractions must not be
interpreted as formal-training GPU utilization.

## Real optimizer-step gate

The HDF5 backend completed one real QuMod-enabled discriminator update and one
generator update using eight patches. All losses were finite:

- reconstruction: `2.425142`;
- discriminator: `0.498100`;
- generator adversarial: `0.940570`;
- local SUV bias: `1.244991`.

Peak CUDA allocation was 7166.5 MiB and peak reservation was 10196 MiB. The
temporary version-2 checkpoint restored generator/discriminator parameters,
both optimizers, scheduler, CPU/CUDA RNG, and was removed after validation.

## Decision and boundary

The `80^3` LZF HDF5 direct-patch design passes the bounded numerical,
throughput, memory, optimizer-step, and checkpoint gates. It is therefore the
preferred candidate for a full train/validation derived cache.

It is not yet the default formal-training backend. Before promotion:

1. build the complete train and validation caches with resumable validation;
2. repeat exact patch checks across a larger cross-center sample;
3. integrate cache-index hashes and backend identity into the strict training
   contract;
4. run a bounded multi-iteration train/validation/resume comparison against
   the NIfTI backend;
5. keep target-derived crop coordinates restricted to training patch sampling;
   clinical/inference cropping must be derived from the low-count input or a
   fixed acquisition field of view.

LeMod remains unsupported by this experimental cache because the current
manifests do not contain lesion masks.
