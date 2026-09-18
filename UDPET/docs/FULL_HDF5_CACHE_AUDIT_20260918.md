# UDPET full HDF5 cache audit

- **Date:** 2026-09-18
- **Cache:** `/mnt/sdb/jinming.hu/UDPET/hdf5_cache_v1_full_20260917`
- **Result:** **passed** with zero audit errors

## Scope and decision

This was a read-only audit of the completed train, validation, Bern test, and
Ruijin test HDF5 cache. It covered three separate questions:

1. Does the cache cover every row and study in the four immutable source
   manifests, without extra, missing, or temporary containers?
2. Can every stored image dataset be decoded, and are its protocol, shape,
   dtype, chunks, compression, source path, affine/crop metadata, and values
   internally valid?
3. For deterministic samples from every split, center, and DRF stratum, are
   full decoded volumes and the actual augmented eight-patch model inputs
   numerically identical to the NIfTI backend?

The cache passes these gates and is suitable as the derived direct-patch
training representation. This audit does **not** make HDF5 an archival
replacement for the original NIfTI files: the cache preserves the arrays and
the NORMAL affine needed by the loader, but not every original NIfTI header
field. No NIfTI file was deleted or moved during this audit.

## Machine-readable artifacts

| Artifact | SHA-256 |
| --- | --- |
| `audit/full_audit.json` | `a40e4c1779bf4a09048278e991cdf84003dc114279a159f78ef17932cc89ce6c` |
| `audit/full_audit.log` | `58ae45888cfc54cda0c638f97e11bc8b50a2bf4a32f267563a81219e482cbaa1` |
| `audit/preflight_metadata_and_equivalence.json` | `514ef9b914c85778d3d23d15d9f5fe5a3b8c5ff760c76c64f1eca252938e9a5a` |

The final audit used four worker processes at low CPU and I/O priority and
took 6,008.6 seconds (100.1 minutes). The preflight opened every container and
dataset, then decoded a smaller deterministic equivalence sample before the
long scan was started.

## Coverage

Every source-manifest row, index row, study record, and HDF5 path matched
exactly. All referenced source files remained present, all recorded source and
cache byte totals matched the filesystem, and no temporary build file remained.

| Split | Rows / low-count files | Studies / HDF5 files | Unique source files | Source GiB | HDF5 GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| train | 5,020 | 927 | 5,947 | 181.655 | 363.893 |
| val | 635 | 117 | 752 | 23.583 | 47.223 |
| test_bern | 180 | 30 | 210 | 9.154 | 18.238 |
| test_ruijin | 465 | 89 | 554 | 14.426 | 29.034 |
| **Total** | **6,300** | **1,163** | **7,463** | **228.817** | **458.387** |

The cache/source size ratio is 2.003. The audit created only small JSON/log
files; free space remained about 867 GiB.

### Source DRF coverage is not rectangular

HDF5 covers the manifests completely, but it does not invent missing source
acquisitions. Bern has all six levels for every study. Ruijin has substantially
fewer `D2` rows, corresponding to the known absence of the nominal 50% count
level in much of the 2023 data.

| Center/split | D2 | D4 | D10 | D20 | D50 | D100 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bern train | 233 | 233 | 233 | 233 | 233 | 233 |
| Bern val | 30 | 30 | 30 | 30 | 30 | 30 |
| Bern test | 30 | 30 | 30 | 30 | 30 | 30 |
| Ruijin train | 156 | 694 | 693 | 693 | 693 | 693 |
| Ruijin val | 20 | 87 | 87 | 87 | 87 | 87 |
| Ruijin test | 20 | 89 | 89 | 89 | 89 | 89 |

This imbalance must be handled explicitly in sampling and reporting. It is a
source-data limitation, not a cache coverage failure.

## Full HDF5 integrity scan

Every stored float32 image array was decompressed in chunk-aligned x slabs.
The audit inspected 727,522,124,800 voxels, equivalent to about 2.647 TiB of
decoded float32 data.

| Check | Result |
| --- | ---: |
| HDF5 containers opened | 1,163 / 1,163 |
| Image datasets opened | 7,463 / 7,463 |
| Image datasets fully decoded | 7,463 / 7,463 |
| Decoded voxels | 727,522,124,800 |
| Non-finite voxels | 0 |
| Negative voxels after the specified clamp | 0 |
| Metadata, structure, or decoding errors | 0 |

The deterministic per-split decoded-content aggregates are:

| Split | Aggregate SHA-256 |
| --- | --- |
| train | `ce866633f78b6aaaff344b915e1dc54301be430a610b2f8bdcb833c3320acdc2` |
| val | `fbdf0d4702f5dbba1ab93db64706cc04b73777da35afca1220e5898144ccca16` |
| test_bern | `e3cda3c924c02d3319f7eaa1bf5603d6a606fd08aee7dab3a392559eee933489` |
| test_ruijin | `68c15d281c043b190aaadcf93023235de2833a0f0988b2751a0544062054354d` |

Each aggregate hashes the sorted source path and the SHA-256 of its complete
decoded float32 array. It is intended to detect later cache drift.

## Cross-center numerical equivalence

Two manifest rows were selected deterministically from every available
`split x center x DRF` stratum. Repeated NORMAL references were compared once,
giving 139 unique full-volume comparisons from 72 selected rows.

| Split | Selected rows | Unique full volumes | Exact full-volume matches |
| --- | ---: | ---: | ---: |
| train | 24 | 48 | 48 / 48 |
| val | 24 | 44 | 44 / 44 |
| test_bern | 12 | 23 | 23 / 23 |
| test_ruijin | 12 | 24 | 24 / 24 |
| **Total** | **72** | **139** | **139 / 139** |

The source NIfTI arrays were converted to float32 and passed through the same
`value <= 0 -> 0` transform as cache construction. All 139 decoded-array
SHA-256 values matched exactly.

The same 72 selected rows were then requested through both dataset backends
with normal training augmentation and eight patches per request. Every site and
DRF contributed six requests across train/validation/test.

| Model-input check | Result |
| --- | ---: |
| Exact eight-patch requests | 72 / 72 |
| Maximum low-count absolute difference | 0.0 |
| Maximum NORMAL absolute difference | 0.0 |
| Maximum weight-map absolute difference | 0.0 |
| Exact crop boxes | 72 / 72 |

## Reproduction command

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

## Cleanup boundary

This result removes cache corruption or numerical conversion as a blocker to
HDF5-backed training. It does not, by itself, authorize deletion of the server
NIfTI files. Before cleanup, the off-server original archive should be audited
against a versioned manifest of all 7,463 compressed source files, including
file size and SHA-256, and a restore drill should be passed for both centers and
all available DRFs. Until that succeeds, the HDF5 cache is a derived working
copy, not the only retained copy of the data.
