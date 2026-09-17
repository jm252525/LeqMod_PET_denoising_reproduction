#!/usr/bin/env python3
"""Build a bounded, versioned HDF5 pilot cache for random UDPET patch reads.

The source NIfTI files remain immutable. Each selected study is written to one
atomic HDF5 container containing one NORMAL volume and all available low-count
levels. Volumes remain full-size float32 arrays; target-derived crop and patch
coordinates are metadata only, so the cache does not make a cropped target
representation the sole copy of the data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from pathlib import Path

import nibabel as nib
import numpy as np

from CSVPairDataset import CSVPairDataset, _apply_box, _candidate_boxes, _crop_box


PROTOCOL = "udpet_hdf5_patch_cache_v1"


def parse_args():
    parser = argparse.ArgumentParser(description="Build a UDPET HDF5 patch-cache pilot")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--centers", nargs="*", default=[])
    parser.add_argument("--count-levels", nargs="*", default=[])
    parser.add_argument("--max-studies", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[80, 80, 80])
    parser.add_argument("--stride-size", nargs=3, type=int, default=[20, 20, 20])
    parser.add_argument("--valid-value-threshold", type=float, default=0.2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.2)
    parser.add_argument("--chunk-size", nargs=3, type=int, default=[80, 80, 80])
    parser.add_argument("--compression", choices=("lzf", "gzip", "none"), default="lzf")
    parser.add_argument("--gzip-level", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Validate and reuse already completed atomic study files",
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def safe_token(value):
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._")
    return token[:80] or "unknown"


def study_key(row):
    return row["site"], row["patient_id"], row["study_id"], row["full_count_path"]


def stable_rank(key, seed):
    payload = (str(seed) + "\0" + "\0".join(map(str, key))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def select_studies(rows, maximum, seed):
    grouped = defaultdict(list)
    for row in rows:
        grouped[study_key(row)].append(row)
    by_site = defaultdict(list)
    for key in grouped:
        by_site[key[0]].append(key)
    for site in by_site:
        by_site[site].sort(key=lambda key: stable_rank(key, seed))
    sites = sorted(by_site)
    selected = []
    offset = 0
    while len(selected) < min(maximum, len(grouped)):
        added = False
        for site in sites:
            if offset < len(by_site[site]):
                selected.append(by_site[site][offset])
                added = True
                if len(selected) == min(maximum, len(grouped)):
                    break
        if not added:
            break
        offset += 1
    return [(key, sorted(grouped[key], key=lambda row: row["count_label"])) for key in selected]


def load_float32(path):
    image = nib.load(path)
    array = np.asarray(image.dataobj, dtype=np.float32).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError(f"NaN/Inf detected in source image: {path}")
    array[array <= 0] = 0
    return image, array


def dataset_options(compression, gzip_level):
    if compression == "none":
        return {}
    options = {"compression": compression, "shuffle": True}
    if compression == "gzip":
        options["compression_opts"] = int(gzip_level)
    return options


def chunk_shape(shape, requested):
    return tuple(min(int(size), int(chunk)) for size, chunk in zip(shape, requested))


def write_study(rows, cache_path, configuration):
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to build the HDF5 patch cache") from error

    first = rows[0]
    full_paths = {row["full_count_path"] for row in rows}
    if len(full_paths) != 1:
        raise ValueError(f"Study has multiple NORMAL paths: {study_key(first)}")
    labels = [row["count_label"] for row in rows]
    if len(labels) != len(set(labels)):
        raise ValueError(f"Study has duplicate count levels: {study_key(first)}")

    cache_path = Path(cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + f".tmp-{os.getpid()}")
    full_image, full = load_float32(first["full_count_path"])
    original_shape = tuple(map(int, full.shape))
    affine = np.asarray(full_image.affine, dtype=np.float64)
    box = tuple(_crop_box(full, configuration["valid_value_threshold"]))
    cropped = _apply_box(full, box)
    candidates = _candidate_boxes(
        cropped,
        tuple(configuration["patch_size"]),
        tuple(configuration["stride_size"]),
        configuration["valid_value_threshold"],
        configuration["min_valid_fraction"],
    )
    if len(candidates) < configuration["patches_per_volume"]:
        raise ValueError(
            f"Only {len(candidates)} valid boxes for {first['patient_id']}; "
            f"need {configuration['patches_per_volume']}"
        )

    create_options = dataset_options(
        configuration["compression"], configuration["gzip_level"]
    )
    source_bytes = Path(first["full_count_path"]).stat().st_size
    try:
        with h5py.File(temporary, "w") as handle:
            handle.attrs["protocol"] = PROTOCOL
            handle.attrs["site"] = first["site"]
            handle.attrs["patient_id"] = first["patient_id"]
            handle.attrs["study_id"] = first["study_id"]
            handle.attrs["unit"] = first["unit"]
            handle.attrs["created_unix_seconds"] = time.time()
            handle.attrs["configuration_json"] = json.dumps(configuration, sort_keys=True)
            normal = handle.create_dataset(
                "normal",
                data=full,
                dtype=np.float32,
                chunks=chunk_shape(original_shape, configuration["chunk_size"]),
                **create_options,
            )
            normal.attrs["source_path"] = first["full_count_path"]
            normal.attrs["affine"] = affine
            handle.create_dataset("crop_box", data=np.asarray(box, dtype=np.int32))
            handle.create_dataset(
                "candidate_boxes", data=np.asarray(candidates, dtype=np.int32)
            )
            low_group = handle.create_group("low")
            for row in rows:
                low_image, low = load_float32(row["low_count_path"])
                if tuple(low.shape) != original_shape:
                    raise ValueError(
                        f"Shape mismatch {low.shape} != {original_shape}: {row['low_count_path']}"
                    )
                affine_difference = float(
                    np.max(np.abs(np.asarray(low_image.affine, dtype=np.float64) - affine))
                )
                if affine_difference > 1e-4:
                    raise ValueError(
                        f"Affine mismatch {affine_difference}: {row['low_count_path']}"
                    )
                dataset = low_group.create_dataset(
                    row["count_label"],
                    data=low,
                    dtype=np.float32,
                    chunks=chunk_shape(original_shape, configuration["chunk_size"]),
                    **create_options,
                )
                dataset.attrs["source_path"] = row["low_count_path"]
                dataset.attrs["count_percent"] = float(row["count_percent"])
                source_bytes += Path(row["low_count_path"]).stat().st_size
                del low
            handle.flush()
        os.replace(temporary, cache_path)
    finally:
        if temporary.exists():
            temporary.unlink()
        del full

    return {
        "cache_path": str(cache_path.resolve()),
        "source_compressed_bytes": int(source_bytes),
        "cache_bytes": int(cache_path.stat().st_size),
        "shape": list(original_shape),
        "crop_box": list(box),
        "candidate_boxes": len(candidates),
    }


def inspect_study(rows, cache_path, configuration):
    """Validate one completed cache file before a resumable build reuses it."""
    try:
        import h5py
    except ImportError as error:
        raise RuntimeError("h5py is required to inspect the HDF5 patch cache") from error
    cache_path = Path(cache_path)
    first = rows[0]
    with h5py.File(cache_path, "r") as handle:
        if handle.attrs.get("protocol", "") != PROTOCOL:
            raise ValueError(f"Invalid cache protocol in {cache_path}")
        observed_configuration = json.loads(handle.attrs["configuration_json"])
        if observed_configuration != configuration:
            raise ValueError(f"Cache configuration mismatch in {cache_path}")
        normal = handle["normal"]
        if normal.attrs.get("source_path", "") != first["full_count_path"]:
            raise ValueError(f"NORMAL source mismatch in {cache_path}")
        for row in rows:
            dataset_name = f"low/{row['count_label']}"
            if dataset_name not in handle:
                raise ValueError(f"Missing {dataset_name} in {cache_path}")
            if handle[dataset_name].attrs.get("source_path", "") != row["low_count_path"]:
                raise ValueError(f"Low-count source mismatch for {dataset_name} in {cache_path}")
        shape = list(map(int, normal.shape))
        box = list(map(int, handle["crop_box"][...].tolist()))
        candidate_count = int(handle["candidate_boxes"].shape[0])
    source_paths = [first["full_count_path"], *(row["low_count_path"] for row in rows)]
    return {
        "cache_path": str(cache_path.resolve()),
        "source_compressed_bytes": int(sum(Path(path).stat().st_size for path in source_paths)),
        "cache_bytes": int(cache_path.stat().st_size),
        "shape": shape,
        "crop_box": box,
        "candidate_boxes": candidate_count,
    }


def write_csv(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main():
    opts = parse_args()
    if opts.overwrite and opts.resume_existing:
        raise ValueError("--overwrite and --resume-existing are mutually exclusive")
    if opts.max_studies <= 0:
        raise ValueError("max_studies must be positive")
    if any(value <= 0 for value in (*opts.patch_size, *opts.stride_size, *opts.chunk_size)):
        raise ValueError("patch, stride, and chunk dimensions must be positive")
    if not 0 <= opts.gzip_level <= 9:
        raise ValueError("gzip_level must be in [0, 9]")
    output_root = Path(opts.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configuration = {
        "patch_size": list(map(int, opts.patch_size)),
        "stride_size": list(map(int, opts.stride_size)),
        "patches_per_volume": 8,
        "valid_value_threshold": float(opts.valid_value_threshold),
        "min_valid_fraction": float(opts.min_valid_fraction),
        "chunk_size": list(map(int, opts.chunk_size)),
        "compression": opts.compression,
        "gzip_level": int(opts.gzip_level),
        "stored_dtype": "float32",
        "stored_extent": "full_volume",
        "nonpositive_values_clamped_to_zero": True,
        "candidate_coordinates": "relative_to_NORMAL_derived_crop_box",
    }

    dataset = CSVPairDataset(
        csv_path=opts.input_csv,
        split=opts.split,
        centers=opts.centers,
        count_levels=opts.count_levels,
        patch_size=opts.patch_size,
        stride_size=opts.stride_size,
        patches_per_volume=8,
        valid_value_threshold=opts.valid_value_threshold,
        min_valid_fraction=opts.min_valid_fraction,
        augmentation=False,
        enable_lemod=False,
        reference_cache_size=0,
        storage_backend="nifti",
        validate_paths=True,
    )
    selected = select_studies(dataset.rows, opts.max_studies, opts.seed)
    with Path(opts.input_csv).open("r", encoding="utf-8-sig", newline="") as handle:
        fieldnames = list(csv.DictReader(handle).fieldnames or [])

    index_rows = {}
    pilot_rows = []
    studies = []
    started = time.perf_counter()
    for position, (key, rows) in enumerate(selected, start=1):
        site, patient_id, study_id, _ = key
        digest = stable_rank(key, opts.seed)[:20]
        cache_path = output_root / safe_token(site) / f"{digest}.h5"
        print(
            f"CACHE_PROGRESS study={position}/{len(selected)} site={site} "
            f"patient={patient_id} levels={len(rows)}",
            flush=True,
        )
        if cache_path.exists() and opts.resume_existing:
            record = inspect_study(rows, cache_path, configuration)
            print(f"CACHE_REUSED path={cache_path}", flush=True)
        else:
            if cache_path.exists() and not opts.overwrite:
                raise FileExistsError(
                    f"Cache already exists: {cache_path}; use --resume-existing to "
                    "validate/reuse it or --overwrite for an intentional rebuild"
                )
            record = write_study(rows, cache_path, configuration)
        record.update({
            "site": site,
            "patient_id": patient_id,
            "study_id": study_id,
            "count_levels": [row["count_label"] for row in rows],
        })
        studies.append(record)
        for row in rows:
            pilot_rows.append({field: row.get(field, "") for field in fieldnames})
            index_rows[row["low_count_path"]] = {
                "cache_path": str(cache_path.resolve()),
                "low_dataset": f"low/{row['count_label']}",
                "full_count_path": row["full_count_path"],
                "count_label": row["count_label"],
                "site": row["site"],
                "patient_id": row["patient_id"],
                "study_id": row["study_id"],
            }

    write_csv(opts.output_csv, fieldnames, pilot_rows)
    index = {
        "protocol": PROTOCOL,
        "scope": "bounded engineering pilot; source NIfTI files are immutable",
        "source_csv": str(Path(opts.input_csv).resolve()),
        "source_csv_sha256": sha256_file(opts.input_csv),
        "pilot_csv": str(Path(opts.output_csv).resolve()),
        "pilot_csv_sha256": sha256_file(opts.output_csv),
        "split": opts.split,
        "selection_seed": opts.seed,
        "configuration": configuration,
        "study_count": len(studies),
        "row_count": len(pilot_rows),
        "source_compressed_bytes": sum(item["source_compressed_bytes"] for item in studies),
        "cache_bytes": sum(item["cache_bytes"] for item in studies),
        "elapsed_seconds": time.perf_counter() - started,
        "studies": studies,
        "rows": index_rows,
    }
    atomic_json(opts.output_index, index)
    print("CACHE_RESULT=" + json.dumps({
        "studies": index["study_count"],
        "rows": index["row_count"],
        "source_compressed_gib": index["source_compressed_bytes"] / 2**30,
        "cache_gib": index["cache_bytes"] / 2**30,
        "cache_over_source": (
            index["cache_bytes"] / index["source_compressed_bytes"]
            if index["source_compressed_bytes"] else None
        ),
        "elapsed_seconds": index["elapsed_seconds"],
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
