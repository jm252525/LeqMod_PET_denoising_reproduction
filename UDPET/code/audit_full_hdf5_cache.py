#!/usr/bin/env python3
"""Audit full UDPET HDF5 cache coverage, integrity, and exact equivalence.

This audit is read-only. It verifies every manifest/index/container mapping,
opens every HDF5 study, validates all dataset metadata, and optionally decodes
every stored voxel. Numerical equivalence is checked on a deterministic sample
stratified by split, center, and DRF at both full-volume and model-patch levels.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np

from CSVPairDataset import CSVPairDataset
from build_hdf5_patch_cache import PROTOCOL


SPLITS = ("train", "val", "test_bern", "test_ruijin")


def parse_args():
    parser = argparse.ArgumentParser(description="Audit a complete UDPET HDF5 cache")
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--comparison-per-stratum", type=int, default=2)
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument("--full-data-scan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--scan-workers", type=int, default=1)
    parser.add_argument("--progress-every-studies", type=int, default=25)
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def row_key(row):
    return (
        row["site"], row["patient_id"], row["study_id"], row["count_label"],
        row["low_count_path"], row["full_count_path"],
    )


def study_key(row):
    return row["site"], row["patient_id"], row["study_id"], row["full_count_path"]


def stable_rank(parts):
    return hashlib.sha256("\0".join(map(str, parts)).encode("utf-8")).hexdigest()


def select_comparison_rows(rows, per_stratum):
    groups = defaultdict(list)
    for row in rows:
        groups[(row["site"], row["count_label"])].append(row)
    selected = []
    strata = {}
    for stratum in sorted(groups):
        ordered = sorted(
            groups[stratum],
            key=lambda row: stable_rank((
                "full_hdf5_audit_v1", *stratum, row["patient_id"],
                row["study_id"], row["low_count_path"],
            )),
        )
        chosen = ordered[:min(per_stratum, len(ordered))]
        selected.extend(chosen)
        strata[f"{stratum[0]}:{stratum[1]}"] = len(chosen)
    return selected, strata


def append_error(errors, split, category, message, path=None):
    record = {"split": split, "category": category, "message": str(message)}
    if path is not None:
        record["path"] = str(path)
    errors.append(record)


def decoded_dataset_scan(dataset):
    """Read every decoded voxel in C-order x slabs and return health/digest."""
    digest = hashlib.sha256()
    voxel_count = 0
    nonfinite_count = 0
    minimum = math.inf
    maximum = -math.inf
    step = int((dataset.chunks or dataset.shape)[0])
    for start in range(0, dataset.shape[0], step):
        block = np.asarray(dataset[start:min(start + step, dataset.shape[0]), :, :])
        block = np.ascontiguousarray(block, dtype=np.float32)
        voxel_count += int(block.size)
        finite = np.isfinite(block)
        nonfinite_count += int(block.size - np.count_nonzero(finite))
        if np.any(finite):
            minimum = min(minimum, float(np.min(block[finite])))
            maximum = max(maximum, float(np.max(block[finite])))
        digest.update(block.tobytes(order="C"))
    return {
        "sha256": digest.hexdigest(),
        "voxels": voxel_count,
        "nonfinite_voxels": nonfinite_count,
        "minimum": minimum,
        "maximum": maximum,
    }


def scan_container_payload(cache_path):
    """Decode and hash every image dataset in one container."""
    import h5py

    records = []
    with h5py.File(cache_path, "r") as handle:
        datasets = [handle["normal"]]
        datasets.extend(handle[f"low/{label}"] for label in sorted(handle["low"].keys()))
        for dataset in datasets:
            source_path = dataset.attrs.get("source_path", "")
            if isinstance(source_path, bytes):
                source_path = source_path.decode("utf-8")
            records.append({"source_path": str(source_path), **decoded_dataset_scan(dataset)})
    return {"cache_path": str(cache_path), "datasets": records}


def nifti_ready_digest(path):
    image = nib.load(path)
    array = np.asarray(image.dataobj, dtype=np.float32).copy()
    if not np.all(np.isfinite(array)):
        raise ValueError("source NIfTI contains NaN/Inf")
    array[array <= 0] = 0
    return {
        "shape": list(map(int, array.shape)),
        "sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def verify_dataset_metadata(dataset, configuration, expected_shape, source_path):
    problems = []
    if tuple(dataset.shape) != tuple(expected_shape):
        problems.append(f"shape {dataset.shape} != {tuple(expected_shape)}")
    if np.dtype(dataset.dtype) != np.dtype(np.float32):
        problems.append(f"dtype {dataset.dtype} != float32")
    expected_chunks = tuple(
        min(int(size), int(chunk))
        for size, chunk in zip(expected_shape, configuration["chunk_size"])
    )
    if tuple(dataset.chunks or ()) != expected_chunks:
        problems.append(f"chunks {dataset.chunks} != {expected_chunks}")
    expected_compression = None if configuration["compression"] == "none" else configuration["compression"]
    if dataset.compression != expected_compression:
        problems.append(f"compression {dataset.compression!r} != {expected_compression!r}")
    if expected_compression is not None and not bool(dataset.shuffle):
        problems.append("shuffle filter is disabled")
    if dataset.attrs.get("source_path", "") != source_path:
        problems.append("source_path attribute mismatch")
    return problems


def audit_split(
    split, split_root, per_stratum, full_data_scan, scan_workers,
    progress_every, errors,
):
    import h5py

    index_path = split_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    source_rows, _ = read_csv(index["source_csv"])
    cached_rows, _ = read_csv(index["pilot_csv"])
    configuration = index["configuration"]
    split_result = {
        "source_manifest": index["source_csv"],
        "cache_manifest": index["pilot_csv"],
        "index": str(index_path),
        "configuration": configuration,
    }

    if index.get("protocol") != PROTOCOL:
        append_error(errors, split, "coverage", f"index protocol {index.get('protocol')!r}")
    if index.get("split") != split:
        append_error(errors, split, "coverage", f"index split {index.get('split')!r} != {split!r}")
    if sha256_file(index["source_csv"]) != index.get("source_csv_sha256"):
        append_error(errors, split, "coverage", "source manifest SHA256 mismatch")
    if sha256_file(index["pilot_csv"]) != index.get("pilot_csv_sha256"):
        append_error(errors, split, "coverage", "cache manifest SHA256 mismatch")

    source_keys = [row_key(row) for row in source_rows]
    cached_keys = [row_key(row) for row in cached_rows]
    source_counter = Counter(source_keys)
    cached_counter = Counter(cached_keys)
    if source_counter != cached_counter:
        missing = list((source_counter - cached_counter).elements())
        extra = list((cached_counter - source_counter).elements())
        append_error(
            errors, split, "coverage",
            f"cache manifest differs: missing={len(missing)} extra={len(extra)}",
        )
    duplicate_rows = sum(value - 1 for value in source_counter.values() if value > 1)
    if duplicate_rows:
        append_error(errors, split, "coverage", f"source manifest has {duplicate_rows} duplicate rows")

    low_paths = {row["low_count_path"] for row in source_rows}
    full_paths = {row["full_count_path"] for row in source_rows}
    source_paths = low_paths | full_paths
    index_rows = index.get("rows", {})
    if set(index_rows) != low_paths:
        append_error(
            errors, split, "coverage",
            f"index low paths differ: expected={len(low_paths)} observed={len(index_rows)}",
        )
    if index.get("row_count") != len(source_rows):
        append_error(errors, split, "coverage", "row_count does not match source manifest")

    groups = defaultdict(list)
    for row in source_rows:
        groups[study_key(row)].append(row)
    if index.get("study_count") != len(groups):
        append_error(errors, split, "coverage", "study_count does not match source manifest")

    data_root = (split_root / "data").resolve()
    referenced_files = {Path(entry["cache_path"]).resolve() for entry in index_rows.values()}
    actual_files = {path.resolve() for path in data_root.rglob("*.h5")}
    if referenced_files != actual_files:
        append_error(
            errors, split, "coverage",
            f"HDF5 file set differs: referenced={len(referenced_files)} actual={len(actual_files)}",
        )
    escaped = [path for path in referenced_files if not path.is_relative_to(data_root)]
    if escaped:
        append_error(errors, split, "coverage", f"{len(escaped)} cache paths escape data root")
    temporary_files = list(split_root.rglob("*.tmp*"))
    if temporary_files:
        append_error(errors, split, "coverage", f"{len(temporary_files)} temporary files remain")

    missing_sources = [path for path in source_paths if not Path(path).is_file()]
    if missing_sources:
        append_error(errors, split, "coverage", f"{len(missing_sources)} source NIfTI files missing")
    observed_source_bytes = sum(Path(path).stat().st_size for path in source_paths if Path(path).is_file())
    observed_cache_bytes = sum(path.stat().st_size for path in actual_files)
    if observed_source_bytes != index.get("source_compressed_bytes"):
        append_error(errors, split, "coverage", "source byte total mismatch")
    if observed_cache_bytes != index.get("cache_bytes"):
        append_error(errors, split, "coverage", "cache byte total mismatch")

    selected_rows, selected_strata = select_comparison_rows(source_rows, per_stratum)
    comparison_sources = {
        path
        for row in selected_rows
        for path in (row["low_count_path"], row["full_count_path"])
    }
    sampled_hdf5_digests = {}
    study_records = {Path(item["cache_path"]).resolve(): item for item in index["studies"]}
    group_by_cache = {}
    for key, rows in groups.items():
        cache_paths = {
            Path(index_rows[row["low_count_path"]]["cache_path"]).resolve() for row in rows
        }
        if len(cache_paths) != 1:
            append_error(errors, split, "coverage", f"study maps to {len(cache_paths)} cache files")
            continue
        group_by_cache[next(iter(cache_paths))] = (key, rows)

    scan_summary = {
        "files_opened": 0,
        "datasets_opened": 0,
        "datasets_fully_decoded": 0,
        "decoded_voxels": 0,
        "nonfinite_voxels": 0,
        "negative_voxels_detected": 0,
        "aggregate_sha256": None,
    }
    aggregate_digest = hashlib.sha256()
    started = time.perf_counter()
    ordered_files = sorted(referenced_files)
    parallel_payload_scan = full_data_scan and scan_workers > 1
    for position, cache_path in enumerate(ordered_files, start=1):
        try:
            if cache_path not in group_by_cache:
                raise ValueError("cache file has no source-study mapping")
            key, rows = group_by_cache[cache_path]
            record = study_records.get(cache_path)
            if record is None:
                raise ValueError("cache file is absent from studies records")
            with h5py.File(cache_path, "r") as handle:
                scan_summary["files_opened"] += 1
                if handle.attrs.get("protocol", "") != PROTOCOL:
                    append_error(errors, split, "metadata", "container protocol mismatch", cache_path)
                observed_configuration = json.loads(handle.attrs["configuration_json"])
                if observed_configuration != configuration:
                    append_error(errors, split, "metadata", "configuration mismatch", cache_path)
                for field, expected in (
                    ("site", key[0]), ("patient_id", key[1]), ("study_id", key[2]),
                ):
                    if handle.attrs.get(field, "") != expected:
                        append_error(errors, split, "metadata", f"{field} attribute mismatch", cache_path)

                normal = handle["normal"]
                expected_shape = tuple(record["shape"])
                for problem in verify_dataset_metadata(
                    normal, configuration, expected_shape, key[3]
                ):
                    append_error(errors, split, "metadata", f"normal: {problem}", cache_path)
                affine = np.asarray(normal.attrs.get("affine"))
                if affine.shape != (4, 4) or not np.all(np.isfinite(affine)):
                    append_error(errors, split, "metadata", "invalid NORMAL affine", cache_path)

                box = np.asarray(handle["crop_box"][...], dtype=np.int64)
                if box.shape != (6,):
                    append_error(errors, split, "metadata", f"crop_box shape {box.shape}", cache_path)
                    crop_shape = None
                else:
                    crop_shape = np.asarray(
                        [box[1] - box[0], box[3] - box[2], box[5] - box[4]], dtype=np.int64
                    )
                    bounds_ok = (
                        0 <= box[0] < box[1] <= expected_shape[0]
                        and 0 <= box[2] < box[3] <= expected_shape[1]
                        and 0 <= box[4] < box[5] <= expected_shape[2]
                    )
                    if not bounds_ok:
                        append_error(errors, split, "metadata", "crop_box is out of bounds", cache_path)
                    if box.tolist() != record["crop_box"]:
                        append_error(errors, split, "metadata", "crop_box differs from index", cache_path)

                candidates = np.asarray(handle["candidate_boxes"][...], dtype=np.int64)
                if candidates.ndim != 2 or candidates.shape[1:] != (6,):
                    append_error(errors, split, "metadata", f"candidate shape {candidates.shape}", cache_path)
                else:
                    if len(candidates) != int(record["candidate_boxes"]):
                        append_error(errors, split, "metadata", "candidate count differs from index", cache_path)
                    if crop_shape is not None and len(candidates):
                        starts = candidates[:, [0, 2, 4]]
                        ends = candidates[:, [1, 3, 5]]
                        widths = ends - starts
                        if np.any(starts < 0) or np.any(ends > crop_shape):
                            append_error(errors, split, "metadata", "candidate boxes out of crop bounds", cache_path)
                        if np.any(widths != np.asarray(configuration["patch_size"])):
                            append_error(errors, split, "metadata", "candidate patch size mismatch", cache_path)

                expected_labels = {row["count_label"] for row in rows}
                observed_labels = set(handle["low"].keys())
                if observed_labels != expected_labels:
                    append_error(errors, split, "metadata", "low-count dataset labels mismatch", cache_path)
                row_by_label = {row["count_label"]: row for row in rows}
                datasets = [(key[3], normal)]
                for label in sorted(expected_labels):
                    row = row_by_label[label]
                    dataset = handle[f"low/{label}"]
                    for problem in verify_dataset_metadata(
                        dataset, configuration, expected_shape, row["low_count_path"]
                    ):
                        append_error(errors, split, "metadata", f"low/{label}: {problem}", cache_path)
                    observed_percent = float(dataset.attrs.get("count_percent", math.nan))
                    if observed_percent != float(row["count_percent"]):
                        append_error(errors, split, "metadata", f"low/{label}: count_percent mismatch", cache_path)
                    datasets.append((row["low_count_path"], dataset))

                for source_path, dataset in datasets:
                    scan_summary["datasets_opened"] += 1
                    # Even the metadata preflight fully decodes the deterministic
                    # comparison sample.  This keeps the exact NIfTI/HDF5 check
                    # meaningful while avoiding a full-cache scan.
                    decode_fully = (
                        (full_data_scan and not parallel_payload_scan)
                        or (not full_data_scan and source_path in comparison_sources)
                    )
                    if decode_fully:
                        scan = decoded_dataset_scan(dataset)
                        scan_summary["datasets_fully_decoded"] += 1
                        scan_summary["decoded_voxels"] += scan["voxels"]
                        scan_summary["nonfinite_voxels"] += scan["nonfinite_voxels"]
                        if scan["minimum"] < 0:
                            scan_summary["negative_voxels_detected"] += 1
                        if scan["nonfinite_voxels"]:
                            append_error(errors, split, "decoded_data", "non-finite voxels", source_path)
                        if scan["minimum"] < 0:
                            append_error(errors, split, "decoded_data", "negative cached voxels", source_path)
                        if full_data_scan:
                            aggregate_digest.update(source_path.encode("utf-8"))
                            aggregate_digest.update(scan["sha256"].encode("ascii"))
                        if source_path in comparison_sources:
                            sampled_hdf5_digests[source_path] = scan
                    else:
                        probe = np.asarray(dataset[0:1, 0:1, 0:1], dtype=np.float32)
                        if not np.all(np.isfinite(probe)):
                            append_error(errors, split, "decoded_data", "non-finite sentinel", source_path)
        except Exception as error:
            append_error(errors, split, "container", f"{type(error).__name__}: {error}", cache_path)
        if position % progress_every == 0 or position == len(ordered_files):
            print(
                f"AUDIT_PROGRESS split={split} studies={position}/{len(ordered_files)} "
                f"datasets_decoded={scan_summary['datasets_fully_decoded']} errors={len(errors)}",
                flush=True,
            )

    if parallel_payload_scan:
        scans_by_source = {}
        with ProcessPoolExecutor(max_workers=scan_workers) as executor:
            for position, result in enumerate(
                executor.map(scan_container_payload, map(str, ordered_files)), start=1
            ):
                for scan in result["datasets"]:
                    source_path = scan.pop("source_path")
                    if source_path in scans_by_source:
                        append_error(
                            errors, split, "decoded_data",
                            "source dataset was decoded more than once", source_path,
                        )
                    scans_by_source[source_path] = scan
                    scan_summary["datasets_fully_decoded"] += 1
                    scan_summary["decoded_voxels"] += scan["voxels"]
                    scan_summary["nonfinite_voxels"] += scan["nonfinite_voxels"]
                    if scan["minimum"] < 0:
                        scan_summary["negative_voxels_detected"] += 1
                    if scan["nonfinite_voxels"]:
                        append_error(errors, split, "decoded_data", "non-finite voxels", source_path)
                    if scan["minimum"] < 0:
                        append_error(errors, split, "decoded_data", "negative cached voxels", source_path)
                if position % progress_every == 0 or position == len(ordered_files):
                    print(
                        f"PAYLOAD_AUDIT_PROGRESS split={split} "
                        f"studies={position}/{len(ordered_files)} "
                        f"datasets_decoded={scan_summary['datasets_fully_decoded']} "
                        f"errors={len(errors)}",
                        flush=True,
                    )
        if set(scans_by_source) != source_paths:
            append_error(
                errors, split, "decoded_data",
                f"decoded source set differs: expected={len(source_paths)} "
                f"observed={len(scans_by_source)}",
            )
        for source_path in sorted(scans_by_source):
            scan = scans_by_source[source_path]
            aggregate_digest.update(source_path.encode("utf-8"))
            aggregate_digest.update(scan["sha256"].encode("ascii"))
            if source_path in comparison_sources:
                sampled_hdf5_digests[source_path] = scan

    scan_summary["aggregate_sha256"] = aggregate_digest.hexdigest() if full_data_scan else None
    scan_summary["elapsed_seconds"] = time.perf_counter() - started

    full_volume_records = []
    for source_path in sorted(comparison_sources):
        record = {"source_path": source_path}
        try:
            nifti = nifti_ready_digest(source_path)
            cached = sampled_hdf5_digests.get(source_path)
            if cached is None:
                raise ValueError("sampled HDF5 digest is unavailable")
            record.update({
                "shape": nifti["shape"],
                "nifti_sha256": nifti["sha256"],
                "hdf5_sha256": cached["sha256"],
                "exact": nifti["sha256"] == cached["sha256"],
                "nifti_minimum": nifti["minimum"],
                "nifti_maximum": nifti["maximum"],
                "hdf5_minimum": cached["minimum"],
                "hdf5_maximum": cached["maximum"],
            })
            if not record["exact"]:
                append_error(errors, split, "full_volume_equivalence", "decoded SHA256 mismatch", source_path)
        except Exception as error:
            record.update({"exact": False, "error": f"{type(error).__name__}: {error}"})
            append_error(errors, split, "full_volume_equivalence", record["error"], source_path)
        full_volume_records.append(record)

    split_result.update({
        "coverage": {
            "rows": len(source_rows),
            "studies": len(groups),
            "low_count_files": len(low_paths),
            "normal_files": len(full_paths),
            "unique_source_files": len(source_paths),
            "hdf5_files": len(actual_files),
            "source_bytes": observed_source_bytes,
            "cache_bytes": observed_cache_bytes,
            "manifest_exact": source_counter == cached_counter,
            "index_low_path_exact": set(index_rows) == low_paths,
            "hdf5_file_set_exact": referenced_files == actual_files,
            "missing_source_files": len(missing_sources),
            "temporary_files": len(temporary_files),
        },
        "integrity": scan_summary,
        "comparison_selection": {
            "rows": len(selected_rows),
            "strata": selected_strata,
            "unique_source_volumes": len(comparison_sources),
        },
        "full_volume_equivalence": {
            "comparisons": len(full_volume_records),
            "exact": sum(bool(record.get("exact")) for record in full_volume_records),
            "failures": sum(not bool(record.get("exact")) for record in full_volume_records),
            "records": full_volume_records,
        },
        "_selected_rows": selected_rows,
    })
    return split_result


def audit_patch_equivalence(split_results, patches_per_volume, errors):
    import torch

    records = []
    for split in SPLITS:
        split_result = split_results[split]
        configuration = split_result["configuration"]
        source_csv = split_result["source_manifest"]
        cache_index = split_result["index"]
        common = dict(
            csv_path=source_csv,
            split=split,
            patch_size=configuration["patch_size"],
            stride_size=configuration["stride_size"],
            patches_per_volume=patches_per_volume,
            valid_value_threshold=configuration["valid_value_threshold"],
            min_valid_fraction=configuration["min_valid_fraction"],
            augmentation=True,
            rotate_degrees=10,
            enable_lemod=False,
            reference_cache_size=1,
            seed=20260918,
            validate_paths=True,
        )
        nifti = CSVPairDataset(storage_backend="nifti", **common)
        cached = CSVPairDataset(
            storage_backend="hdf5",
            chunk_cache_index=cache_index,
            hdf5_handle_cache_size=1,
            **common,
        )
        row_indices = {row["low_count_path"]: index for index, row in enumerate(nifti.rows)}
        for row in split_result.pop("_selected_rows"):
            index = row_indices[row["low_count_path"]]
            sample_seed = int.from_bytes(
                hashlib.sha256(
                    ("patch_equivalence_v1\0" + split + "\0" + row["low_count_path"]).encode("utf-8")
                ).digest()[:4],
                byteorder="little",
            )
            request = (index, sample_seed)
            record = {
                "split": split,
                "site": row["site"],
                "count_label": row["count_label"],
                "patient_id": row["patient_id"],
                "study_id": row["study_id"],
                "row_index": index,
                "sample_seed": sample_seed,
            }
            try:
                source = nifti[request]
                direct = cached[request]
                exact = True
                for key, label in (
                    ("vol_low", "low"), ("vol_high", "high"), ("vol_weight", "weight"),
                ):
                    difference = float(torch.max(torch.abs(source[key] - direct[key])))
                    equal = bool(torch.equal(source[key], direct[key]))
                    record[f"{label}_max_abs_difference"] = difference
                    record[f"{label}_exact"] = equal
                    exact = exact and equal
                record["crop_box_exact"] = source["crop_box"] == direct["crop_box"]
                record["exact"] = exact and record["crop_box_exact"]
                if not record["exact"]:
                    append_error(
                        errors, split, "patch_equivalence", "sampled patch mismatch",
                        row["low_count_path"],
                    )
            except Exception as error:
                record.update({"exact": False, "error": f"{type(error).__name__}: {error}"})
                append_error(errors, split, "patch_equivalence", record["error"], row["low_count_path"])
            records.append(record)
        cached._close_hdf5_handles()
        print(
            f"PATCH_AUDIT_PROGRESS split={split} comparisons="
            f"{sum(record['split'] == split for record in records)}",
            flush=True,
        )
    return {
        "comparisons": len(records),
        "exact": sum(bool(record.get("exact")) for record in records),
        "failures": sum(not bool(record.get("exact")) for record in records),
        "maximum_low_abs_difference": max(
            (record.get("low_max_abs_difference", math.inf) for record in records), default=math.inf
        ),
        "maximum_high_abs_difference": max(
            (record.get("high_max_abs_difference", math.inf) for record in records), default=math.inf
        ),
        "records": records,
    }


def main():
    opts = parse_args()
    if opts.comparison_per_stratum <= 0:
        raise ValueError("comparison_per_stratum must be positive")
    if (
        opts.patches_per_volume <= 0
        or opts.progress_every_studies <= 0
        or opts.scan_workers <= 0
    ):
        raise ValueError("patch and progress counts must be positive")
    cache_root = Path(opts.cache_root).resolve()
    if not (cache_root / "BUILD_COMPLETE").is_file():
        raise FileNotFoundError("BUILD_COMPLETE marker is missing")

    started = time.perf_counter()
    errors = []
    split_results = {}
    for split in SPLITS:
        split_results[split] = audit_split(
            split,
            cache_root / split,
            opts.comparison_per_stratum,
            opts.full_data_scan,
            opts.scan_workers,
            opts.progress_every_studies,
            errors,
        )
    patch_equivalence = audit_patch_equivalence(
        split_results, opts.patches_per_volume, errors
    )
    coverage_totals = {
        name: sum(result["coverage"][name] for result in split_results.values())
        for name in (
            "rows", "studies", "low_count_files", "normal_files",
            "unique_source_files", "hdf5_files", "source_bytes", "cache_bytes",
        )
    }
    integrity_totals = {
        name: sum(result["integrity"][name] for result in split_results.values())
        for name in (
            "files_opened", "datasets_opened", "datasets_fully_decoded",
            "decoded_voxels", "nonfinite_voxels", "negative_voxels_detected",
        )
    }
    full_volume_comparisons = sum(
        result["full_volume_equivalence"]["comparisons"] for result in split_results.values()
    )
    full_volume_failures = sum(
        result["full_volume_equivalence"]["failures"] for result in split_results.values()
    )
    report = {
        "protocol": "udpet_full_hdf5_cache_audit_v1_20260918",
        "scope": (
            "read-only coverage and integrity audit of all cache containers plus "
            "split-center-DRF-stratified exact NIfTI equivalence"
        ),
        "cache_root": str(cache_root),
        "configuration": {
            "splits": list(SPLITS),
            "full_data_scan": opts.full_data_scan,
            "scan_workers": opts.scan_workers,
            "comparison_per_split_center_drf_stratum": opts.comparison_per_stratum,
            "patches_per_volume": opts.patches_per_volume,
        },
        "status": "passed" if not errors else "failed",
        "coverage_totals": coverage_totals,
        "integrity_totals": integrity_totals,
        "full_volume_equivalence": {
            "comparisons": full_volume_comparisons,
            "failures": full_volume_failures,
        },
        "patch_equivalence": patch_equivalence,
        "splits": split_results,
        "errors": errors,
        "error_count": len(errors),
        "nifti_deleted_or_moved": False,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(opts.output_json, report)
    print("AUDIT_RESULT=" + json.dumps({
        "status": report["status"],
        "errors": len(errors),
        "coverage": coverage_totals,
        "integrity": integrity_totals,
        "full_volume_comparisons": full_volume_comparisons,
        "full_volume_failures": full_volume_failures,
        "patch_comparisons": patch_equivalence["comparisons"],
        "patch_failures": patch_equivalence["failures"],
        "elapsed_seconds": report["elapsed_seconds"],
    }, sort_keys=True), flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
