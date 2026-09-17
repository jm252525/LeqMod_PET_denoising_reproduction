#!/usr/bin/env python3
"""Compare deterministic real-data patches from NIfTI and HDF5 backends."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from CSVPairDataset import CSVPairDataset, make_weighted_sampler


def parse_args():
    parser = argparse.ArgumentParser(description="Validate UDPET HDF5 patch equivalence")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--num-requests", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[80, 80, 80])
    parser.add_argument("--stride-size", nargs=3, type=int, default=[20, 20, 20])
    parser.add_argument("--valid-value-threshold", type=float, default=0.2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.2)
    return parser.parse_args()


def tensor_sha256(value):
    return hashlib.sha256(value.contiguous().numpy().tobytes()).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def make_dataset(opts, backend):
    return CSVPairDataset(
        csv_path=opts.csv,
        split="train",
        patch_size=opts.patch_size,
        stride_size=opts.stride_size,
        patches_per_volume=opts.patches_per_volume,
        valid_value_threshold=opts.valid_value_threshold,
        min_valid_fraction=opts.min_valid_fraction,
        augmentation=True,
        rotate_degrees=10,
        enable_lemod=False,
        reference_cache_size=1,
        storage_backend=backend,
        chunk_cache_index=opts.cache_index if backend == "hdf5" else None,
        hdf5_handle_cache_size=1,
        seed=opts.seed,
        validate_paths=True,
    )


def main():
    opts = parse_args()
    if opts.num_requests <= 0:
        raise ValueError("num_requests must be positive")
    nifti = make_dataset(opts, "nifti")
    hdf5 = make_dataset(opts, "hdf5")
    sampler = make_weighted_sampler(nifti, opts.seed, opts.num_requests)
    requests = list(sampler)
    records = []
    global_max = {"low": 0.0, "high": 0.0, "weight": 0.0}
    for position, request in enumerate(requests):
        source = nifti[request]
        cached = hdf5[request]
        record = {
            "position": position,
            "row_index": int(request[0]),
            "sample_seed": int(request[1]),
            "patient_id": source["patient_id"],
            "count_label": source["count_label"],
        }
        for key, output_name in (
            ("vol_low", "low"),
            ("vol_high", "high"),
            ("vol_weight", "weight"),
        ):
            left = source[key]
            right = cached[key]
            maximum = float(torch.max(torch.abs(left - right)))
            record[f"{output_name}_max_abs_difference"] = maximum
            record[f"{output_name}_exact_equal"] = bool(torch.equal(left, right))
            record[f"{output_name}_nifti_sha256"] = tensor_sha256(left)
            record[f"{output_name}_hdf5_sha256"] = tensor_sha256(right)
            global_max[output_name] = max(global_max[output_name], maximum)
        if source["crop_box"] != cached["crop_box"]:
            raise AssertionError(f"Crop-box mismatch for request {request}")
        records.append(record)
        print(f"VALIDATION_PROGRESS request={position + 1}/{len(requests)}", flush=True)

    failures = [
        record for record in records
        if not all(record[f"{name}_exact_equal"] for name in ("low", "high", "weight"))
    ]
    report = {
        "protocol": "udpet_hdf5_patch_equivalence_v1_20260917",
        "scope": "engineering diagnostic; train pilot only; no optimizer and no test manifest",
        "csv": str(Path(opts.csv).resolve()),
        "cache_index": str(Path(opts.cache_index).resolve()),
        "requests": len(records),
        "patches_per_request": opts.patches_per_volume,
        "global_max_abs_difference": global_max,
        "exact_failure_count": len(failures),
        "all_exact": not failures,
        "records": records,
        "nifti_io_stats": nifti.io_stats(),
        "hdf5_io_stats": hdf5.io_stats(),
        "formal_training_started": False,
        "test_manifest_read": False,
    }
    atomic_json(opts.output_json, report)
    print("VALIDATION_RESULT=" + json.dumps({
        "requests": len(records),
        "all_exact": report["all_exact"],
        "global_max_abs_difference": global_max,
    }, sort_keys=True), flush=True)
    if failures:
        raise AssertionError(f"{len(failures)} HDF5 patch requests differed from NIfTI")


if __name__ == "__main__":
    main()
