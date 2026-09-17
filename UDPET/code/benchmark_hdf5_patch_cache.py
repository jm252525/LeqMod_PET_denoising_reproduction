#!/usr/bin/env python3
"""ABBA benchmark of NIfTI whole-volume and HDF5 direct-patch loaders."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import platform
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from benchmark_loader_pipeline import MemoryMonitor, atomic_json, describe, request_hash


def parse_args():
    parser = argparse.ArgumentParser(description="UDPET HDF5 direct-patch ABBA benchmark")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--cache-index", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-csv", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--num-batches", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[80, 80, 80])
    parser.add_argument("--stride-size", nargs=3, type=int, default=[20, 20, 20])
    parser.add_argument("--valid-value-threshold", type=float, default=0.2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--steady-state-skip", type=int, default=5)
    parser.add_argument("--memory-poll-seconds", type=float, default=0.05)
    parser.add_argument(
        "--arm-order", nargs="+", choices=("nifti", "hdf5"),
        default=["nifti", "hdf5", "hdf5", "nifti"],
    )
    return parser.parse_args()


def make_dataset(opts, backend, dependencies):
    return dependencies["dataset"](
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


def benchmark_arm(opts, arm, run_index, repetition, model, torch, dependencies):
    dataset = make_dataset(opts, arm, dependencies)
    sampler = dependencies["sampler"](dataset, opts.seed, opts.num_batches)
    dataset.set_epoch(0)
    sampler.set_epoch(0)
    expected_requests = list(iter(sampler))
    loader = dependencies["dataloader"](
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=opts.num_workers,
        pin_memory=True,
        worker_init_fn=dependencies["seed_worker"],
        generator=torch.Generator().manual_seed(opts.seed),
        persistent_workers=True,
        prefetch_factor=opts.prefetch_factor,
    )
    iterator = iter(loader)
    worker_pids = [worker.pid for worker in getattr(iterator, "_workers", [])]
    monitor = MemoryMonitor([os.getpid(), *worker_pids], opts.memory_poll_seconds)
    monitor.start()
    torch.cuda.reset_peak_memory_stats()
    records = []
    observed_requests = []
    run_start = time.perf_counter()
    try:
        for batch_index in range(opts.num_batches):
            fetch_start = time.perf_counter()
            batch = next(iterator)
            fetch_end = time.perf_counter()
            row_index = int(batch["row_index"][0])
            sample_seed = int(batch["sample_seed"][0])
            observed_requests.append((row_index, sample_seed))
            gpu_start = time.perf_counter()
            low = batch["vol_low"].flatten(0, 1).unsqueeze(1).to(
                "cuda:0", dtype=torch.float32, non_blocking=True
            )
            high = batch["vol_high"].flatten(0, 1).unsqueeze(1).to(
                "cuda:0", dtype=torch.float32, non_blocking=True
            )
            with torch.inference_mode():
                prediction = model(low)
                checksum = float((prediction.mean() + high.mean()).item())
            torch.cuda.synchronize()
            gpu_end = time.perf_counter()
            records.append({
                "run_index": run_index,
                "arm": arm,
                "repetition": repetition,
                "batch_index": batch_index,
                "row_index": row_index,
                "sample_seed": sample_seed,
                "count_label": str(batch["count_label"][0]),
                "fetch_seconds": fetch_end - fetch_start,
                "gpu_forward_seconds": gpu_end - gpu_start,
                "checksum": checksum,
            })
            del prediction, low, high, batch
            if (batch_index + 1) % 25 == 0 or batch_index + 1 == opts.num_batches:
                print(
                    f"BENCHMARK_PROGRESS arm={arm} repetition={repetition} "
                    f"batch={batch_index + 1}/{opts.num_batches}", flush=True,
                )
    finally:
        run_seconds = time.perf_counter() - run_start
        monitor.stop()
        if hasattr(iterator, "_shutdown_workers"):
            iterator._shutdown_workers()
        del iterator, loader, sampler, dataset
        gc.collect()

    if observed_requests != expected_requests:
        raise AssertionError(f"Observed DataLoader request order changed for {arm}")
    steady = records[min(opts.steady_state_skip, len(records) - 1):]
    fetch = [record["fetch_seconds"] for record in records]
    fetch_steady = [record["fetch_seconds"] for record in steady]
    gpu = [record["gpu_forward_seconds"] for record in records]
    gpu_steady = [record["gpu_forward_seconds"] for record in steady]
    waiting = sum(fetch_steady)
    consuming = sum(gpu_steady)
    result = {
        "run_index": run_index,
        "arm": arm,
        "repetition": repetition,
        "configuration": {
            "storage_backend": arm,
            "sampling_mode": "volume_grouped_weighted",
            "reference_cache_size": 1,
            "hdf5_handle_cache_size": 1 if arm == "hdf5" else None,
            "persistent_workers": True,
        },
        "batches": len(records),
        "steady_state_skip": opts.steady_state_skip,
        "wall_seconds": run_seconds,
        "pipeline_batches_per_second": len(records) / run_seconds,
        "fetch_all_seconds": describe(fetch),
        "fetch_steady_seconds": describe(fetch_steady),
        "gpu_forward_all_seconds": describe(gpu),
        "gpu_forward_steady_seconds": describe(gpu_steady),
        "gpu_wait_fraction_steady": waiting / (waiting + consuming),
        "request_multiset_sha256": request_hash(observed_requests, False),
        "request_order_sha256": request_hash(observed_requests, True),
        "worker_pids": worker_pids,
        "memory": monitor.result(),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
    print("BENCHMARK_RUN=" + json.dumps(result, sort_keys=True), flush=True)
    return result, records


def summarize(runs, arm):
    selected = [run for run in runs if run["arm"] == arm]
    return {
        "runs": len(selected),
        "median_wall_seconds": statistics.median(run["wall_seconds"] for run in selected),
        "median_pipeline_batches_per_second": statistics.median(
            run["pipeline_batches_per_second"] for run in selected
        ),
        "median_fetch_steady_mean_seconds": statistics.median(
            run["fetch_steady_seconds"]["mean"] for run in selected
        ),
        "median_fetch_steady_p95_seconds": statistics.median(
            run["fetch_steady_seconds"]["p95"] for run in selected
        ),
        "median_gpu_forward_steady_mean_seconds": statistics.median(
            run["gpu_forward_steady_seconds"]["mean"] for run in selected
        ),
        "median_gpu_wait_fraction_steady": statistics.median(
            run["gpu_wait_fraction_steady"] for run in selected
        ),
        "maximum_tree_pss_mib": max(
            run["memory"]["peak_tree_pss_mib"] for run in selected
        ),
    }


def main():
    opts = parse_args()
    if opts.num_batches <= opts.steady_state_skip:
        raise ValueError("num_batches must exceed steady_state_skip")
    if opts.num_workers <= 0:
        raise ValueError("This benchmark requires at least one DataLoader worker")
    counts = Counter(opts.arm_order)
    if counts["nifti"] != counts["hdf5"] or counts["nifti"] == 0:
        raise ValueError("arm-order must contain equal nonzero NIfTI and HDF5 runs")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_id)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from CSVPairDataset import (
        CSVPairDataset, make_volume_grouped_weighted_sampler, seed_worker,
    )
    from nets_GAN import Unet, gaussian_weights_init

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this benchmark")
    torch.manual_seed(opts.seed)
    torch.cuda.manual_seed_all(opts.seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    cudnn.benchmark = False
    cudnn.deterministic = True
    cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    model = Unet(
        inshape=opts.patch_size,
        nb_features=[[48, 96, 192, 384], [384, 192, 96, 48, 24, 1]],
    ).to("cuda:0")
    model.apply(gaussian_weights_init)
    model.eval()
    with torch.inference_mode():
        warmup = torch.zeros(
            (opts.patches_per_volume, 1, *opts.patch_size),
            device="cuda:0", dtype=torch.float32,
        )
        _ = model(warmup)
        torch.cuda.synchronize()
        del warmup, _

    dependencies = {
        "dataset": CSVPairDataset,
        "dataloader": DataLoader,
        "sampler": make_volume_grouped_weighted_sampler,
        "seed_worker": seed_worker,
    }
    repetitions = defaultdict(int)
    runs = []
    batch_records = []
    started = time.perf_counter()
    for run_index, arm in enumerate(opts.arm_order, start=1):
        repetitions[arm] += 1
        run, records = benchmark_arm(
            opts, arm, run_index, repetitions[arm], model, torch, dependencies
        )
        runs.append(run)
        batch_records.extend(records)

    request_hashes = {run["request_multiset_sha256"] for run in runs}
    if len(request_hashes) != 1:
        raise AssertionError("Benchmark arms consumed different request multisets")
    by_request = defaultdict(list)
    for record in batch_records:
        by_request[(record["row_index"], record["sample_seed"])].append(record["checksum"])
    maximum_checksum_spread = max(max(values) - min(values) for values in by_request.values())
    nifti = summarize(runs, "nifti")
    hdf5 = summarize(runs, "hdf5")
    comparison = {
        "pipeline_wall_speedup_nifti_over_hdf5": (
            nifti["median_wall_seconds"] / hdf5["median_wall_seconds"]
        ),
        "steady_fetch_speedup_nifti_over_hdf5": (
            nifti["median_fetch_steady_mean_seconds"]
            / hdf5["median_fetch_steady_mean_seconds"]
        ),
        "gpu_wait_fraction_absolute_change_hdf5_minus_nifti": (
            hdf5["median_gpu_wait_fraction_steady"]
            - nifti["median_gpu_wait_fraction_steady"]
        ),
        "tree_pss_ratio_hdf5_over_nifti": (
            hdf5["maximum_tree_pss_mib"] / nifti["maximum_tree_pss_mib"]
        ),
        "maximum_same_request_checksum_spread": maximum_checksum_spread,
    }
    report = {
        "protocol": "udpet_hdf5_direct_patch_abba_v1_20260917",
        "scope": "engineering diagnostic; pilot train CSV only; no optimizer or test manifest",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "physical_gpu_id": str(opts.gpu_id),
        },
        "configuration": {
            "csv": str(Path(opts.csv).resolve()),
            "cache_index": str(Path(opts.cache_index).resolve()),
            "num_batches_per_run": opts.num_batches,
            "arm_order": opts.arm_order,
            "num_workers": opts.num_workers,
            "prefetch_factor": opts.prefetch_factor,
            "patches_per_volume": opts.patches_per_volume,
            "patch_size": opts.patch_size,
            "stride_size": opts.stride_size,
            "seed": opts.seed,
            "consumer": "H2D low+NORMAL plus deterministic generator forward",
        },
        "request_multiset_sha256": next(iter(request_hashes)),
        "total_benchmark_seconds": time.perf_counter() - started,
        "runs": runs,
        "arm_summary": {"nifti": nifti, "hdf5": hdf5},
        "comparison": comparison,
        "formal_training_started": False,
        "test_manifest_read": False,
    }
    atomic_json(opts.output_json, report)
    batch_path = Path(opts.batch_csv)
    batch_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = batch_path.with_suffix(batch_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(batch_records[0]))
        writer.writeheader()
        writer.writerows(batch_records)
    os.replace(temporary, batch_path)
    print("BENCHMARK_COMPARISON=" + json.dumps(comparison, sort_keys=True), flush=True)
    print(f"BENCHMARK_REPORT={opts.output_json}", flush=True)


if __name__ == "__main__":
    main()
