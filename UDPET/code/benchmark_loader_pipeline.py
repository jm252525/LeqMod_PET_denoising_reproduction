#!/usr/bin/env python3
"""Benchmark legacy and optimized UDPET loaders with a real GPU consumer.

The benchmark uses the same deterministic sample-request multiset for every
arm. ``fetch_seconds`` is the interval during which the synchronous GPU
pipeline has finished the previous forward and is waiting for the next batch;
it is therefore a forward-pipeline starvation measure, not a full-training GPU
utilization measurement.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import statistics
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="UDPET loader ABBA benchmark")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--batch-csv", required=True)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--num-batches", type=int, default=200)
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
        "--arm-order",
        nargs="+",
        choices=("legacy", "optimized"),
        default=["legacy", "optimized", "optimized", "legacy"],
    )
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def percentile(values, quantile):
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("Cannot calculate a percentile of an empty sequence")
    position = (len(ordered) - 1) * float(quantile)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def describe(values):
    values = [float(value) for value in values]
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "maximum": max(values),
        "total": sum(values),
    }


def process_memory_mib(pid):
    """Return Linux RSS/PSS MiB; a vanished worker contributes zero."""
    result = {"rss_mib": 0.0, "pss_mib": 0.0}
    try:
        with Path(f"/proc/{int(pid)}/status").open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    result["rss_mib"] = float(line.split()[1]) / 1024.0
                    break
        rollup = Path(f"/proc/{int(pid)}/smaps_rollup")
        if rollup.is_file():
            with rollup.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("Pss:"):
                        result["pss_mib"] = float(line.split()[1]) / 1024.0
                        break
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        pass
    return result


class MemoryMonitor:
    def __init__(self, pids, interval):
        self.pids = [int(pid) for pid in pids]
        self.main_pid = os.getpid()
        self.interval = float(interval)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.samples = 0
        self.peak_tree_rss_mib = 0.0
        self.peak_tree_pss_mib = 0.0
        self.peak_main_rss_mib = 0.0
        self.peak_worker_rss_sum_mib = 0.0

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, 5.0 * self.interval))

    def _run(self):
        while not self.stop_event.is_set():
            records = {
                pid: process_memory_mib(pid) for pid in self.pids
            }
            rss = sum(record["rss_mib"] for record in records.values())
            pss = sum(record["pss_mib"] for record in records.values())
            main = records.get(
                self.main_pid, {"rss_mib": 0.0}
            )["rss_mib"]
            worker_rss = sum(
                record["rss_mib"]
                for pid, record in records.items()
                if pid != self.main_pid
            )
            self.samples += 1
            self.peak_tree_rss_mib = max(self.peak_tree_rss_mib, rss)
            self.peak_tree_pss_mib = max(self.peak_tree_pss_mib, pss)
            self.peak_main_rss_mib = max(self.peak_main_rss_mib, main)
            self.peak_worker_rss_sum_mib = max(
                self.peak_worker_rss_sum_mib, worker_rss
            )
            self.stop_event.wait(self.interval)

    def result(self):
        return {
            "samples": self.samples,
            "peak_tree_rss_mib": self.peak_tree_rss_mib,
            "peak_tree_pss_mib": self.peak_tree_pss_mib,
            "peak_main_rss_mib": self.peak_main_rss_mib,
            "peak_worker_rss_sum_mib": self.peak_worker_rss_sum_mib,
            "rss_note": "RSS sum double-counts shared pages; tree PSS is the preferred memory comparison",
        }


def request_hash(requests, preserve_order):
    normalized = [f"{int(row)}:{int(seed)}" for row, seed in requests]
    if not preserve_order:
        normalized.sort()
    return hashlib.sha256("\n".join(normalized).encode("ascii")).hexdigest()


def arm_configuration(arm):
    if arm == "legacy":
        return {
            "sampling_mode": "csv_weighted",
            "reference_cache_size": 0,
            "persistent_workers": False,
        }
    if arm == "optimized":
        return {
            "sampling_mode": "volume_grouped_weighted",
            "reference_cache_size": 1,
            "persistent_workers": True,
        }
    raise ValueError(arm)


def make_sampler(arm, dataset, seed, num_batches, factories):
    if arm == "legacy":
        return factories["legacy"](dataset, seed, num_batches)
    return factories["optimized"](dataset, seed, num_batches)


def benchmark_arm(opts, arm, run_index, repetition, model, torch, dependencies):
    CSVPairDataset = dependencies["dataset"]
    DataLoader = dependencies["dataloader"]
    seed_worker = dependencies["seed_worker"]
    configuration = arm_configuration(arm)
    dataset = CSVPairDataset(
        csv_path=opts.train_csv,
        split="train",
        patch_size=opts.patch_size,
        stride_size=opts.stride_size,
        patches_per_volume=opts.patches_per_volume,
        valid_value_threshold=opts.valid_value_threshold,
        min_valid_fraction=opts.min_valid_fraction,
        augmentation=True,
        rotate_degrees=10,
        enable_lemod=False,
        reference_cache_size=configuration["reference_cache_size"],
        seed=opts.seed,
        validate_paths=False,
    )
    sampler = make_sampler(
        arm,
        dataset,
        opts.seed,
        opts.num_batches,
        dependencies["samplers"],
    )
    dataset.set_epoch(0)
    sampler.set_epoch(0)
    expected_requests = list(iter(sampler))
    loader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        sampler=sampler,
        num_workers=opts.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(opts.seed),
        persistent_workers=configuration["persistent_workers"],
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
                # Materialize a scalar so the forward cannot be optimized away.
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
                    f"batch={batch_index + 1}/{opts.num_batches}",
                    flush=True,
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
    fetch_all = [row["fetch_seconds"] for row in records]
    fetch_steady = [row["fetch_seconds"] for row in steady]
    gpu_all = [row["gpu_forward_seconds"] for row in records]
    gpu_steady = [row["gpu_forward_seconds"] for row in steady]
    wait = sum(fetch_steady)
    consumer = sum(gpu_steady)
    result = {
        "run_index": run_index,
        "arm": arm,
        "repetition": repetition,
        "configuration": configuration,
        "batches": len(records),
        "steady_state_skip": opts.steady_state_skip,
        "wall_seconds": run_seconds,
        "pipeline_batches_per_second": len(records) / run_seconds,
        "fetch_all_seconds": describe(fetch_all),
        "fetch_steady_seconds": describe(fetch_steady),
        "gpu_forward_all_seconds": describe(gpu_all),
        "gpu_forward_steady_seconds": describe(gpu_steady),
        "gpu_wait_fraction_steady": wait / (wait + consumer),
        "request_multiset_sha256": request_hash(observed_requests, False),
        "request_order_sha256": request_hash(observed_requests, True),
        "worker_pids": worker_pids,
        "memory": monitor.result(),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024 ** 2),
        "gpu_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024 ** 2),
    }
    print("BENCHMARK_RUN=" + json.dumps(result, sort_keys=True), flush=True)
    return result, records


def arm_summary(runs, arm):
    selected = [run for run in runs if run["arm"] == arm]
    return {
        "runs": len(selected),
        "median_wall_seconds": statistics.median(
            run["wall_seconds"] for run in selected
        ),
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
        "maximum_tree_rss_mib": max(
            run["memory"]["peak_tree_rss_mib"] for run in selected
        ),
    }


def main():
    opts = parse_args()
    if opts.num_batches <= opts.steady_state_skip:
        raise ValueError("num_batches must be greater than steady_state_skip")
    if opts.num_workers <= 0:
        raise ValueError("This benchmark requires at least one DataLoader worker")
    if opts.prefetch_factor <= 0 or opts.memory_poll_seconds <= 0:
        raise ValueError("prefetch factor and memory poll interval must be positive")
    counts = Counter(opts.arm_order)
    if counts["legacy"] != counts["optimized"] or counts["legacy"] == 0:
        raise ValueError("arm-order must contain the same nonzero number of both arms")

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(opts.gpu_id)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader

    from CSVPairDataset import (
        CSVPairDataset,
        make_volume_grouped_weighted_sampler,
        make_weighted_sampler,
        seed_worker,
    )
    from nets_GAN import Unet, gaussian_weights_init

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the forward-pipeline benchmark")
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
            device="cuda:0",
            dtype=torch.float32,
        )
        _ = model(warmup)
        torch.cuda.synchronize()
        del warmup, _

    dependencies = {
        "dataset": CSVPairDataset,
        "dataloader": DataLoader,
        "seed_worker": seed_worker,
        "samplers": {
            "legacy": make_weighted_sampler,
            "optimized": make_volume_grouped_weighted_sampler,
        },
    }
    repetitions = defaultdict(int)
    runs = []
    batch_records = []
    benchmark_start = time.perf_counter()
    for run_index, arm in enumerate(opts.arm_order, start=1):
        repetitions[arm] += 1
        run, records = benchmark_arm(
            opts,
            arm,
            run_index,
            repetitions[arm],
            model,
            torch,
            dependencies,
        )
        runs.append(run)
        batch_records.extend(records)
    total_seconds = time.perf_counter() - benchmark_start

    request_hashes = {run["request_multiset_sha256"] for run in runs}
    if len(request_hashes) != 1:
        raise AssertionError("Arms did not consume the same request multiset")
    legacy = arm_summary(runs, "legacy")
    optimized = arm_summary(runs, "optimized")
    comparison = {
        "pipeline_wall_speedup_legacy_over_optimized": (
            legacy["median_wall_seconds"] / optimized["median_wall_seconds"]
        ),
        "steady_fetch_speedup_legacy_over_optimized": (
            legacy["median_fetch_steady_mean_seconds"]
            / optimized["median_fetch_steady_mean_seconds"]
        ),
        "gpu_wait_fraction_absolute_change_optimized_minus_legacy": (
            optimized["median_gpu_wait_fraction_steady"]
            - legacy["median_gpu_wait_fraction_steady"]
        ),
        "tree_pss_ratio_optimized_over_legacy": (
            optimized["maximum_tree_pss_mib"] / legacy["maximum_tree_pss_mib"]
        ),
    }
    report = {
        "protocol": "udpet_loader_abba_gpu_forward_v1_20260917",
        "scope": "engineering diagnostic; no optimizer update; no test manifest",
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "physical_gpu_id": str(opts.gpu_id),
        },
        "configuration": {
            "train_csv": str(Path(opts.train_csv).resolve()),
            "num_batches_per_run": opts.num_batches,
            "arm_order": opts.arm_order,
            "num_workers": opts.num_workers,
            "prefetch_factor": opts.prefetch_factor,
            "patches_per_volume": opts.patches_per_volume,
            "patch_size": opts.patch_size,
            "stride_size": opts.stride_size,
            "augmentation": True,
            "seed": opts.seed,
            "steady_state_skip": opts.steady_state_skip,
            "consumer": "H2D low+NORMAL plus deterministic generator forward",
        },
        "request_multiset_sha256": next(iter(request_hashes)),
        "total_benchmark_seconds": total_seconds,
        "runs": runs,
        "arm_summary": {"legacy": legacy, "optimized": optimized},
        "comparison": comparison,
        "interpretation_boundary": (
            "fetch wait is a synchronous generator-forward pipeline starvation "
            "measure; full adversarial backward is slower and was not repeated "
            "for hundreds of diagnostic batches"
        ),
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
