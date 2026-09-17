"""Deterministic, patient-level quantitative validation for UDPET restoration."""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


METRIC_NAMES = (
    "body_mae_suv",
    "body_rmse_suv",
    "body_psnr_suv30_db",
    "body_target_suvmean",
    "body_pred_suvmean",
    "body_suvmean_bias_percent",
    "body_suvmean_abs_bias_percent",
    "hotspot_target_suvmean",
    "hotspot_pred_suvmean",
    "hotspot_suvmean_bias_percent",
    "hotspot_suvmean_abs_bias_percent",
)


def _new_accumulator(site, cohort, patient_id, count_label):
    return {
        "site": site,
        "cohort": cohort,
        "patient_id": patient_id,
        "count_label": count_label,
        "study_ids": set(),
        "row_indices": set(),
        "sample_seeds": set(),
        "n_patches": 0,
        "body_voxels": 0,
        "body_abs_error_sum": 0.0,
        "body_squared_error_sum": 0.0,
        "body_target_sum": 0.0,
        "body_pred_sum": 0.0,
        "hotspot_voxels": 0,
        "hotspot_target_sum": 0.0,
        "hotspot_pred_sum": 0.0,
    }


def accumulate_patch_batch(accumulator, prediction, target, body_threshold):
    """Accumulate patch tensors without computing patch-level metrics.

    ``prediction`` and ``target`` have shape ``(patches, x, y, z)``. The body
    mask and hotspot ranking are reference-defined so model output cannot move
    the evaluation target. Hotspots are the top one percent of body voxels in
    each sampled reference patch; they are not lesion annotations.
    """
    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError(
            f"Expected matching (patches,x,y,z), got {prediction.shape} and {target.shape}"
        )
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Non-finite prediction or target during validation")

    for patch_prediction, patch_target in zip(prediction, target):
        body = patch_target > float(body_threshold)
        body_count = int(body.sum().item())
        if body_count == 0:
            continue
        pred_body = patch_prediction[body].double()
        target_body = patch_target[body].double()
        error = pred_body - target_body
        accumulator["n_patches"] += 1
        accumulator["body_voxels"] += body_count
        accumulator["body_abs_error_sum"] += float(error.abs().sum().item())
        accumulator["body_squared_error_sum"] += float(error.square().sum().item())
        accumulator["body_target_sum"] += float(target_body.sum().item())
        accumulator["body_pred_sum"] += float(pred_body.sum().item())

        hotspot_count = max(1, int(math.ceil(0.01 * body_count)))
        hotspot_indices = torch.topk(target_body, hotspot_count, sorted=False).indices
        accumulator["hotspot_voxels"] += hotspot_count
        accumulator["hotspot_target_sum"] += float(target_body[hotspot_indices].sum().item())
        accumulator["hotspot_pred_sum"] += float(pred_body[hotspot_indices].sum().item())


def finalize_accumulator(accumulator, psnr_data_range=30.0, epsilon=1e-8):
    body_count = int(accumulator["body_voxels"])
    hotspot_count = int(accumulator["hotspot_voxels"])
    if body_count <= 0 or hotspot_count <= 0:
        raise ValueError("Validation accumulator contains no evaluable body voxels")

    body_mae = accumulator["body_abs_error_sum"] / body_count
    body_rmse = math.sqrt(accumulator["body_squared_error_sum"] / body_count)
    body_target_mean = accumulator["body_target_sum"] / body_count
    body_pred_mean = accumulator["body_pred_sum"] / body_count
    body_bias = 100.0 * (body_pred_mean - body_target_mean) / max(
        abs(body_target_mean), epsilon
    )
    hotspot_target_mean = accumulator["hotspot_target_sum"] / hotspot_count
    hotspot_pred_mean = accumulator["hotspot_pred_sum"] / hotspot_count
    hotspot_bias = 100.0 * (hotspot_pred_mean - hotspot_target_mean) / max(
        abs(hotspot_target_mean), epsilon
    )

    return {
        "site": accumulator["site"],
        "cohort": accumulator["cohort"],
        "patient_id": accumulator["patient_id"],
        "count_label": accumulator["count_label"],
        "n_studies": len(accumulator["study_ids"]),
        "row_indices": ";".join(map(str, sorted(accumulator["row_indices"]))),
        "sample_seeds": ";".join(map(str, sorted(accumulator["sample_seeds"]))),
        "n_patches": int(accumulator["n_patches"]),
        "body_voxels": body_count,
        "hotspot_voxels": hotspot_count,
        "body_mae_suv": body_mae,
        "body_rmse_suv": body_rmse,
        "body_psnr_suv30_db": 20.0 * math.log10(
            float(psnr_data_range) / max(body_rmse, epsilon)
        ),
        "body_target_suvmean": body_target_mean,
        "body_pred_suvmean": body_pred_mean,
        "body_suvmean_bias_percent": body_bias,
        "body_suvmean_abs_bias_percent": abs(body_bias),
        "hotspot_target_suvmean": hotspot_target_mean,
        "hotspot_pred_suvmean": hotspot_pred_mean,
        "hotspot_suvmean_bias_percent": hotspot_bias,
        "hotspot_suvmean_abs_bias_percent": abs(hotspot_bias),
    }


def _patient_balanced_rows(rows):
    """Average DRF rows within patient before an overall patient bootstrap."""
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["site"], row["patient_id"])].append(row)
    balanced = []
    for (site, patient_id), patient_rows in sorted(grouped.items()):
        item = {"site": site, "patient_id": patient_id}
        for metric in METRIC_NAMES:
            item[metric] = float(np.mean([float(row[metric]) for row in patient_rows]))
        balanced.append(item)
    return balanced


def _bootstrap_metrics(rows, replicates, seed):
    if not rows:
        raise ValueError("Cannot summarize an empty validation group")
    if int(replicates) <= 0:
        raise ValueError("bootstrap replicates must be positive")
    values = np.asarray(
        [[float(row[metric]) for metric in METRIC_NAMES] for row in rows],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(values)):
        raise FloatingPointError("Non-finite patient metric before bootstrap")
    rng = np.random.default_rng(int(seed))
    sample_indices = rng.integers(0, len(rows), size=(int(replicates), len(rows)))
    bootstrap_means = values[sample_indices].mean(axis=1)
    result = {}
    for metric_index, metric in enumerate(METRIC_NAMES):
        result[metric] = {
            "mean": float(values[:, metric_index].mean()),
            "ci95_low": float(np.quantile(bootstrap_means[:, metric_index], 0.025)),
            "ci95_high": float(np.quantile(bootstrap_means[:, metric_index], 0.975)),
        }
    return result


def summarize_patient_metrics(rows, bootstrap_replicates=1000, seed=20260910):
    by_count = defaultdict(list)
    for row in rows:
        by_count[row["count_label"]].append(row)
    count_summaries = {}
    for offset, count_label in enumerate(sorted(by_count)):
        count_rows = by_count[count_label]
        count_summaries[count_label] = {
            "patients": len({(row["site"], row["patient_id"]) for row in count_rows}),
            "patient_drf_rows": len(count_rows),
            "metrics": _bootstrap_metrics(
                count_rows, bootstrap_replicates, int(seed) + offset + 1
            ),
        }
    overall_rows = _patient_balanced_rows(rows)
    return {
        "aggregation": {
            "primary_unit": "patient",
            "per_drf": "one pooled patch-voxel record per patient and DRF",
            "overall": "equal-weight DRF mean within patient, then equal-weight patients",
            "bootstrap_unit": "patient",
        },
        "bootstrap_replicates": int(bootstrap_replicates),
        "bootstrap_seed": int(seed),
        "patient_drf_rows": len(rows),
        "unique_patients": len(overall_rows),
        "by_count_level": count_summaries,
        "overall_patient_balanced": {
            "patients": len(overall_rows),
            "metrics": _bootstrap_metrics(
                overall_rows, bootstrap_replicates, int(seed)
            ),
        },
    }


def write_patient_metrics_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "site", "cohort", "patient_id", "count_label", "n_studies",
        "row_indices", "sample_seeds",
        "n_patches", "body_voxels", "hotspot_voxels", *METRIC_NAMES,
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def evaluate_model(
    model,
    loader,
    body_threshold,
    bootstrap_replicates,
    bootstrap_seed,
):
    """Run fixed-patch validation and return patient rows plus summary."""
    network = model.net_G
    was_training = network.training
    network.eval()
    accumulators = {}
    try:
        with torch.no_grad():
            for batch in loader:
                low = batch["vol_low"].to(model.opts.device).float()
                target = batch["vol_high"].to(model.opts.device).float()
                if low.ndim != 5 or target.shape != low.shape:
                    raise ValueError(
                        f"Expected (batch,patches,x,y,z), got {low.shape} and {target.shape}"
                    )
                batch_size, patches = low.shape[:2]
                prediction = network(low.flatten(0, 1).unsqueeze(1)).squeeze(1)
                prediction = prediction.reshape(batch_size, patches, *low.shape[2:])
                if not torch.isfinite(prediction).all():
                    raise FloatingPointError("Non-finite model output during validation")

                for batch_index in range(batch_size):
                    site = str(batch["center"][batch_index])
                    cohort = str(batch["cohort"][batch_index])
                    patient_id = str(batch["patient_id"][batch_index])
                    count_label = str(batch["count_label"][batch_index])
                    key = (site, patient_id, count_label)
                    if key not in accumulators:
                        accumulators[key] = _new_accumulator(
                            site, cohort, patient_id, count_label
                        )
                    accumulator = accumulators[key]
                    accumulator["study_ids"].add(str(batch["study_id"][batch_index]))
                    accumulator["row_indices"].add(int(batch["row_index"][batch_index]))
                    accumulator["sample_seeds"].add(int(batch["sample_seed"][batch_index]))
                    accumulate_patch_batch(
                        accumulator,
                        prediction[batch_index],
                        target[batch_index],
                        body_threshold,
                    )
    finally:
        network.train(was_training)

    rows = [
        finalize_accumulator(accumulators[key]) for key in sorted(accumulators)
    ]
    summary = summarize_patient_metrics(
        rows,
        bootstrap_replicates=bootstrap_replicates,
        seed=bootstrap_seed,
    )
    return rows, summary
