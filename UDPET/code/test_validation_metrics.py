#!/usr/bin/env python3
"""Regression tests for patient-level UDPET validation aggregation."""

import copy

import torch

from validation import (
    _new_accumulator,
    accumulate_patch_batch,
    finalize_accumulator,
    summarize_patient_metrics,
)


def make_row(site, patient, count_label, error):
    accumulator = _new_accumulator(site, site, patient, count_label)
    accumulator["study_ids"].add("study")
    target = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]])
    prediction = target + float(error)
    accumulate_patch_batch(accumulator, prediction, target, body_threshold=0.2)
    return finalize_accumulator(accumulator)


def test_voxel_pooling_before_metrics():
    accumulator = _new_accumulator("site", "cohort", "patient", "D10")
    accumulator["study_ids"].add("study")
    target = torch.ones((2, 1, 1, 2))
    prediction = torch.tensor([[[[2.0, 2.0]]], [[[3.0, 3.0]]]])
    accumulate_patch_batch(accumulator, prediction, target, body_threshold=0.2)
    row = finalize_accumulator(accumulator)
    assert row["n_patches"] == 2
    assert row["body_voxels"] == 4
    assert abs(row["body_mae_suv"] - 1.5) < 1e-12
    assert abs(row["body_rmse_suv"] - (2.5 ** 0.5)) < 1e-12
    assert abs(row["body_suvmean_bias_percent"] - 150.0) < 1e-12


def test_patient_bootstrap_is_deterministic_and_patient_balanced():
    rows = [
        make_row("A", "p1", "D4", 0.1),
        make_row("A", "p1", "D10", 0.3),
        make_row("A", "p2", "D4", 0.5),
    ]
    first = summarize_patient_metrics(rows, bootstrap_replicates=200, seed=17)
    second = summarize_patient_metrics(copy.deepcopy(rows), bootstrap_replicates=200, seed=17)
    assert first == second
    assert first["unique_patients"] == 2
    expected = ((0.1 + 0.3) / 2.0 + 0.5) / 2.0
    observed = first["overall_patient_balanced"]["metrics"]["body_mae_suv"]["mean"]
    assert abs(observed - expected) < 1e-6


def main():
    test_voxel_pooling_before_metrics()
    test_patient_bootstrap_is_deterministic_and_patient_balanced()
    print("PASS: patient/DRF pooling and deterministic patient bootstrap")


if __name__ == "__main__":
    main()
