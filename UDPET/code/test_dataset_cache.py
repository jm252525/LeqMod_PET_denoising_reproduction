"""Regression tests for UDPET reference caching and locality-aware sampling."""

from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from CSVPairDataset import (
    CSVPairDataset,
    VolumeGroupedWeightedSampler,
    make_epoch_shuffle_sampler,
    make_weighted_sampler,
)


class _SamplerDataset:
    def __init__(self):
        self.groups = ("study-a", "study-a", "study-b", "study-b")
        self.weights = (1.0, 2.0, 3.0, 4.0)

    def __len__(self):
        return len(self.groups)

    def sampling_weights(self):
        return self.weights

    def group_key(self, index):
        return self.groups[index]


class DatasetCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        affine = np.eye(4, dtype=np.float32)
        grid = np.arange(8 * 8 * 8, dtype=np.float32).reshape(8, 8, 8) / 100.0 + 1.0
        self.full_path = self.root / "normal.nii.gz"
        self.low_paths = [self.root / "d4.nii.gz", self.root / "d10.nii.gz"]
        nib.save(nib.Nifti1Image(grid, affine), self.full_path)
        nib.save(nib.Nifti1Image(grid * 0.8, affine), self.low_paths[0])
        nib.save(nib.Nifti1Image(grid * 0.9, affine), self.low_paths[1])
        self.csv_path = self.root / "train.csv"
        fieldnames = [
            "site", "cohort", "patient_id", "study_id", "split", "unit",
            "count_label", "count_percent", "low_count_path", "full_count_path",
            "pair_qc_status", "leqmod_sampling_weight",
        ]
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for label, percent, low_path, weight in (
                ("D4", 25.0, self.low_paths[0], 1.0),
                ("D10", 10.0, self.low_paths[1], 2.0),
            ):
                writer.writerow({
                    "site": "synthetic", "cohort": "synthetic",
                    "patient_id": "patient-1", "study_id": "study-1",
                    "split": "train", "unit": "SUVbw", "count_label": label,
                    "count_percent": percent, "low_count_path": low_path,
                    "full_count_path": self.full_path, "pair_qc_status": "passed",
                    "leqmod_sampling_weight": weight,
                })

    def tearDown(self):
        self.temporary.cleanup()

    def make_dataset(self, cache_size):
        return CSVPairDataset(
            self.csv_path,
            patch_size=(4, 4, 4),
            stride_size=(4, 4, 4),
            patches_per_volume=1,
            augmentation=False,
            reference_cache_size=cache_size,
        )

    def test_shared_normal_reference_is_loaded_once(self):
        dataset = self.make_dataset(cache_size=1)
        first = dataset[(0, 7)]
        second = dataset[(1, 7)]
        stats = dataset.io_stats()
        self.assertEqual(stats["full_loads"], 1)
        self.assertEqual(stats["low_loads"], 2)
        self.assertEqual(stats["reference_cache_hits"], 1)
        self.assertEqual(stats["reference_cache_misses"], 1)
        torch.testing.assert_close(first["vol_high"], second["vol_high"])

    def test_zero_cache_preserves_values_but_reloads_reference(self):
        cached = self.make_dataset(cache_size=1)
        uncached = self.make_dataset(cache_size=0)
        np.random.seed(11)
        cached_item = cached[1]
        np.random.seed(11)
        uncached_item = uncached[1]
        torch.testing.assert_close(cached_item["vol_low"], uncached_item["vol_low"])
        torch.testing.assert_close(cached_item["vol_high"], uncached_item["vol_high"])
        uncached[0]
        self.assertEqual(uncached.io_stats()["full_loads"], 2)
        self.assertEqual(uncached.io_stats()["reference_cache_hits"], 0)

    def test_grouped_sampler_preserves_weighted_draw_multiset(self):
        dataset = _SamplerDataset()
        seed = 23
        num_samples = 100
        sampler = VolumeGroupedWeightedSampler(dataset, seed, num_samples)
        grouped_requests = list(sampler)
        grouped_draw = [row_index for row_index, _ in grouped_requests]

        generator = torch.Generator().manual_seed(seed)
        direct_draw = torch.multinomial(
            torch.as_tensor(dataset.weights, dtype=torch.double),
            num_samples,
            replacement=True,
            generator=generator,
        ).tolist()
        self.assertEqual(Counter(grouped_draw), Counter(direct_draw))

        observed_groups = [dataset.group_key(index) for index in grouped_draw]
        group_runs = [
            group for position, group in enumerate(observed_groups)
            if position == 0 or group != observed_groups[position - 1]
        ]
        self.assertEqual(len(group_runs), len(set(group_runs)))
        self.assertEqual(
            grouped_requests,
            list(VolumeGroupedWeightedSampler(dataset, seed, num_samples)),
        )
        sampler.set_epoch(1)
        self.assertNotEqual(grouped_requests, list(sampler))

    def test_sample_request_is_independent_of_global_rng_state(self):
        dataset = self.make_dataset(cache_size=1)
        dataset.augmentation = True
        request = (0, 123456)
        np.random.seed(1)
        first = dataset[request]
        np.random.seed(999)
        second = dataset[request]
        torch.testing.assert_close(first["vol_low"], second["vol_low"])
        torch.testing.assert_close(first["vol_high"], second["vol_high"])
        self.assertEqual(first["sample_seed"], second["sample_seed"])

    def test_all_sampler_modes_are_epoch_deterministic(self):
        dataset = _SamplerDataset()
        factories = (
            lambda: make_weighted_sampler(dataset, 31, 4),
            lambda: VolumeGroupedWeightedSampler(dataset, 31, 4),
            lambda: make_epoch_shuffle_sampler(dataset, 31, 4),
        )
        for factory in factories:
            first = factory()
            epoch_zero = list(first)
            self.assertEqual(epoch_zero, list(factory()))
            first.set_epoch(1)
            self.assertNotEqual(epoch_zero, list(first))


if __name__ == "__main__":
    unittest.main()
