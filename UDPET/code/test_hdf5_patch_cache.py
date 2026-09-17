"""Regression tests for exact HDF5 direct-patch equivalence."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

try:
    import h5py  # noqa: F401
except ImportError:
    h5py = None

from CSVPairDataset import CSVPairDataset
from build_hdf5_patch_cache import PROTOCOL, inspect_study, write_study


@unittest.skipUnless(h5py is not None, "h5py is not installed")
class HDF5PatchCacheTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        affine = np.eye(4, dtype=np.float32)
        grid = np.arange(12 * 12 * 12, dtype=np.float32).reshape(12, 12, 12) / 100.0
        grid[:1] = -1.0
        self.full_path = self.root / "normal.nii.gz"
        self.low_paths = [self.root / "d4.nii.gz", self.root / "d10.nii.gz"]
        nib.save(nib.Nifti1Image(grid, affine), self.full_path)
        nib.save(nib.Nifti1Image(grid * 0.8, affine), self.low_paths[0])
        nib.save(nib.Nifti1Image(grid * 0.9, affine), self.low_paths[1])
        self.csv_path = self.root / "train.csv"
        self.fieldnames = [
            "site", "cohort", "patient_id", "study_id", "split", "unit",
            "count_label", "count_percent", "low_count_path", "full_count_path",
            "pair_qc_status", "leqmod_sampling_weight",
        ]
        self.rows = []
        for label, percent, low_path, weight in (
            ("D4", 25.0, self.low_paths[0], 1.0),
            ("D10", 10.0, self.low_paths[1], 2.0),
        ):
            self.rows.append({
                "site": "synthetic", "cohort": "synthetic",
                "patient_id": "patient-1", "study_id": "study-1",
                "split": "train", "unit": "SUVbw", "count_label": label,
                "count_percent": str(percent), "low_count_path": str(low_path),
                "full_count_path": str(self.full_path), "pair_qc_status": "passed",
                "leqmod_sampling_weight": str(weight),
            })
        with self.csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)

        self.configuration = {
            "patch_size": [4, 4, 4],
            "stride_size": [2, 2, 2],
            "patches_per_volume": 2,
            "valid_value_threshold": 0.2,
            "min_valid_fraction": 0.2,
            "chunk_size": [4, 4, 4],
            "compression": "lzf",
            "gzip_level": 1,
            "stored_dtype": "float32",
            "stored_extent": "full_volume",
            "nonpositive_values_clamped_to_zero": True,
            "candidate_coordinates": "relative_to_NORMAL_derived_crop_box",
        }
        self.cache_path = self.root / "cache.h5"
        write_study(self.rows, self.cache_path, self.configuration)
        self.index_path = self.root / "index.json"
        index = {
            "protocol": PROTOCOL,
            "configuration": self.configuration,
            "rows": {
                row["low_count_path"]: {
                    "cache_path": str(self.cache_path),
                    "low_dataset": f"low/{row['count_label']}",
                    "full_count_path": row["full_count_path"],
                    "count_label": row["count_label"],
                }
                for row in self.rows
            },
        }
        self.index_path.write_text(json.dumps(index), encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def make_dataset(self, backend, augmentation):
        return CSVPairDataset(
            self.csv_path,
            patch_size=(4, 4, 4),
            stride_size=(2, 2, 2),
            patches_per_volume=2,
            augmentation=augmentation,
            rotate_degrees=10,
            reference_cache_size=1,
            storage_backend=backend,
            chunk_cache_index=self.index_path if backend == "hdf5" else None,
            hdf5_handle_cache_size=1,
            validate_paths=True,
        )

    def test_exact_patch_equivalence_without_augmentation(self):
        nifti = self.make_dataset("nifti", augmentation=False)
        cached = self.make_dataset("hdf5", augmentation=False)
        for request in ((0, 17), (1, 23)):
            source = nifti[request]
            direct = cached[request]
            torch.testing.assert_close(source["vol_low"], direct["vol_low"], rtol=0, atol=0)
            torch.testing.assert_close(source["vol_high"], direct["vol_high"], rtol=0, atol=0)
            torch.testing.assert_close(source["vol_weight"], direct["vol_weight"], rtol=0, atol=0)
            self.assertEqual(source["crop_box"], direct["crop_box"])

    def test_exact_patch_equivalence_with_deterministic_rotation(self):
        nifti = self.make_dataset("nifti", augmentation=True)
        cached = self.make_dataset("hdf5", augmentation=True)
        for seed in range(1, 30):
            source = nifti[(0, seed)]
            direct = cached[(0, seed)]
            torch.testing.assert_close(source["vol_low"], direct["vol_low"], rtol=0, atol=0)
            torch.testing.assert_close(source["vol_high"], direct["vol_high"], rtol=0, atol=0)

    def test_cache_configuration_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "patch_size mismatch"):
            CSVPairDataset(
                self.csv_path,
                patch_size=(6, 4, 4),
                stride_size=(2, 2, 2),
                patches_per_volume=2,
                augmentation=False,
                storage_backend="hdf5",
                chunk_cache_index=self.index_path,
            )

    def test_completed_study_can_be_validated_for_resume(self):
        record = inspect_study(self.rows, self.cache_path, self.configuration)
        self.assertEqual(record["shape"], [12, 12, 12])
        self.assertGreater(record["candidate_boxes"], 2)
        self.assertGreater(record["cache_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
