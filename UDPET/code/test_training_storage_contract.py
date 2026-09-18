"""Unit tests for storage identity in the strict training contract."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from train_LeqModGan_csv import build_storage_contract
from training_state import canonical_sha256


class TrainingStorageContractTest(unittest.TestCase):
    def test_hdf5_contract_binds_complete_train_and_validation_indexes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            train_index = root / "train.json"
            val_index = root / "val.json"
            train_index.write_bytes(b'{"split":"train","rows":1}')
            val_index.write_bytes(b'{"split":"val","rows":2}')

            contract = build_storage_contract(
                "hdf5", train_index, val_index, hdf5_handle_cache_size=2
            )

            self.assertEqual(contract["backend"], "hdf5")
            self.assertEqual(
                contract["train_chunk_cache_index_sha256"],
                hashlib.sha256(train_index.read_bytes()).hexdigest(),
            )
            self.assertEqual(
                contract["val_chunk_cache_index_sha256"],
                hashlib.sha256(val_index.read_bytes()).hexdigest(),
            )
            self.assertEqual(contract["hdf5_handle_cache_size_per_worker"], 2)

            original_hash = canonical_sha256(contract)
            train_index.write_bytes(b'{"split":"train","rows":3}')
            changed = build_storage_contract("hdf5", train_index, val_index, 2)
            self.assertNotEqual(original_hash, canonical_sha256(changed))

    def test_hdf5_loader_only_contract_allows_no_validation_index(self):
        with tempfile.TemporaryDirectory() as temporary:
            train_index = Path(temporary) / "train.json"
            train_index.write_text("{}", encoding="utf-8")
            contract = build_storage_contract("hdf5", train_index)
            self.assertIsNone(contract["val_chunk_cache_index_sha256"])

    def test_nifti_contract_rejects_hdf5_index(self):
        with self.assertRaisesRegex(ValueError, "only be supplied"):
            build_storage_contract("nifti", "unused.json")

    def test_hdf5_contract_requires_train_index(self):
        with self.assertRaisesRegex(ValueError, "chunk-cache-index is required"):
            build_storage_contract("hdf5")


if __name__ == "__main__":
    unittest.main()
