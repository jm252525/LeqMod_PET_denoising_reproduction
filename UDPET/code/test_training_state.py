"""Unit tests for strict UDPET checkpoint and RNG contracts."""

from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from training_state import (
    CHECKPOINT_VERSION,
    canonical_sha256,
    capture_rng_state,
    restore_rng_state,
    validate_checkpoint_contract,
)


class TrainingStateTest(unittest.TestCase):
    def test_rng_state_round_trip(self):
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        state = capture_rng_state()
        expected = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )
        random.seed(99)
        np.random.seed(99)
        torch.manual_seed(99)
        restore_rng_state(state, strict_cuda=False)
        observed = (
            random.random(),
            float(np.random.random()),
            torch.rand(4),
        )
        self.assertEqual(expected[0], observed[0])
        self.assertEqual(expected[1], observed[1])
        torch.testing.assert_close(expected[2], observed[2], rtol=0, atol=0)

    def test_contract_hash_is_order_independent(self):
        self.assertEqual(canonical_sha256({"a": 1, "b": 2}), canonical_sha256({"b": 2, "a": 1}))

    def test_strict_contract_rejects_missing_state(self):
        expected = canonical_sha256({"protocol": "test"})
        checkpoint = {"checkpoint_version": CHECKPOINT_VERSION, "training_state": {}}
        with self.assertRaisesRegex(ValueError, "Strict resume rejected"):
            validate_checkpoint_contract(checkpoint, expected)
        problems = validate_checkpoint_contract(checkpoint, expected, allow_inexact=True)
        self.assertTrue(problems)


if __name__ == "__main__":
    unittest.main()
