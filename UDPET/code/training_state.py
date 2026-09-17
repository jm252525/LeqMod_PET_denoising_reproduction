"""Strict epoch-boundary training-state helpers for UDPET runs."""

from __future__ import annotations

import hashlib
import json
import random

import numpy as np
import torch


CHECKPOINT_VERSION = 2


def canonical_sha256(payload):
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "state": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state, strict_cuda=True):
    if not state:
        raise ValueError("Checkpoint does not contain RNG state")
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((
        numpy_state["bit_generator"],
        numpy_state["state"].cpu().numpy().astype(np.uint32, copy=False),
        int(numpy_state["position"]),
        int(numpy_state["has_gauss"]),
        float(numpy_state["cached_gaussian"]),
    ))
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda") or []
    if torch.cuda.is_available():
        if strict_cuda and len(cuda_states) != torch.cuda.device_count():
            raise ValueError(
                "CUDA RNG device count mismatch: checkpoint has "
                f"{len(cuda_states)}, current run has {torch.cuda.device_count()}"
            )
        if cuda_states:
            torch.cuda.set_rng_state_all([cuda_state.cpu() for cuda_state in cuda_states])


def validate_checkpoint_contract(checkpoint, expected_sha256, allow_inexact=False):
    version = int(checkpoint.get("checkpoint_version", 0))
    training_state = checkpoint.get("training_state") or {}
    observed_sha256 = training_state.get("training_contract_sha256")
    problems = []
    if version < CHECKPOINT_VERSION:
        problems.append(
            f"checkpoint_version={version}, required>={CHECKPOINT_VERSION}"
        )
    if observed_sha256 != expected_sha256:
        problems.append(
            "training contract mismatch: "
            f"checkpoint={observed_sha256!r}, current={expected_sha256!r}"
        )
    for key in ("rng_state", "loader_generator_state", "sampler_epoch"):
        if key not in training_state:
            problems.append(f"missing training_state.{key}")
    if problems and not allow_inexact:
        raise ValueError("Strict resume rejected: " + "; ".join(problems))
    return problems
