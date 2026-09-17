"""One real eight-patch UDPET optimizer step plus strict checkpoint round-trip."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import tempfile
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from CSVPairDataset import CSVPairDataset, make_epoch_shuffle_sampler
from model_LeqModGan import modelGAN
from training_state import canonical_sha256, capture_rng_state, restore_rng_state


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser.parse_args()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def module_digest(module):
    digest = hashlib.sha256()
    state = modelGAN._unwrap(module).state_dict()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def model_options(device):
    return Namespace(
        patch_size=[80, 80, 80],
        device=device,
        numGPUs=1 if device.type == "cuda" else 0,
        lr=1e-4,
        preWeight=None,
        weightLoss_mse=10.0,
        weightLoss_localSUVbias=2.0,
        weightLoss_lesionSUVbias=None,
        segProbThresh=0.2,
        lr_policy="ReduceLROnPlateau",
        gamma=0.1,
        Plateau_step_size=10,
        multi_step_size=[1],
        n_epochs=2,
    )


def main():
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA smoke test requested but CUDA is unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    dataset = CSVPairDataset(
        args.train_csv,
        split="train",
        patch_size=(80, 80, 80),
        stride_size=(20, 20, 20),
        patches_per_volume=8,
        augmentation=False,
        enable_lemod=False,
        reference_cache_size=1,
        seed=args.seed,
    )
    sampler = make_epoch_shuffle_sampler(dataset, args.seed, num_samples=1)
    sampler.set_epoch(0)
    loader_generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        num_workers=0,
        generator=loader_generator,
    )
    batch = next(iter(loader))
    if batch["vol_low"].shape != (1, 8, 80, 80, 80):
        raise AssertionError(f"Unexpected low-count batch shape: {batch['vol_low'].shape}")
    if not torch.isfinite(batch["vol_low"]).all() or not torch.isfinite(batch["vol_high"]).all():
        raise FloatingPointError("Real input batch contains NaN/Inf")

    opts = model_options(device)
    model = modelGAN(opts)
    model.initialize()
    model.set_scheduler(opts)
    model.train()
    model.set_epoch(0)
    model.set_input(batch)
    model_input_shape = list(model.inp_vol_low.shape)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    model.optimize(total_iter=1)
    losses = model.get_current_losses()
    if not all(np.isfinite(value) for value in losses.values()):
        raise FloatingPointError(f"Non-finite optimizer-step loss: {losses}")
    learning_rate = model.update_learning_rate(losses["loss_recon"])
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_allocated_mib = torch.cuda.max_memory_allocated(device) / 1048576
        peak_reserved_mib = torch.cuda.max_memory_reserved(device) / 1048576
    else:
        peak_allocated_mib = None
        peak_reserved_mib = None

    generator_digest = module_digest(model.net_G)
    discriminator_digest = module_digest(model.net_D)
    contract = {
        "protocol": "real_8patch_optimizer_smoke_v1",
        "seed": args.seed,
        "patches_per_volume": 8,
        "qumod_enabled": True,
    }
    contract_sha256 = canonical_sha256(contract)
    training_state = {
        "rng_state": capture_rng_state(),
        "loader_generator_state": loader_generator.get_state(),
        "sampler_epoch": 0,
        "training_contract_sha256": contract_sha256,
    }
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="optimizer_smoke_", dir=output_path.parent) as temp_dir:
        checkpoint_path = Path(temp_dir) / "one_step_checkpoint.pt"
        model.save(checkpoint_path, epoch=0, total_iter=1, training_state=training_state)
        checkpoint_size_mib = checkpoint_path.stat().st_size / 1048576

        restore_rng_state(training_state["rng_state"], strict_cuda=True)
        expected_cpu_random = torch.rand(4)
        expected_cuda_random = torch.rand(4, device=device).cpu() if device.type == "cuda" else None

        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        resumed = modelGAN(opts)
        resumed.set_scheduler(opts)
        checkpoint = resumed.resume(checkpoint_path, strict_training_state=True)
        resumed_generator_digest = module_digest(resumed.net_G)
        resumed_discriminator_digest = module_digest(resumed.net_D)
        restore_rng_state(checkpoint["training_state"]["rng_state"], strict_cuda=True)
        observed_cpu_random = torch.rand(4)
        observed_cuda_random = torch.rand(4, device=device).cpu() if device.type == "cuda" else None

        parameters_restored = (
            generator_digest == resumed_generator_digest
            and discriminator_digest == resumed_discriminator_digest
        )
        rng_restored = torch.equal(expected_cpu_random, observed_cpu_random)
        if device.type == "cuda":
            rng_restored = rng_restored and torch.equal(expected_cuda_random, observed_cuda_random)
        optimizer_restored = bool(resumed.optimizer_G.state) and bool(resumed.optimizer_D.state)
        scheduler_restored = (
            checkpoint["schedulers"]
            == [scheduler.state_dict() for scheduler in resumed.schedulers]
        )
        if not all((parameters_restored, rng_restored, optimizer_restored, scheduler_restored)):
            raise AssertionError(
                "Checkpoint round-trip failed: "
                f"parameters={parameters_restored}, rng={rng_restored}, "
                f"optimizer={optimizer_restored}, scheduler={scheduler_restored}"
            )

    report = {
        "status": "passed",
        "protocol": contract["protocol"],
        "device": str(device),
        "patient_id": list(batch["patient_id"]),
        "count_label": list(batch["count_label"]),
        "row_index": batch["row_index"].tolist(),
        "sample_seed": batch["sample_seed"].tolist(),
        "input_shape_before_flatten": list(batch["vol_low"].shape),
        "model_input_shape": model_input_shape,
        "losses": losses,
        "learning_rate_after_scheduler_step": learning_rate,
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "checkpoint_size_mib": checkpoint_size_mib,
        "checkpoint_version": int(checkpoint["checkpoint_version"]),
        "completed_epochs": int(checkpoint["completed_epochs"]),
        "total_iter": int(checkpoint["total_iter"]),
        "parameters_restored": parameters_restored,
        "optimizer_restored": optimizer_restored,
        "scheduler_restored": scheduler_restored,
        "rng_restored": rng_restored,
        "temporary_checkpoint_removed": True,
    }
    atomic_json(output_path, report)
    print("OPTIMIZER_SMOKE=" + json.dumps(report, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
