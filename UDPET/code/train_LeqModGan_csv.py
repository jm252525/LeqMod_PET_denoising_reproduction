"""CSV-driven, patient-safe UDPET training entry point.

The default action is a one-batch loader smoke test. Formal training only starts
when --run-training is explicitly supplied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="LeqMod UDPET CSV training")
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--experiment-name", default="leqmod_csv")
    parser.add_argument("--centers", nargs="*", default=[])
    parser.add_argument("--count-levels", nargs="*", default=[])
    parser.add_argument(
        "--sampling-mode",
        choices=("volume_grouped_weighted", "csv_weighted", "uniform"),
        default="volume_grouped_weighted",
    )
    parser.add_argument("--epoch-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--gpu-ids", default=None, help="Optional CUDA_VISIBLE_DEVICES value; no GPU is hard-coded")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--reference-cache-size", type=int, default=1,
        help="Number of cropped NORMAL references cached in each DataLoader worker",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--patches-per-volume", type=int, default=8)
    parser.add_argument("--patch-size", nargs=3, type=int, default=[80, 80, 80])
    parser.add_argument("--stride-size", nargs=3, type=int, default=[20, 20, 20])
    parser.add_argument("--valid-value-threshold", type=float, default=0.2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.2)
    parser.add_argument("--augmentation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--rotate-train", type=int, default=10)
    parser.add_argument("--validate-paths", action="store_true")

    parser.add_argument("--enable-lemod", action="store_true")
    parser.add_argument("--adapt-lesion-sampling", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lemod-weight", type=float, default=2.0)
    parser.add_argument("--seg-prob-threshold", type=float, default=0.2)
    parser.add_argument("--disable-qumod", action="store_true")
    parser.add_argument("--qumod-weight", type=float, default=2.0)
    parser.add_argument("--mse-weight", type=float, default=10.0)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--n-epochs", type=int, default=500)
    parser.add_argument("--lr-policy", choices=("ReduceLROnPlateau", "multistep", "cosine"), default="ReduceLROnPlateau")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--plateau-step-size", type=int, default=10)
    parser.add_argument("--multi-step-size", nargs="+", type=int, default=[100, 200, 300, 400])
    parser.add_argument("--save-model-epochs", type=int, default=1)
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--allow-inexact-resume", action="store_true",
        help="Allow legacy/incompatible checkpoints; disabled by default",
    )
    parser.add_argument("--pre-weight", default=None)
    parser.add_argument("--run-training", action="store_true", help="Required to start optimizer/training iterations")
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
    os.replace(temporary, path)


def summarize_batch(batch, torch):
    return {
        "vol_low_shape": list(batch["vol_low"].shape),
        "vol_high_shape": list(batch["vol_high"].shape),
        "vol_weight_shape": list(batch["vol_weight"].shape),
        "patient_id": list(batch["patient_id"]),
        "cohort": list(batch["cohort"]),
        "count_label": list(batch["count_label"]),
        "row_index": batch["row_index"].tolist(),
        "sample_seed": batch["sample_seed"].tolist(),
        "finite_low": bool(torch.isfinite(batch["vol_low"]).all()),
        "finite_high": bool(torch.isfinite(batch["vol_high"]).all()),
        "low_min": float(batch["vol_low"].min()),
        "low_max": float(batch["vol_low"].max()),
        "high_min": float(batch["vol_high"].min()),
        "high_max": float(batch["vol_high"].max()),
    }


def build_training_contract(opts, train_csv_sha256, effective_epoch_samples):
    return {
        "protocol": "udpet_training_v3_20260917",
        "seed": opts.seed,
        "train_csv_sha256": train_csv_sha256,
        "centers": list(opts.centers),
        "count_levels": list(opts.count_levels),
        "sampling_mode": opts.sampling_mode,
        "epoch_samples": effective_epoch_samples,
        "batch_size": opts.batch_size,
        "patches_per_volume": opts.patches_per_volume,
        "patch_size": list(opts.patch_size),
        "stride_size": list(opts.stride_size),
        "valid_value_threshold": opts.valid_value_threshold,
        "min_valid_fraction": opts.min_valid_fraction,
        "augmentation": opts.augmentation,
        "rotate_train": opts.rotate_train,
        "enable_lemod": opts.enable_lemod,
        "adapt_lesion_sampling": opts.adapt_lesion_sampling,
        "lemod_weight": opts.lemod_weight,
        "seg_prob_threshold": opts.seg_prob_threshold,
        "qumod_enabled": not opts.disable_qumod,
        "qumod_weight": opts.qumod_weight,
        "mse_weight": opts.mse_weight,
        "lr": opts.lr,
        "n_epochs": opts.n_epochs,
        "lr_policy": opts.lr_policy,
        "gamma": opts.gamma,
        "plateau_step_size": opts.plateau_step_size,
        "multi_step_size": list(opts.multi_step_size),
        "device_type": opts.device.type,
        "visible_gpu_count": opts.numGPUs,
    }


def main():
    opts = parse_args()
    if opts.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if opts.prefetch_factor < 1:
        raise ValueError("--prefetch-factor must be positive")
    if opts.reference_cache_size < 0:
        raise ValueError("--reference-cache-size must be non-negative")
    if opts.n_epochs <= 0:
        raise ValueError("--n-epochs must be positive")
    if opts.save_model_epochs <= 0:
        raise ValueError("--save-model-epochs must be positive")
    if opts.epoch_samples is not None and opts.epoch_samples <= 0:
        raise ValueError("--epoch-samples must be positive")
    if opts.resume is not None and opts.pre_weight is not None:
        raise ValueError("--resume and --pre-weight cannot be used together")
    if opts.gpu_ids is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from CSVPairDataset import (
        CSVPairDataset,
        make_epoch_shuffle_sampler,
        make_volume_grouped_weighted_sampler,
        make_weighted_sampler,
        seed_worker,
    )
    from model_LeqModGan import modelGAN
    from training_state import (
        canonical_sha256,
        capture_rng_state,
        restore_rng_state,
        validate_checkpoint_contract,
    )

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True

    if opts.device == "cpu":
        opts.device = torch.device("cpu")
    elif opts.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        opts.device = torch.device("cuda:0")
    else:
        opts.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    opts.numGPUs = torch.cuda.device_count() if opts.device.type == "cuda" else 0

    opts.weightLoss_mse = opts.mse_weight
    opts.weightLoss_localSUVbias = None if opts.disable_qumod else opts.qumod_weight
    opts.weightLoss_lesionSUVbias = opts.lemod_weight if opts.enable_lemod else None
    opts.adaptSampleByLesion = bool(opts.enable_lemod and opts.adapt_lesion_sampling)
    opts.segProbThresh = opts.seg_prob_threshold
    opts.preWeight = opts.pre_weight
    opts.SamplePatchNumPerImage = opts.patches_per_volume
    opts.validValueThresh = opts.valid_value_threshold
    opts.AUG = opts.augmentation
    opts.rotate_train = opts.rotate_train
    opts.batch_size = opts.batch_size
    opts.patch_size = opts.patch_size
    opts.stride_size = opts.stride_size
    opts.n_epochs = opts.n_epochs
    opts.Plateau_step_size = opts.plateau_step_size
    opts.multi_step_size = opts.multi_step_size
    opts.saveModel_epochs = opts.save_model_epochs

    dataset = CSVPairDataset(
        csv_path=opts.train_csv,
        split="train",
        centers=opts.centers,
        count_levels=opts.count_levels,
        patch_size=opts.patch_size,
        stride_size=opts.stride_size,
        patches_per_volume=opts.patches_per_volume,
        valid_value_threshold=opts.valid_value_threshold,
        min_valid_fraction=opts.min_valid_fraction,
        augmentation=opts.augmentation,
        rotate_degrees=opts.rotate_train,
        enable_lemod=opts.enable_lemod,
        reference_cache_size=opts.reference_cache_size,
        seed=opts.seed,
        validate_paths=opts.validate_paths,
    )
    sampler = None
    if opts.sampling_mode == "csv_weighted":
        sampler = make_weighted_sampler(dataset, opts.seed, opts.epoch_samples)
    elif opts.sampling_mode == "volume_grouped_weighted":
        sampler = make_volume_grouped_weighted_sampler(dataset, opts.seed, opts.epoch_samples)
    elif opts.sampling_mode == "uniform":
        sampler = make_epoch_shuffle_sampler(dataset, opts.seed, opts.epoch_samples)
    effective_epoch_samples = len(sampler)
    generator = torch.Generator().manual_seed(opts.seed)
    loader_options = dict(
        dataset=dataset,
        batch_size=opts.batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=opts.num_workers,
        pin_memory=opts.device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    if opts.num_workers > 0:
        loader_options["persistent_workers"] = opts.persistent_workers
        loader_options["prefetch_factor"] = opts.prefetch_factor
    loader = DataLoader(**loader_options)

    output_directory = Path(opts.output_path) / opts.experiment_name
    output_directory.mkdir(parents=True, exist_ok=True)
    train_csv_sha256 = sha256(opts.train_csv)
    training_contract = build_training_contract(
        opts, train_csv_sha256, effective_epoch_samples
    )
    training_contract_sha256 = canonical_sha256(training_contract)
    config = {
        "protocol": "udpet_training_v3_20260917",
        "seed": opts.seed,
        "train_csv": str(Path(opts.train_csv).resolve()),
        "train_csv_sha256": train_csv_sha256,
        "dataset": dataset.summary(),
        "filters": {"centers": opts.centers, "count_levels": opts.count_levels},
        "sampling_mode": opts.sampling_mode,
        "epoch_samples": effective_epoch_samples,
        "loader": {
            "num_workers": opts.num_workers,
            "persistent_workers": bool(opts.persistent_workers and opts.num_workers > 0),
            "prefetch_factor": opts.prefetch_factor if opts.num_workers > 0 else None,
            "reference_cache_size_per_worker": opts.reference_cache_size,
        },
        "batch_size": opts.batch_size,
        "patches_per_volume": opts.patches_per_volume,
        "patch_size": opts.patch_size,
        "stride_size": opts.stride_size,
        "valid_value_threshold": opts.valid_value_threshold,
        "min_valid_fraction": opts.min_valid_fraction,
        "augmentation": opts.augmentation,
        "device": str(opts.device),
        "visible_gpu_count": opts.numGPUs,
        "lemod_enabled": opts.enable_lemod,
        "qumod_enabled": not opts.disable_qumod,
        "formal_training_authorized": opts.run_training,
        "epoch_semantics": "exactly n_epochs; Python stop is exclusive",
        "n_epochs": opts.n_epochs,
        "iterations_per_epoch": len(loader),
        "planned_training_iterations": opts.n_epochs * len(loader),
        "optimizer_steps_per_iteration": {"generator": 1, "discriminator": 1},
        "resume_checkpoint": opts.resume,
        "strict_resume": not opts.allow_inexact_resume,
        "training_contract": training_contract,
        "training_contract_sha256": training_contract_sha256,
    }
    atomic_json(output_directory / "configuration.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    if not opts.run_training:
        sampler.set_epoch(0)
        dataset.set_epoch(0)
        first_batch = next(iter(loader))
        smoke = summarize_batch(first_batch, torch)
        atomic_json(output_directory / "loader_smoke_test.json", smoke)
        print("LOADER_SMOKE_TEST=" + json.dumps(smoke, ensure_ascii=False), flush=True)
        print("DRY_RUN_COMPLETE: loader verified; training was not started", flush=True)
        return

    model = modelGAN(opts)
    model.set_scheduler(opts)
    if opts.resume is None:
        model.initialize()
        start_epoch, total_iter = 0, 0
    else:
        checkpoint = model.resume(
            opts.resume,
            strict_training_state=not opts.allow_inexact_resume,
        )
        resume_problems = validate_checkpoint_contract(
            checkpoint,
            training_contract_sha256,
            allow_inexact=opts.allow_inexact_resume,
        )
        if resume_problems:
            print("WARNING_INEXACT_RESUME=" + json.dumps(resume_problems), flush=True)
        start_epoch = int(checkpoint.get("completed_epochs", checkpoint["epoch"] + 1))
        total_iter = int(checkpoint["total_iter"])
        training_state = checkpoint.get("training_state") or {}
        if "loader_generator_state" in training_state:
            generator.set_state(training_state["loader_generator_state"].cpu())
        if "rng_state" in training_state:
            restore_rng_state(
                training_state["rng_state"],
                strict_cuda=not opts.allow_inexact_resume,
            )
        if not opts.allow_inexact_resume:
            expected_sampler_epoch = start_epoch - 1
            if int(training_state["sampler_epoch"]) != expected_sampler_epoch:
                raise ValueError(
                    "Strict resume sampler epoch mismatch: "
                    f"checkpoint={training_state['sampler_epoch']}, "
                    f"expected={expected_sampler_epoch}"
                )
            expected_total_iter = start_epoch * len(loader)
            if total_iter != expected_total_iter:
                raise ValueError(
                    "Strict resume requires an epoch-boundary checkpoint: "
                    f"total_iter={total_iter}, expected={expected_total_iter}"
                )
        if start_epoch > opts.n_epochs:
            raise ValueError(
                f"Checkpoint completed {start_epoch} epochs, exceeding n_epochs={opts.n_epochs}"
            )
    checkpoints = output_directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    loss_path = output_directory / "train_loss.csv"
    loss_header = ["completed_epoch", "total_iter", "learning_rate", *model.loss_names]
    if start_epoch == 0:
        with loss_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(loss_header)
    elif not loss_path.is_file():
        if not opts.allow_inexact_resume:
            raise FileNotFoundError(
                f"Strict resume requires the existing loss history: {loss_path}"
            )
        with loss_path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(loss_header)
    else:
        with loss_path.open("r", encoding="utf-8", newline="") as handle:
            loss_rows = list(csv.reader(handle))
        observed_header = loss_rows[0] if loss_rows else None
        if observed_header != loss_header:
            raise ValueError(
                f"Loss history header mismatch: expected {loss_header}, got {observed_header}"
            )
        if not opts.allow_inexact_resume:
            if len(loss_rows) != start_epoch + 1:
                raise ValueError(
                    "Strict resume loss-history length mismatch: "
                    f"rows={len(loss_rows) - 1}, completed_epochs={start_epoch}"
                )
            if int(loss_rows[-1][0]) != start_epoch or int(loss_rows[-1][1]) != total_iter:
                raise ValueError(
                    "Strict resume loss-history endpoint does not match checkpoint"
                )

    if start_epoch == opts.n_epochs:
        print("TRAINING_ALREADY_COMPLETE: checkpoint reached n_epochs", flush=True)
        return

    smoke_path = output_directory / "loader_smoke_test.json"
    smoke_recorded = smoke_path.is_file()
    for epoch in range(start_epoch, opts.n_epochs):
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        epoch_losses = []
        model.train()
        model.set_epoch(epoch)
        train_bar = tqdm(loader)
        for data in train_bar:
            if not smoke_recorded:
                smoke = summarize_batch(data, torch)
                atomic_json(smoke_path, smoke)
                print("LOADER_SMOKE_TEST=" + json.dumps(smoke, ensure_ascii=False), flush=True)
                smoke_recorded = True
            total_iter += 1
            model.set_input(data)
            model.optimize(total_iter)
            current = model.get_current_losses()
            if not all(np.isfinite(value) for value in current.values()):
                raise FloatingPointError(
                    f"Non-finite loss at epoch={epoch + 1}, total_iter={total_iter}: {current}"
                )
            epoch_losses.append(current)
            train_bar.set_description(
                f"[Epoch {epoch + 1}/{opts.n_epochs}] {model.loss_summary()}"
            )
        means = {name: float(np.mean([row[name] for row in epoch_losses])) for name in model.loss_names}
        current_lr = model.update_learning_rate(means["loss_recon"])
        completed_epoch = epoch + 1
        with loss_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([
                completed_epoch, total_iter, current_lr,
                *[means[name] for name in model.loss_names],
            ])
        should_stop = current_lr < 1e-7
        should_save = (
            completed_epoch % opts.saveModel_epochs == 0
            or completed_epoch == opts.n_epochs
            or should_stop
        )
        if should_save:
            training_state = {
                "rng_state": capture_rng_state(),
                "loader_generator_state": generator.get_state(),
                "sampler_epoch": epoch,
                "training_contract_sha256": training_contract_sha256,
            }
            model.save(
                str(checkpoints / f"model_epoch_{completed_epoch:04d}.pt"),
                epoch,
                total_iter,
                training_state,
            )
        if should_stop:
            print("Terminating training: learning rate below threshold", flush=True)
            break


if __name__ == "__main__":
    main()
