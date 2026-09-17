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
    parser.add_argument(
        "--val-csv",
        default=None,
        help="Required with --run-training; validation never reads a test manifest",
    )
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
    parser.add_argument("--val-every-epochs", type=int, default=1)
    parser.add_argument("--val-max-pairs", type=int, default=None)
    parser.add_argument("--val-patches-per-volume", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=2)
    parser.add_argument("--val-bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--max-epochs-this-invocation",
        type=int,
        default=None,
        help=(
            "Bound one process invocation without changing the declared n_epochs "
            "or strict-resume contract"
        ),
    )
    parser.add_argument("--resume", default=None)
    parser.add_argument(
        "--allow-inexact-resume", action="store_true",
        help="Allow legacy/incompatible checkpoints; disabled by default",
    )
    parser.add_argument("--pre-weight", default=None)
    parser.add_argument(
        "--run-role",
        choices=("engineering_diagnostic", "qumod_baseline", "formal"),
        default="engineering_diagnostic",
    )
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


def build_training_contract(
    opts,
    train_csv_sha256,
    effective_epoch_samples,
    val_csv_sha256=None,
    effective_val_pairs=None,
):
    return {
        "protocol": "udpet_training_v6_exact_resume_20260917",
        "seed": opts.seed,
        "run_role": opts.run_role,
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
        "numerical_determinism": {
            "torch_deterministic_algorithms": "strict",
            "cublas_workspace_config": ":4096:8",
            "cuda_matmul_tf32": False,
            "cudnn_tf32": False,
            "sample_randomness": "explicit_sampler_request_seed",
            "checkpoint_loader_generator": "canonical_seed_plus_completed_epoch",
        },
        "validation": {
            "val_csv_sha256": val_csv_sha256,
            "fixed_pair_requests": effective_val_pairs,
            "patches_per_volume": opts.val_patches_per_volume,
            "every_epochs": opts.val_every_epochs,
            "body_mask": f"NORMAL_SUV_gt_{opts.valid_value_threshold}",
            "hotspot": "top_1_percent_NORMAL_body_voxels_per_sampled_patch",
            "psnr_data_range_suv": 30.0,
            "aggregation": "patch_voxel_sums_to_patient_DRF_then_patient_bootstrap",
            "bootstrap_replicates": opts.val_bootstrap_replicates,
            "bootstrap_seed": opts.seed + 100000,
        },
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
    if opts.val_every_epochs <= 0:
        raise ValueError("--val-every-epochs must be positive")
    if opts.val_num_workers < 0:
        raise ValueError("--val-num-workers must be non-negative")
    if opts.val_patches_per_volume <= 0:
        raise ValueError("--val-patches-per-volume must be positive")
    if opts.val_bootstrap_replicates <= 0:
        raise ValueError("--val-bootstrap-replicates must be positive")
    if opts.val_max_pairs is not None and opts.val_max_pairs <= 0:
        raise ValueError("--val-max-pairs must be positive")
    if opts.max_epochs_this_invocation is not None and opts.max_epochs_this_invocation <= 0:
        raise ValueError("--max-epochs-this-invocation must be positive")
    if opts.epoch_samples is not None and opts.epoch_samples <= 0:
        raise ValueError("--epoch-samples must be positive")
    if opts.resume is not None and opts.pre_weight is not None:
        raise ValueError("--resume and --pre-weight cannot be used together")
    if opts.run_training and opts.val_csv is None:
        raise ValueError("--val-csv is required with --run-training")
    if opts.run_training and opts.lr_policy == "ReduceLROnPlateau" and opts.val_every_epochs != 1:
        raise ValueError("ReduceLROnPlateau requires --val-every-epochs 1")
    if opts.gpu_ids is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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
    from validation import evaluate_model, write_patient_metrics_csv

    random.seed(opts.seed)
    np.random.seed(opts.seed)
    torch.manual_seed(opts.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opts.seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        cudnn.allow_tf32 = False

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

    val_dataset = None
    val_sampler = None
    val_loader = None
    val_generator = None
    if opts.val_csv is not None:
        val_dataset = CSVPairDataset(
            csv_path=opts.val_csv,
            split="val",
            centers=opts.centers,
            count_levels=opts.count_levels,
            patch_size=opts.patch_size,
            stride_size=opts.stride_size,
            patches_per_volume=opts.val_patches_per_volume,
            valid_value_threshold=opts.valid_value_threshold,
            min_valid_fraction=opts.min_valid_fraction,
            augmentation=False,
            rotate_degrees=0,
            enable_lemod=False,
            reference_cache_size=opts.reference_cache_size,
            seed=opts.seed + 1,
            validate_paths=opts.validate_paths,
        )
        val_sampler = make_epoch_shuffle_sampler(
            val_dataset, opts.seed + 1, opts.val_max_pairs
        )
        val_dataset.set_epoch(0)
        val_sampler.set_epoch(0)
        val_generator = torch.Generator().manual_seed(opts.seed + 1)
        val_loader_options = dict(
            dataset=val_dataset,
            batch_size=1,
            shuffle=False,
            sampler=val_sampler,
            num_workers=opts.val_num_workers,
            pin_memory=opts.device.type == "cuda",
            worker_init_fn=seed_worker,
            generator=val_generator,
        )
        if opts.val_num_workers > 0:
            val_loader_options["persistent_workers"] = opts.persistent_workers
            val_loader_options["prefetch_factor"] = opts.prefetch_factor
        val_loader = DataLoader(**val_loader_options)

    output_directory = Path(opts.output_path) / opts.experiment_name
    output_directory.mkdir(parents=True, exist_ok=True)
    train_csv_sha256 = sha256(opts.train_csv)
    val_csv_sha256 = sha256(opts.val_csv) if opts.val_csv is not None else None
    effective_val_pairs = len(val_sampler) if val_sampler is not None else None
    training_contract = build_training_contract(
        opts,
        train_csv_sha256,
        effective_epoch_samples,
        val_csv_sha256,
        effective_val_pairs,
    )
    training_contract_sha256 = canonical_sha256(training_contract)
    config = {
        "protocol": "udpet_training_v6_exact_resume_20260917",
        "seed": opts.seed,
        "train_csv": str(Path(opts.train_csv).resolve()),
        "train_csv_sha256": train_csv_sha256,
        "dataset": dataset.summary(),
        "validation": None if val_dataset is None else {
            "val_csv": str(Path(opts.val_csv).resolve()),
            "val_csv_sha256": val_csv_sha256,
            "dataset": val_dataset.summary(),
            "fixed_pair_requests": effective_val_pairs,
            "patches_per_volume": opts.val_patches_per_volume,
            "num_workers": opts.val_num_workers,
            "every_epochs": opts.val_every_epochs,
            "bootstrap_replicates": opts.val_bootstrap_replicates,
            "bootstrap_seed": opts.seed + 100000,
            "body_mask": f"NORMAL SUV > {opts.valid_value_threshold}",
            "hotspot_definition": "top 1% NORMAL body voxels per sampled patch",
            "aggregation": "patient x DRF before patient bootstrap",
        },
        "filters": {"centers": opts.centers, "count_levels": opts.count_levels},
        "sampling_mode": opts.sampling_mode,
        "epoch_samples": effective_epoch_samples,
        "loader": {
            "num_workers": opts.num_workers,
            "persistent_workers": bool(opts.persistent_workers and opts.num_workers > 0),
            "prefetch_factor": opts.prefetch_factor if opts.num_workers > 0 else None,
            "reference_cache_size_per_worker": opts.reference_cache_size,
            "sample_randomness": "explicit sampler request seed",
            "checkpoint_generator_state": "canonical seed + completed epoch",
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
        "run_role": opts.run_role,
        "optimizer_training_authorized": opts.run_training,
        "formal_training_authorized": bool(
            opts.run_training and opts.run_role == "formal"
        ),
        "epoch_semantics": "exactly n_epochs; Python stop is exclusive",
        "n_epochs": opts.n_epochs,
        "max_epochs_this_invocation": opts.max_epochs_this_invocation,
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
            if training_state.get("loader_generator_state_semantics") != (
                "canonical_seed_plus_completed_epoch"
            ):
                raise ValueError(
                    "Strict resume requires the canonical loader-generator state contract"
                )
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
    validation_directory = output_directory / "validation"
    validation_directory.mkdir(parents=True, exist_ok=True)
    loss_path = output_directory / "train_loss.csv"
    validation_loss_fields = [
        "val_body_rmse_suv",
        "val_body_suvmean_abs_bias_percent",
        "val_hotspot_suvmean_abs_bias_percent",
    ]
    loss_header = [
        "completed_epoch", "total_iter", "learning_rate",
        *model.loss_names, *validation_loss_fields,
    ]
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

    invocation_end_epoch = opts.n_epochs
    if opts.max_epochs_this_invocation is not None:
        invocation_end_epoch = min(
            opts.n_epochs, start_epoch + opts.max_epochs_this_invocation
        )

    smoke_path = output_directory / "loader_smoke_test.json"
    smoke_recorded = smoke_path.is_file()
    for epoch in range(start_epoch, invocation_end_epoch):
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
        completed_epoch = epoch + 1
        val_values = {field: "" for field in validation_loss_fields}
        validation_metric = means["loss_recon"]
        should_validate = completed_epoch % opts.val_every_epochs == 0
        if should_validate:
            val_dataset.set_epoch(0)
            val_sampler.set_epoch(0)
            patient_rows, validation_summary = evaluate_model(
                model,
                val_loader,
                body_threshold=opts.valid_value_threshold,
                bootstrap_replicates=opts.val_bootstrap_replicates,
                bootstrap_seed=opts.seed + 100000,
            )
            validation_summary.update({
                "protocol": "udpet_patient_quant_validation_v1_20260917",
                "completed_epoch": completed_epoch,
                "total_iter": total_iter,
                "fixed_pair_requests": len(val_sampler),
                "patches_per_volume": opts.val_patches_per_volume,
                "body_mask": f"NORMAL SUV > {opts.valid_value_threshold}",
                "hotspot_definition": "top 1% NORMAL body voxels per sampled patch",
                "checkpoint_selection_use": "validation_only",
                "test_manifest_read": False,
            })
            metric_block = validation_summary["overall_patient_balanced"]["metrics"]
            validation_metric = metric_block["body_rmse_suv"]["mean"]
            val_values = {
                "val_body_rmse_suv": validation_metric,
                "val_body_suvmean_abs_bias_percent": (
                    metric_block["body_suvmean_abs_bias_percent"]["mean"]
                ),
                "val_hotspot_suvmean_abs_bias_percent": (
                    metric_block["hotspot_suvmean_abs_bias_percent"]["mean"]
                ),
            }
            write_patient_metrics_csv(
                validation_directory / f"epoch_{completed_epoch:04d}_patient_metrics.csv",
                patient_rows,
            )
            atomic_json(
                validation_directory / f"epoch_{completed_epoch:04d}_summary.json",
                validation_summary,
            )
            print(
                "VALIDATION=" + json.dumps({
                    "completed_epoch": completed_epoch,
                    **val_values,
                    "patients": validation_summary["unique_patients"],
                }, ensure_ascii=False),
                flush=True,
            )
        current_lr = model.update_learning_rate(validation_metric)
        with loss_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([
                completed_epoch, total_iter, current_lr,
                *[means[name] for name in model.loss_names],
                *[val_values[name] for name in validation_loss_fields],
            ])
        should_stop = current_lr < 1e-7
        should_save = (
            completed_epoch % opts.saveModel_epochs == 0
            or completed_epoch == opts.n_epochs
            or completed_epoch == invocation_end_epoch
            or should_stop
        )
        if should_save:
            canonical_loader_generator_state = (
                torch.Generator()
                .manual_seed(opts.seed + completed_epoch)
                .get_state()
            )
            training_state = {
                "rng_state": capture_rng_state(),
                "loader_generator_state": canonical_loader_generator_state,
                "loader_generator_state_semantics": (
                    "canonical_seed_plus_completed_epoch"
                ),
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

    if invocation_end_epoch < opts.n_epochs:
        print(
            "INVOCATION_LIMIT_REACHED: "
            f"completed_epoch={invocation_end_epoch}; declared_n_epochs={opts.n_epochs}; "
            "resume from the saved epoch-boundary checkpoint to continue",
            flush=True,
        )


if __name__ == "__main__":
    main()
