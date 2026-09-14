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
    parser.add_argument("--sampling-mode", choices=("csv_weighted", "uniform"), default="csv_weighted")
    parser.add_argument("--epoch-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--gpu-ids", default=None, help="Optional CUDA_VISIBLE_DEVICES value; no GPU is hard-coded")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--num-workers", type=int, default=2)
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


def main():
    opts = parse_args()
    if opts.gpu_ids is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = opts.gpu_ids

    import torch
    import torch.backends.cudnn as cudnn
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from CSVPairDataset import CSVPairDataset, make_weighted_sampler, seed_worker
    from model_LeqModGan import modelGAN

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
        seed=opts.seed,
        validate_paths=opts.validate_paths,
    )
    sampler = None
    shuffle = opts.sampling_mode == "uniform"
    if opts.sampling_mode == "csv_weighted":
        sampler = make_weighted_sampler(dataset, opts.seed, opts.epoch_samples)
        shuffle = False
    generator = torch.Generator().manual_seed(opts.seed)
    loader = DataLoader(
        dataset,
        batch_size=opts.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=opts.num_workers,
        pin_memory=opts.device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=False,
    )

    output_directory = Path(opts.output_path) / opts.experiment_name
    output_directory.mkdir(parents=True, exist_ok=True)
    config = {
        "protocol": "udpET_csv_loader_v1_20260912",
        "seed": opts.seed,
        "train_csv": str(Path(opts.train_csv).resolve()),
        "train_csv_sha256": sha256(opts.train_csv),
        "dataset": dataset.summary(),
        "filters": {"centers": opts.centers, "count_levels": opts.count_levels},
        "sampling_mode": opts.sampling_mode,
        "epoch_samples": opts.epoch_samples or len(dataset),
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
    }
    atomic_json(output_directory / "configuration.json", config)
    print(json.dumps(config, ensure_ascii=False, indent=2), flush=True)

    first_batch = next(iter(loader))
    smoke = {
        "vol_low_shape": list(first_batch["vol_low"].shape),
        "vol_high_shape": list(first_batch["vol_high"].shape),
        "vol_weight_shape": list(first_batch["vol_weight"].shape),
        "patient_id": list(first_batch["patient_id"]),
        "cohort": list(first_batch["cohort"]),
        "count_label": list(first_batch["count_label"]),
        "finite_low": bool(torch.isfinite(first_batch["vol_low"]).all()),
        "finite_high": bool(torch.isfinite(first_batch["vol_high"]).all()),
        "low_min": float(first_batch["vol_low"].min()),
        "low_max": float(first_batch["vol_low"].max()),
        "high_min": float(first_batch["vol_high"].min()),
        "high_max": float(first_batch["vol_high"].max()),
    }
    atomic_json(output_directory / "loader_smoke_test.json", smoke)
    print("LOADER_SMOKE_TEST=" + json.dumps(smoke, ensure_ascii=False), flush=True)
    if not opts.run_training:
        print("DRY_RUN_COMPLETE: loader verified; training was not started", flush=True)
        return

    model = modelGAN(opts)
    if opts.resume is None:
        model.initialize()
        start_epoch, total_iter = 0, 0
    else:
        resumed_epoch, total_iter = model.resume(opts.resume)
        start_epoch = resumed_epoch + 1
    model.set_scheduler(opts, start_epoch - 1)
    checkpoints = output_directory / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    loss_path = output_directory / "train_loss.csv"
    with loss_path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerow(["epoch", *model.loss_names])

    for epoch in range(start_epoch, opts.n_epochs + 1):
        dataset.set_epoch(epoch)
        epoch_losses = []
        model.train()
        model.set_epoch(epoch)
        train_bar = tqdm(loader)
        for data in train_bar:
            total_iter += 1
            model.set_input(data)
            model.optimize(total_iter)
            current = model.get_current_losses()
            epoch_losses.append(current)
            train_bar.set_description(f"[Epoch {epoch}] {model.loss_summary()}")
        means = {name: float(np.mean([row[name] for row in epoch_losses])) for name in model.loss_names}
        current_lr = model.update_learning_rate(means["loss_recon"])
        with loss_path.open("a", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow([epoch, *[means[name] for name in model.loss_names]])
        if (epoch + 1) % opts.saveModel_epochs == 0:
            model.save(str(checkpoints / f"model_{epoch}.pt"), epoch, total_iter)
        if current_lr < 1e-7:
            print("Terminating training: learning rate below threshold", flush=True)
            break


if __name__ == "__main__":
    main()
