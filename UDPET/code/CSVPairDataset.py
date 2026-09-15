"""Patient-paired UDPET loader driven by the fixed split CSV manifests."""

from __future__ import annotations

import csv
import json
import random
from collections import OrderedDict
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset, Sampler, WeightedRandomSampler, get_worker_info


REQUIRED_COLUMNS = {
    "site", "cohort", "patient_id", "study_id", "split", "unit",
    "count_label", "count_percent", "low_count_path", "full_count_path",
}
COUNT_ALIASES = {
    "D2": "D2", "50": "D2", "50%": "D2",
    "D4": "D4", "25": "D4", "25%": "D4",
    "D10": "D10", "10": "D10", "10%": "D10",
    "D20": "D20", "5": "D20", "5%": "D20",
    "D50": "D50", "2": "D50", "2%": "D50",
    "D100": "D100", "1": "D100", "1%": "D100",
}


def normalize_count_levels(values):
    if not values:
        return None
    normalized = set()
    for value in values:
        key = str(value).strip().upper()
        if key not in COUNT_ALIASES:
            raise ValueError(f"Unsupported count level: {value}")
        normalized.add(COUNT_ALIASES[key])
    return normalized


def _axis_starts(size, patch, stride):
    if size < patch:
        return []
    starts = list(range(0, size - patch + 1, stride))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


def _crop_box(reference, threshold):
    masks = (
        np.max(reference, axis=(1, 2)) > threshold,
        np.max(reference, axis=(0, 2)) > threshold,
        np.max(reference, axis=(0, 1)) > threshold,
    )
    bounds = []
    for mask, size in zip(masks, reference.shape):
        if not np.any(mask):
            raise ValueError(f"NORMAL image has no voxel above SUV {threshold}")
        bounds.extend((int(np.argmax(mask)), int(size - np.argmax(mask[::-1]))))
    return bounds


def _apply_box(array, box):
    return array[box[0]:box[1], box[2]:box[3], box[4]:box[5]]


def _integral_volume(mask):
    integral = np.pad(mask.astype(np.uint32, copy=False), ((1, 0), (1, 0), (1, 0)))
    np.cumsum(integral, axis=0, dtype=np.uint32, out=integral)
    np.cumsum(integral, axis=1, dtype=np.uint32, out=integral)
    np.cumsum(integral, axis=2, dtype=np.uint32, out=integral)
    return integral


def _box_sum(integral, x, y, z, px, py, pz):
    x1, y1, z1 = x + px, y + py, z + pz
    value = lambda a, b, c: int(integral[a, b, c])
    return (
        value(x1, y1, z1)
        - value(x, y1, z1)
        - value(x1, y, z1)
        - value(x1, y1, z)
        + value(x, y, z1)
        + value(x, y1, z)
        + value(x1, y, z)
        - value(x, y, z)
    )


def _candidate_boxes(reference, patch_size, stride_size, threshold, min_valid_fraction):
    px, py, pz = patch_size
    starts = (
        _axis_starts(reference.shape[0], px, stride_size[0]),
        _axis_starts(reference.shape[1], py, stride_size[1]),
        _axis_starts(reference.shape[2], pz, stride_size[2]),
    )
    if any(not axis for axis in starts):
        raise ValueError(f"Cropped shape {reference.shape} is smaller than patch size {patch_size}")
    integral = _integral_volume(reference > threshold)
    minimum = float(min_valid_fraction) * px * py * pz
    boxes = []
    for x in starts[0]:
        for y in starts[1]:
            for z in starts[2]:
                if _box_sum(integral, x, y, z, px, py, pz) > minimum:
                    boxes.append((x, x + px, y, y + py, z, z + pz))
    return boxes


class CSVPairDataset(Dataset):
    def __init__(
        self,
        csv_path,
        split="train",
        centers=None,
        count_levels=None,
        patch_size=(80, 80, 80),
        stride_size=(20, 20, 20),
        patches_per_volume=8,
        valid_value_threshold=0.2,
        min_valid_fraction=0.2,
        augmentation=True,
        rotate_degrees=10,
        enable_lemod=False,
        reference_cache_size=1,
        seed=20260910,
        validate_paths=False,
    ):
        self.csv_path = Path(csv_path)
        self.patch_size = tuple(map(int, patch_size))
        self.stride_size = tuple(map(int, stride_size))
        self.patches_per_volume = int(patches_per_volume)
        self.valid_value_threshold = float(valid_value_threshold)
        self.min_valid_fraction = float(min_valid_fraction)
        self.augmentation = bool(augmentation)
        self.rotate_degrees = int(rotate_degrees)
        self.enable_lemod = bool(enable_lemod)
        self.reference_cache_size = int(reference_cache_size)
        if self.reference_cache_size < 0:
            raise ValueError("reference_cache_size must be non-negative")
        self.seed = int(seed)
        self.epoch = 0
        # Dataset instances live inside DataLoader workers, so this cache and its
        # counters are deliberately process-local and require no synchronization.
        self._reference_cache = OrderedDict()
        self._io_stats = {
            "low_loads": 0,
            "full_loads": 0,
            "segmentation_loads": 0,
            "reference_cache_hits": 0,
            "reference_cache_misses": 0,
            "reference_cache_evictions": 0,
        }

        with self.csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
            if missing:
                raise ValueError(f"CSV missing required columns: {sorted(missing)}")
            rows = list(reader)

        center_filters = {str(item).strip().lower() for item in (centers or [])}
        allowed_levels = normalize_count_levels(count_levels)
        selected = []
        for row in rows:
            center = (row.get("center") or row["site"]).strip().lower()
            cohort = row["cohort"].strip().lower()
            if row["split"].strip().lower() != str(split).strip().lower():
                continue
            if center_filters and center not in center_filters and cohort not in center_filters:
                continue
            if allowed_levels and row["count_label"].strip().upper() not in allowed_levels:
                continue
            if row["unit"].strip().lower() != "suvbw":
                raise ValueError(f"Non-SUVbw row selected: {row['low_count_path']}")
            if row.get("pair_qc_status") and not row["pair_qc_status"].startswith("passed"):
                continue
            selected.append(row)

        if not selected:
            raise ValueError("No CSV rows remain after split/center/count filters")
        keys = [(r["site"], r["patient_id"], r["study_id"], r["count_label"]) for r in selected]
        if len(keys) != len(set(keys)):
            raise ValueError("Duplicate patient/study/count pair found in selected CSV rows")
        if self.enable_lemod:
            if "segmentation_path" not in rows[0]:
                raise ValueError("LeMod requires a segmentation_path column; current manifests do not contain one")
            missing_masks = [r["segmentation_path"] for r in selected if not r.get("segmentation_path")]
            if missing_masks:
                raise ValueError(f"LeMod enabled but {len(missing_masks)} segmentation paths are empty")
        if validate_paths:
            missing_paths = []
            for row in selected:
                for field in ("low_count_path", "full_count_path"):
                    if not Path(row[field]).is_file():
                        missing_paths.append(row[field])
                if self.enable_lemod and not Path(row["segmentation_path"]).is_file():
                    missing_paths.append(row["segmentation_path"])
            if missing_paths:
                sample = "\n".join(missing_paths[:5])
                raise FileNotFoundError(f"Missing {len(missing_paths)} selected files. First entries:\n{sample}")
        self.rows = selected

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def sampling_weights(self):
        return [float(row.get("leqmod_sampling_weight") or 1.0) for row in self.rows]

    def group_key(self, index):
        """Return the acquisition/reference identity used for locality-aware sampling."""
        row = self.rows[index]
        return (
            row["site"], row["patient_id"], row["study_id"], row["full_count_path"],
        )

    def io_stats(self):
        """Return I/O counters for this process-local dataset instance."""
        return dict(self._io_stats)

    def reset_io_stats(self, clear_cache=False):
        for key in self._io_stats:
            self._io_stats[key] = 0
        if clear_cache:
            self._reference_cache.clear()

    def summary(self):
        by_cohort = {}
        by_level = {}
        for row in self.rows:
            by_cohort[row["cohort"]] = by_cohort.get(row["cohort"], 0) + 1
            by_level[row["count_label"]] = by_level.get(row["count_label"], 0) + 1
        return {
            "csv_path": str(self.csv_path),
            "pairs": len(self.rows),
            "patients": len({(r["site"], r["patient_id"]) for r in self.rows}),
            "studies": len({(r["site"], r["patient_id"], r["study_id"]) for r in self.rows}),
            "by_cohort": dict(sorted(by_cohort.items())),
            "by_count_level": dict(sorted(by_level.items())),
            "enable_lemod": self.enable_lemod,
            "reference_cache_size_per_worker": self.reference_cache_size,
        }

    def _reference_key(self, row):
        segmentation_path = row.get("segmentation_path", "") if self.enable_lemod else ""
        return row["full_count_path"], segmentation_path

    def _load_reference(self, row):
        key = self._reference_key(row)
        if key in self._reference_cache:
            self._io_stats["reference_cache_hits"] += 1
            self._reference_cache.move_to_end(key)
            return self._reference_cache[key]

        self._io_stats["reference_cache_misses"] += 1
        high_img = nib.load(row["full_count_path"])
        original_shape = tuple(high_img.shape)
        affine = np.asarray(high_img.affine).copy()
        high = np.asarray(high_img.dataobj, dtype=np.float32).copy()
        self._io_stats["full_loads"] += 1
        if not np.all(np.isfinite(high)):
            raise ValueError(f"NaN/Inf detected in NORMAL image: {row['patient_id']}")
        high[high <= 0] = 0
        box = _crop_box(high, self.valid_value_threshold)
        high = _apply_box(high, box).copy()

        if self.enable_lemod:
            seg_img = nib.load(row["segmentation_path"])
            seg = np.asarray(seg_img.dataobj, dtype=np.float32)
            self._io_stats["segmentation_loads"] += 1
            if tuple(seg.shape) != original_shape:
                raise ValueError(f"Segmentation shape mismatch: {row['patient_id']}")
            seg = _apply_box(seg, box).copy()
        else:
            seg = None

        candidates = _candidate_boxes(
            high, self.patch_size, self.stride_size,
            self.valid_value_threshold, self.min_valid_fraction,
        )
        if len(candidates) < self.patches_per_volume:
            raise ValueError(
                f"Only {len(candidates)} valid patches for {row['patient_id']}; "
                f"requested {self.patches_per_volume}"
            )
        reference = {
            "original_shape": original_shape,
            "affine": affine,
            "high": high,
            "seg": seg,
            "box": box,
            "candidates": candidates,
        }
        if self.reference_cache_size > 0:
            self._reference_cache[key] = reference
            while len(self._reference_cache) > self.reference_cache_size:
                self._reference_cache.popitem(last=False)
                self._io_stats["reference_cache_evictions"] += 1
        return reference

    def __getitem__(self, index):
        row = self.rows[index]
        reference = self._load_reference(row)
        low_img = nib.load(row["low_count_path"])
        if tuple(low_img.shape) != reference["original_shape"]:
            raise ValueError(
                f"Shape mismatch: {low_img.shape} vs {reference['original_shape']}: "
                f"{row['patient_id']}"
            )
        affine_diff = float(np.max(np.abs(np.asarray(low_img.affine) - reference["affine"])))
        if affine_diff > 1e-4:
            raise ValueError(f"Affine mismatch {affine_diff}: {row['patient_id']}")
        low = np.asarray(low_img.dataobj, dtype=np.float32).copy()
        self._io_stats["low_loads"] += 1
        if not np.all(np.isfinite(low)):
            raise ValueError(f"NaN/Inf detected in low-count image: {row['patient_id']}")
        low[low <= 0] = 0
        box = reference["box"]
        low = _apply_box(low, box)
        high = reference["high"]
        seg = reference["seg"]
        candidates = reference["candidates"]
        probabilities = None
        if seg is not None:
            lesion_scores = np.asarray([
                float(np.max(seg[b[0]:b[1], b[2]:b[3], b[4]:b[5]])) for b in candidates
            ], dtype=np.float64)
            lesion_scores = np.maximum(lesion_scores, 0.2)
            probabilities = lesion_scores / lesion_scores.sum()
        chosen = np.random.choice(
            len(candidates), size=self.patches_per_volume, replace=False, p=probabilities
        )

        low_patches, high_patches, weight_patches = [], [], []
        for candidate_index in chosen:
            b = candidates[int(candidate_index)]
            low_patch = low[b[0]:b[1], b[2]:b[3], b[4]:b[5]]
            high_patch = high[b[0]:b[1], b[2]:b[3], b[4]:b[5]]
            weight_patch = (
                seg[b[0]:b[1], b[2]:b[3], b[4]:b[5]]
                if seg is not None else np.zeros(self.patch_size, dtype=np.float32)
            )
            if self.augmentation and random.random() > 0.8:
                angle = random.randint(-self.rotate_degrees, self.rotate_degrees)
                low_patch = ndimage.rotate(low_patch, angle, axes=(1, 2), reshape=False, mode="reflect")
                high_patch = ndimage.rotate(high_patch, angle, axes=(1, 2), reshape=False, mode="reflect")
                weight_patch = ndimage.rotate(weight_patch, angle, axes=(1, 2), reshape=False, mode="reflect")
            low_patches.append(np.asarray(low_patch, dtype=np.float32))
            high_patches.append(np.asarray(high_patch, dtype=np.float32))
            weight_patches.append(np.asarray(weight_patch, dtype=np.float32))

        return {
            "vol_low": torch.from_numpy(np.stack(low_patches)),
            "vol_high": torch.from_numpy(np.stack(high_patches)),
            "vol_weight": torch.from_numpy(np.stack(weight_patches)),
            "vol_path": row["full_count_path"],
            "patient_id": row["patient_id"],
            "study_id": row["study_id"],
            "center": row.get("center") or row["site"],
            "cohort": row["cohort"],
            "count_label": row["count_label"],
            "count_percent": float(row["count_percent"]),
            "crop_box": json.dumps(box),
        }


def make_weighted_sampler(dataset, seed, num_samples=None):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(dataset.sampling_weights(), dtype=torch.double),
        num_samples=int(num_samples or len(dataset)),
        replacement=True,
        generator=generator,
    )


class VolumeGroupedWeightedSampler(Sampler):
    """Weighted row sampling reordered into contiguous reference-volume groups.

    The row indices are first drawn with replacement using exactly the manifest
    weights. Reordering happens only after the draw, so count-level and cohort
    sampling probabilities are unchanged while repeated DRFs reuse the cached
    NORMAL reference and its precomputed crop/candidate boxes.
    """

    def __init__(self, dataset, seed, num_samples=None):
        self.dataset = dataset
        self.seed = int(seed)
        self.num_samples = int(num_samples or len(dataset))
        if self.num_samples <= 0:
            raise ValueError("num_samples must be positive")
        self.weights = torch.as_tensor(dataset.sampling_weights(), dtype=torch.double)
        if len(self.weights) != len(dataset):
            raise ValueError("sampling weight count does not match dataset length")
        self.epoch = 0

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        drawn = torch.multinomial(
            self.weights, self.num_samples, replacement=True, generator=generator
        ).tolist()

        groups = OrderedDict()
        for index in drawn:
            groups.setdefault(self.dataset.group_key(index), []).append(index)
        keys = list(groups)
        if len(keys) > 1:
            key_order = torch.randperm(len(keys), generator=generator).tolist()
        else:
            key_order = list(range(len(keys)))
        for key_index in key_order:
            indices = groups[keys[key_index]]
            if len(indices) > 1:
                order = torch.randperm(len(indices), generator=generator).tolist()
                indices = [indices[position] for position in order]
            yield from indices


def make_volume_grouped_weighted_sampler(dataset, seed, num_samples=None):
    return VolumeGroupedWeightedSampler(dataset, seed, num_samples)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
