#!/usr/bin/env python3
"""Leakage-controlled data interfaces for this study V3 spatial folds.

This module is separate from the frozen V2 ``dataset_common.py``. Callers must
select a fold explicitly. Fold-specific labels, role masks, and normalization
statistics are verified before use. Training patches must lie wholly inside a
training partition, and validation/test inference windows must lie wholly
inside the requested role partition.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset


PATCH = 128
STRIDE = 64
ROLE_CODES = {"train": 1, "validation": 2, "test": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def available_folds(project_root: Path | str = Path.cwd()) -> list[str]:
    root = Path(project_root).resolve()
    manifest = _read_json(root / "04_results/split_design_v3/fold_manifest_v3.json")
    return [record["fold_id"] for record in manifest["folds"]]


def load_fold(
    fold_id: str,
    *,
    project_root: Path | str = Path.cwd(),
    normalize: bool = True,
    verify_hashes: bool = True,
) -> dict:
    """Load one accepted V3 fold and optionally apply its training-only z-score.

    The returned ``train``, ``val``, and ``test`` arrays are the accepted
    eligible masks, not merely geographic role masks. ``partition`` retains
    the full geographic role codes and is used for patch/window containment.
    """

    root = Path(project_root).resolve()
    dataset_v2 = root / "01_data/interim/dataset_v2"
    dataset_v3 = root / "01_data/interim/dataset_v3"
    normalization_v3 = root / "01_data/interim/normalization_v3"
    manifest_path = root / "04_results/split_design_v3/fold_manifest_v3.json"
    manifest = _read_json(manifest_path)
    fold_lookup = {record["fold_id"]: record for record in manifest["folds"]}
    if fold_id not in fold_lookup:
        raise ValueError(f"Unknown fold {fold_id!r}; choose from {sorted(fold_lookup)}")
    fold_record = fold_lookup[fold_id]

    stage2_acceptance = _read_json(dataset_v3 / "stage2_acceptance_v3.json")
    stage3_acceptance = _read_json(normalization_v3 / "stage3_acceptance_v3.json")
    if not stage2_acceptance.get("overall_accepted") or not all(
        item["passed"] for item in stage2_acceptance["checks"]
    ):
        raise ValueError("Stage-2 fold dataset is not fully accepted")
    if not stage3_acceptance.get("overall_accepted") or not all(
        item["passed"] for item in stage3_acceptance["checks"]
    ):
        raise ValueError("Stage-3 normalization is not fully accepted")

    fold_dir = dataset_v3 / fold_id
    stats_path = normalization_v3 / fold_id / "norm_stats.json"
    paths = {
        "X": dataset_v2 / "X.npy",
        "valid": dataset_v2 / "valid.npy",
        "meta": dataset_v2 / "meta.json",
        "y": fold_dir / "y.npy",
        "partition": fold_dir / "partition.npy",
        "train": fold_dir / "train.npy",
        "validation": fold_dir / "val.npy",
        "test": fold_dir / "test.npy",
        "normalization": stats_path,
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing V3 {name}: {path}")

    stats = _read_json(stats_path)
    meta = _read_json(paths["meta"])
    if stats["fold_id"] != fold_id:
        raise ValueError(f"Normalization fold mismatch: {stats['fold_id']} != {fold_id}")
    if verify_hashes:
        for name, expected in (
            ("X", stats["input_sha256"]["X"]),
            ("train", stats["input_sha256"]["train_mask"]),
            ("y", stats["input_sha256"]["fold_label"]),
            ("partition", stats["input_sha256"]["partition"]),
        ):
            actual = sha256(paths[name])
            if actual != expected:
                raise ValueError(f"{fold_id} {name} SHA-256 mismatch: {actual} != {expected}")

    X_source = np.load(paths["X"])
    valid = np.load(paths["valid"]).astype(bool)
    y = np.load(paths["y"])
    partition = np.load(paths["partition"])
    train = np.load(paths["train"]).astype(bool)
    validation = np.load(paths["validation"]).astype(bool)
    test = np.load(paths["test"]).astype(bool)
    shape = tuple(manifest["grid"]["shape"])
    if X_source.shape[:2] != shape or X_source.ndim != 3:
        raise ValueError(f"Unexpected feature shape: {X_source.shape}")
    if any(array.shape != shape for array in (valid, y, partition, train, validation, test)):
        raise ValueError(f"Fold grid mismatch: {fold_id}")
    if not np.array_equal(train, (partition == ROLE_CODES["train"]) & valid & (y != 2)):
        raise ValueError(f"Train mask rule mismatch: {fold_id}")
    if not np.array_equal(
        validation, (partition == ROLE_CODES["validation"]) & valid & (y != 2)
    ):
        raise ValueError(f"Validation mask rule mismatch: {fold_id}")
    if not np.array_equal(test, (partition == ROLE_CODES["test"]) & valid & (y != 2)):
        raise ValueError(f"Test mask rule mismatch: {fold_id}")
    if np.any(train & validation) or np.any(train & test) or np.any(validation & test):
        raise ValueError(f"Eligible role masks overlap: {fold_id}")

    channel_names = [
        f"{band}_{season}" for band in meta["bands"] for season in meta["seasons"]
    ]
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if len(channel_names) != X_source.shape[2] or stats["channel_names"] != channel_names:
        raise ValueError("Feature channel metadata differs from normalization metadata")
    if mean.shape != (X_source.shape[2],) or std.shape != mean.shape:
        raise ValueError(f"Normalization vector shape mismatch: mean={mean.shape}, std={std.shape}")
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or not np.all(std > 0):
        raise ValueError(f"Invalid normalization values: {fold_id}")

    X = np.asarray(X_source, dtype=np.float32)
    if normalize:
        X = np.empty_like(X_source, dtype=np.float32)
        np.subtract(X_source, mean, out=X)
        np.divide(X, std, out=X)
    else:
        X = X.copy()
    X[~valid] = 0.0
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Loaded feature tensor contains non-finite values: {fold_id}")

    return {
        "project_root": root,
        "fold_id": fold_id,
        "X": X,
        "y": y,
        "valid": valid,
        "partition": partition,
        "train": train,
        "val": validation,
        "test": test,
        "normalization": stats,
        "normalized": normalize,
        "channel_names": channel_names,
        "role_row_ranges": {
            role: [tuple(pair) for pair in fold_record["roles"][role]["row_ranges"]]
            for role in ROLE_CODES
        },
    }


def complete_patch_centers(
    partition: np.ndarray,
    *,
    role: str = "train",
    patch: int = PATCH,
) -> np.ndarray:
    """Return centers whose complete ``patch`` square has one role code.

    For an even patch, center ``(cy, cx)`` maps to
    ``[cy-patch//2:cy+patch//2, cx-patch//2:cx+patch//2]``.
    """

    if role not in ROLE_CODES:
        raise ValueError(f"Unknown role: {role}")
    if patch <= 0 or patch % 2:
        raise ValueError("Patch size must be a positive even integer")
    height, width = partition.shape
    if patch > height or patch > width:
        raise ValueError(f"Patch {patch} exceeds grid {partition.shape}")
    bad = (partition != ROLE_CODES[role]).astype(np.int32)
    integral = np.pad(bad.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)))
    patch_bad = (
        integral[patch:, patch:]
        - integral[:-patch, patch:]
        - integral[patch:, :-patch]
        + integral[:-patch, :-patch]
    )
    centers = np.zeros(partition.shape, dtype=bool)
    half = patch // 2
    centers[half:half + patch_bad.shape[0], half:half + patch_bad.shape[1]] = patch_bad == 0
    return centers


class FoldTrainPatchDataset(Dataset):
    """Online balanced patch sampling contained wholly in a fold's train role."""

    def __init__(
        self,
        data: dict,
        *,
        fault_frac: float = 0.5,
        n: int = 4096,
        seed: int = 0,
        patch: int = PATCH,
    ) -> None:
        if not 0.0 <= fault_frac <= 1.0:
            raise ValueError("fault_frac must be between 0 and 1")
        if not data.get("normalized"):
            raise ValueError("Training patches require fold-normalized features")
        self.X = data["X"]
        self.y = data["y"]
        self.train = data["train"]
        self.partition = data["partition"]
        self.patch = patch
        self.half = patch // 2
        self.fault_frac = fault_frac
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.complete_centers = complete_patch_centers(
            self.partition, role="train", patch=patch
        )
        eligible_centers = self.complete_centers & self.train
        self.pos_centers = np.argwhere(eligible_centers & (self.y == 1))
        self.bg_centers = np.argwhere(eligible_centers & (self.y == 0))
        if not len(self.pos_centers) or not len(self.bg_centers):
            raise ValueError(
                f"Empty training center pool: positive={len(self.pos_centers)}, "
                f"background={len(self.bg_centers)}"
            )

    def __len__(self) -> int:
        return self.n

    def _draw_center(self) -> tuple[int, int]:
        pool = self.pos_centers if self.rng.random() < self.fault_frac else self.bg_centers
        cy, cx = pool[self.rng.integers(len(pool))]
        return int(cy), int(cx)

    def _patch_at(
        self,
        cy: int,
        cx: int,
        *,
        flip_horizontal: bool,
        flip_vertical: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        half = self.half
        row_slice = slice(cy - half, cy + half)
        col_slice = slice(cx - half, cx + half)
        x = self.X[row_slice, col_slice]
        y = self.y[row_slice, col_slice]
        loss_mask = self.train[row_slice, col_slice]
        if x.shape[:2] != (self.patch, self.patch):
            raise RuntimeError(f"Incomplete patch at {(cy, cx)}")
        if not np.all(self.partition[row_slice, col_slice] == ROLE_CODES["train"]):
            raise RuntimeError(f"Patch crosses a non-training partition at {(cy, cx)}")
        if flip_horizontal:
            x, y, loss_mask = x[:, ::-1], y[:, ::-1], loss_mask[:, ::-1]
        if flip_vertical:
            x, y, loss_mask = x[::-1], y[::-1], loss_mask[::-1]
        if np.any(loss_mask & (y == 2)):
            raise RuntimeError("Loss mask includes ignored labels after augmentation")
        return (
            torch.from_numpy(x.transpose(2, 0, 1).copy()),
            torch.from_numpy(y.copy()).long(),
            torch.from_numpy(loss_mask.copy()).float(),
        )

    def __getitem__(self, _: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cy, cx = self._draw_center()
        return self._patch_at(
            cy,
            cx,
            flip_horizontal=bool(self.rng.random() < 0.5),
            flip_vertical=bool(self.rng.random() < 0.5),
        )


def _axis_starts(start: int, end: int, patch: int, stride: int) -> list[int]:
    if stride <= 0:
        raise ValueError("Stride must be positive")
    if end - start < patch:
        raise ValueError(f"Range [{start}, {end}) is shorter than patch {patch}")
    starts = set(range(start, end - patch + 1, stride))
    starts.add(end - patch)
    return sorted(starts)


def inference_window_starts(
    data: dict,
    *,
    role: str,
    patch: int = PATCH,
    stride: int = STRIDE,
) -> list[tuple[int, int]]:
    """List role-contained windows that fully cover every role-core pixel."""

    if role not in ROLE_CODES:
        raise ValueError(f"Unknown role: {role}")
    height, width = data["partition"].shape
    x_starts = _axis_starts(0, width, patch, stride)
    windows: list[tuple[int, int]] = []
    for row_start, row_end in data["role_row_ranges"][role]:
        for y0 in _axis_starts(row_start, row_end, patch, stride):
            for x0 in x_starts:
                if not np.all(
                    data["partition"][y0:y0 + patch, x0:x0 + patch]
                    == ROLE_CODES[role]
                ):
                    raise RuntimeError(
                        f"{role} inference window crosses a role boundary: {(y0, x0)}"
                    )
                windows.append((y0, x0))
    return windows


def inference_coverage(
    data: dict,
    *,
    role: str,
    patch: int = PATCH,
    stride: int = STRIDE,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    windows = inference_window_starts(data, role=role, patch=patch, stride=stride)
    coverage = np.zeros(data["partition"].shape, dtype=np.uint16)
    for y0, x0 in windows:
        coverage[y0:y0 + patch, x0:x0 + patch] += 1
    return coverage, windows


def sliding_predict_fold(
    model,
    data: dict,
    device,
    *,
    role: str = "test",
    patch: int = PATCH,
    stride: int = STRIDE,
    tta: bool = False,
) -> np.ndarray:
    """Predict with windows contained in and covering one fold role."""

    windows = inference_window_starts(data, role=role, patch=patch, stride=stride)
    X = data["X"]
    accumulator = np.zeros(data["partition"].shape, dtype=np.float64)
    count = np.zeros(data["partition"].shape, dtype=np.uint16)
    model.eval()
    with torch.no_grad():
        for y0, x0 in windows:
            patch_array = X[y0:y0 + patch, x0:x0 + patch]
            tensor = torch.from_numpy(
                patch_array.transpose(2, 0, 1).copy()
            )[None].to(device)
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
            if tta:
                flipped = torch.from_numpy(
                    patch_array[:, ::-1].transpose(2, 0, 1).copy()
                )[None].to(device)
                flipped_probability = torch.sigmoid(model(flipped))[0, 0].cpu().numpy()
                probability = 0.5 * (probability + flipped_probability[:, ::-1])
            accumulator[y0:y0 + patch, x0:x0 + patch] += probability
            count[y0:y0 + patch, x0:x0 + patch] += 1
    target = data["partition"] == ROLE_CODES[role]
    if not np.all(count[target] > 0) or np.any(count[~target] > 0):
        raise RuntimeError(f"Invalid {role} inference coverage")
    return np.divide(
        accumulator,
        np.maximum(count, 1),
        out=np.zeros_like(accumulator),
        where=count > 0,
    )


def pos_weight_from(data: dict) -> float:
    positive = int((data["train"] & (data["y"] == 1)).sum())
    negative = int((data["train"] & (data["y"] == 0)).sum())
    if positive == 0:
        raise ValueError("Training fold has no positive pixels")
    return negative / positive


def best_threshold(y: np.ndarray, probability: np.ndarray, mask: np.ndarray, n: int = 51):
    selected = mask & (y != 2)
    truth = (y[selected] == 1).astype(np.int8)
    score = np.clip(probability[selected], 0, 1)
    best_t, best_f1 = 0.5, -1.0
    for threshold in np.linspace(0.0, 1.0, n):
        predicted = score >= threshold
        tp = int((predicted & (truth == 1)).sum())
        fp = int((predicted & (truth == 0)).sum())
        fn = int((~predicted & (truth == 1)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        if f1 > best_f1:
            best_t, best_f1 = threshold, f1
    return float(best_t), float(best_f1)


def metrics(y: np.ndarray, probability: np.ndarray, mask: np.ndarray, thresh: float = 0.5):
    from sklearn.metrics import average_precision_score, roc_auc_score

    selected = mask & (y != 2)
    truth = (y[selected] == 1).astype(np.int8)
    score = np.clip(probability[selected], 0, 1)
    auc = roc_auc_score(truth, score)
    average_precision = average_precision_score(truth, score)
    predicted = score >= thresh
    tp = int((predicted & (truth == 1)).sum())
    fp = int((predicted & (truth == 0)).sum())
    fn = int((~predicted & (truth == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "n": int(len(truth)),
        "pos": int(truth.sum()),
        "auc": float(auc),
        "ap": float(average_precision),
        "thresh": float(thresh),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def rf_features(data: dict, ctx_channels: Iterable[int] | None = None) -> np.ndarray:
    """Build the frozen V2 RF feature family from fold-normalized V3 inputs."""

    X = data["X"]
    if ctx_channels is None:
        ctx_channels = list(range(4)) + list(range(8, 12)) + list(range(16, 20))
    features = [X]
    for channel in ctx_channels:
        mean = uniform_filter(X[..., channel], size=7, mode="constant")
        second = uniform_filter(X[..., channel] ** 2, size=7, mode="constant")
        std = np.sqrt(np.maximum(second - mean ** 2, 0))
        features.extend((mean[..., None], std[..., None]))
    return np.concatenate(features, axis=-1)
