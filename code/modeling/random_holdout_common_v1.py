#!/usr/bin/env python3
"""Data and inference primitives for the random-holdout diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def protocol_context(root: Path, confirm_sha: str | None = None) -> dict:
    path = root / "03_configs/random_holdout_v1_protocol.json"
    protocol = read_json(path)
    digest = sha256(path)
    if protocol.get("status") != "frozen_before_training":
        raise ValueError("Random-holdout protocol is not frozen")
    if confirm_sha is not None and confirm_sha != digest:
        raise ValueError(f"Protocol confirmation mismatch: {confirm_sha} != {digest}")
    return {"root": root, "path": path, "protocol": protocol, "sha256": digest}


def load_random_split(root: Path, split_id: str, protocol_sha: str) -> dict:
    context = protocol_context(root, protocol_sha)
    records = {item["split_id"]: item for item in context["protocol"]["splits"]}
    if split_id not in records:
        raise ValueError(f"Unknown split: {split_id}")
    split_dir = root / "01_data/interim/random_holdout_v1" / protocol_sha / split_id
    manifest = read_json(split_dir / "split_manifest.json")
    if manifest.get("status") != "accepted" or manifest.get("protocol_sha256") != protocol_sha:
        raise ValueError(f"Unaccepted random split: {split_id}")
    names = {
        "X": root / "01_data/interim/dataset_v2/X.npy",
        "valid": root / "01_data/interim/dataset_v2/valid.npy",
        "y": root / "01_data/interim/dataset_v3" / records[split_id]["source_fold"] / "y.npy",
        "train": split_dir / "train.npy",
        "validation": split_dir / "val.npy",
        "test": split_dir / "test.npy",
        "normalization": split_dir / "norm_stats.json",
    }
    for name, path in names.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    for name in ("X", "valid", "y", "train", "validation", "test", "normalization"):
        if sha256(names[name]) != manifest["file_sha256"][name]:
            raise ValueError(f"Hash mismatch for {split_id} {name}")
    X_source = np.load(names["X"])
    valid = np.load(names["valid"]).astype(bool)
    y = np.load(names["y"])
    train = np.load(names["train"]).astype(bool)
    val = np.load(names["validation"]).astype(bool)
    test = np.load(names["test"]).astype(bool)
    stats = read_json(names["normalization"])
    if any(array.shape != y.shape for array in (valid, train, val, test)):
        raise ValueError(f"Shape mismatch in {split_id}")
    if np.any(train & val) or np.any(train & test) or np.any(val & test):
        raise ValueError(f"Role overlap in {split_id}")
    if np.any((train | val | test) & (~valid | (y == 2))):
        raise ValueError(f"Ineligible assigned pixel in {split_id}")
    mean = np.asarray(stats["mean"], dtype=np.float32)
    std = np.asarray(stats["std"], dtype=np.float32)
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)) or np.any(std <= 0):
        raise ValueError(f"Invalid normalization in {split_id}")
    X = np.asarray(X_source, dtype=np.float32).copy()
    np.subtract(X, mean, out=X)
    np.divide(X, std, out=X)
    X[~valid] = 0.0
    if not np.all(np.isfinite(X)):
        raise ValueError(f"Non-finite normalized features in {split_id}")
    return {"X": X, "valid": valid, "y": y, "train": train, "val": val,
            "test": test, "split_id": split_id, "source_fold": records[split_id]["source_fold"],
            "normalization": stats, "protocol_sha256": protocol_sha}


class RandomTrainPatchDataset(Dataset):
    """Sample random-training centers; only randomized train labels enter loss."""

    def __init__(self, data: dict, *, n: int, seed: int, patch: int = 128) -> None:
        if patch <= 0 or patch % 2:
            raise ValueError("Patch must be a positive even number")
        self.X, self.y, self.train = data["X"], data["y"], data["train"]
        self.n, self.patch, self.half = n, patch, patch // 2
        self.rng = np.random.default_rng(seed)
        center_ok = np.zeros_like(self.train)
        center_ok[self.half:self.train.shape[0] - self.half + 1,
                  self.half:self.train.shape[1] - self.half + 1] = True
        eligible = center_ok & self.train
        self.pos = np.argwhere(eligible & (self.y == 1))
        self.neg = np.argwhere(eligible & (self.y == 0))
        if not len(self.pos) or not len(self.neg):
            raise ValueError("Empty positive or negative center pool")
        self.positive_fraction = len(self.pos) / (len(self.pos) + len(self.neg))

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, _: int):
        pool = self.pos if self.rng.random() < self.positive_fraction else self.neg
        cy, cx = map(int, pool[self.rng.integers(len(pool))])
        rs, cs = slice(cy - self.half, cy + self.half), slice(cx - self.half, cx + self.half)
        x, y, mask = self.X[rs, cs], self.y[rs, cs], self.train[rs, cs]
        if x.shape[:2] != (self.patch, self.patch) or not np.any(mask):
            raise RuntimeError("Invalid random-holdout training patch")
        if self.rng.random() < 0.5:
            x, y, mask = x[:, ::-1], y[:, ::-1], mask[:, ::-1]
        if self.rng.random() < 0.5:
            x, y, mask = x[::-1], y[::-1], mask[::-1]
        if np.any(mask & (y == 2)):
            raise RuntimeError("Ignored label entered loss mask")
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(y.copy()).long(), torch.from_numpy(mask.copy()).float())


def _starts(length: int, patch: int, stride: int) -> list[int]:
    values = set(range(0, length - patch + 1, stride))
    values.add(length - patch)
    return sorted(values)


def blend_weight(patch: int, floor: float) -> np.ndarray:
    coordinate = np.arange(patch, dtype=np.float64)
    axis = floor + (1.0 - floor) * np.sin(np.pi * (coordinate + 0.5) / patch) ** 2
    return np.outer(axis, axis)


def full_grid_predict(model, X: np.ndarray, device: torch.device, *, patch: int = 128,
                      stride: int = 64, floor: float = 0.05, batch_size: int = 16) -> np.ndarray:
    windows = [(y0, x0) for y0 in _starts(X.shape[0], patch, stride)
               for x0 in _starts(X.shape[1], patch, stride)]
    weight = blend_weight(patch, floor)
    accumulator = np.zeros(X.shape[:2], dtype=np.float64)
    denominator = np.zeros(X.shape[:2], dtype=np.float64)
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(windows), batch_size):
            group = windows[offset:offset + batch_size]
            arrays = np.stack([X[y:y + patch, x:x + patch].transpose(2, 0, 1)
                               for y, x in group]).astype(np.float32, copy=False)
            probabilities = torch.sigmoid(model(torch.from_numpy(arrays).to(device)))[:, 0].cpu().numpy()
            for (y0, x0), probability in zip(group, probabilities):
                accumulator[y0:y0 + patch, x0:x0 + patch] += probability * weight
                denominator[y0:y0 + patch, x0:x0 + patch] += weight
    if not np.all(denominator > 0):
        raise RuntimeError("Full-grid inference left uncovered pixels")
    output = accumulator / denominator
    if not np.all(np.isfinite(output)) or output.min() < 0 or output.max() > 1:
        raise RuntimeError("Invalid full-grid probability")
    return output.astype(np.float32)
