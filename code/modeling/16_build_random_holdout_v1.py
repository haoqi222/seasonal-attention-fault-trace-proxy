#!/usr/bin/env python3
"""Build exact-count stratified random-pixel holdouts for this study."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from random_holdout_common_v1 import protocol_context, sha256  # noqa: E402


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def counts(mask: np.ndarray, y: np.ndarray) -> dict:
    return {"n": int(mask.sum()), "positive": int((mask & (y == 1)).sum()),
            "negative": int((mask & (y == 0)).sum())}


def run(root: Path, confirm_sha: str) -> None:
    context = protocol_context(root.resolve(), confirm_sha)
    protocol, digest = context["protocol"], context["sha256"]
    destination = root / "01_data/interim/random_holdout_v1" / digest
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite prepared diagnostic data: {destination}")
    attempt = destination.with_name(destination.name + ".attempt")
    if attempt.exists():
        raise FileExistsError(f"Existing build attempt requires review: {attempt}")
    attempt.mkdir(parents=True)
    X_path = root / protocol["source"]["feature_cube"]
    valid_path = root / protocol["source"]["valid_mask"]
    X = np.load(X_path, mmap_mode="r")
    valid = np.load(valid_path).astype(bool)
    summaries = []
    for record in protocol["splits"]:
        split_id, fold = record["split_id"], record["source_fold"]
        source = root / protocol["source"]["fold_dataset"] / fold
        y_path = source / "y.npy"
        y = np.load(y_path)
        original = {"train": np.load(source / "train.npy").astype(bool),
                    "validation": np.load(source / "val.npy").astype(bool),
                    "test": np.load(source / "test.npy").astype(bool)}
        if np.any(original["train"] & original["validation"]) or np.any(original["train"] & original["test"]) or np.any(original["validation"] & original["test"]):
            raise ValueError(f"Source masks overlap: {fold}")
        universe = original["train"] | original["validation"] | original["test"]
        if np.any(universe & (~valid | (y == 2))):
            raise ValueError(f"Source universe includes ineligible cells: {fold}")
        random_masks = {role: np.zeros_like(universe) for role in original}
        rng = np.random.default_rng(int(record["split_seed"]))
        for label in (0, 1):
            flat = np.flatnonzero(universe & (y == label))
            flat = flat[rng.permutation(len(flat))]
            cursor = 0
            for role in ("train", "validation", "test"):
                take = int((original[role] & (y == label)).sum())
                random_masks[role].flat[flat[cursor:cursor + take]] = True
                cursor += take
            if cursor != len(flat):
                raise RuntimeError(f"Class allocation mismatch: {split_id} label={label}")
        assigned = random_masks["train"] | random_masks["validation"] | random_masks["test"]
        if not np.array_equal(assigned, universe) or np.any(random_masks["train"] & random_masks["validation"]) or np.any(random_masks["train"] & random_masks["test"]) or np.any(random_masks["validation"] & random_masks["test"]):
            raise RuntimeError(f"Random split partition failure: {split_id}")
        split_dir = attempt / split_id
        split_dir.mkdir()
        for role, filename in (("train", "train.npy"), ("validation", "val.npy"), ("test", "test.npy")):
            if counts(random_masks[role], y) != counts(original[role], y):
                raise RuntimeError(f"Count mismatch: {split_id} {role}")
            np.save(split_dir / filename, random_masks[role])
        train_values = np.asarray(X[random_masks["train"]], dtype=np.float64)
        mean, std = train_values.mean(axis=0), train_values.std(axis=0)
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(std)) or np.any(std <= 0):
            raise RuntimeError(f"Invalid training normalization: {split_id}")
        stats = {"split_id": split_id, "source_fold": fold, "protocol_sha256": digest,
                 "fit_mask": "randomized training pixels only", "ddof": 0,
                 "n": int(len(train_values)), "mean": mean.tolist(), "std": std.tolist()}
        write_json(split_dir / "norm_stats.json", stats)
        file_paths = {"X": X_path, "valid": valid_path, "y": y_path,
                      "train": split_dir / "train.npy", "validation": split_dir / "val.npy",
                      "test": split_dir / "test.npy", "normalization": split_dir / "norm_stats.json"}
        manifest = {"status": "accepted", "protocol_sha256": digest, "split_id": split_id,
                    "source_fold": fold, "split_seed": record["split_seed"],
                    "source_counts": {role: counts(mask, y) for role, mask in original.items()},
                    "random_counts": {role: counts(mask, y) for role, mask in random_masks.items()},
                    "exact_class_counts_matched": True, "roles_disjoint": True,
                    "assigned_union_equals_source_union": True,
                    "normalization_fit_on_train_only": True,
                    "file_sha256": {name: sha256(path) for name, path in file_paths.items()}}
        write_json(split_dir / "split_manifest.json", manifest)
        summaries.append(manifest)
    write_json(attempt / "build_acceptance.json", {"status": "accepted", "protocol_sha256": digest,
               "split_count": len(summaries), "all_exact": all(x["exact_class_counts_matched"] for x in summaries),
               "splits": summaries})
    destination.parent.mkdir(parents=True, exist_ok=True)
    attempt.rename(destination)
    print(json.dumps({"status": "accepted", "protocol_sha256": digest, "output": str(destination)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirm-protocol-sha", required=True)
    args = parser.parse_args()
    run(args.project_root, args.confirm_protocol_sha)
