#!/usr/bin/env python3
"""Fit leakage-controlled normalization statistics for this study V3 folds.

Stage 3 reads the immutable V2 feature tensor and the accepted stage-2 fold
masks. For each fold, channel-wise z-score statistics are fitted exclusively
on pixels selected by that fold's ``train.npy`` mask. Outputs are written to a
new ``01_data/interim/normalization_v3`` directory so ``dataset_v3`` remains
unchanged. This stage does not normalize the full tensor or train a model.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"No rows supplied for {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    dataset_v2 = root / "01_data/interim/dataset_v2"
    dataset_v3 = root / "01_data/interim/dataset_v3"
    output_dir = (
        args.output_dir or root / "01_data/interim/normalization_v3"
    ).resolve()
    manifest_path = root / "04_results/split_design_v3/fold_manifest_v3.json"
    stage2_acceptance_path = dataset_v3 / "stage2_acceptance_v3.json"
    stage2_integrity_path = dataset_v3 / "source_integrity_v3.json"
    required = {
        "X": dataset_v2 / "X.npy",
        "valid": dataset_v2 / "valid.npy",
        "meta": dataset_v2 / "meta.json",
        "stage1_manifest": manifest_path,
        "stage2_acceptance": stage2_acceptance_path,
        "stage2_integrity": stage2_integrity_path,
    }
    for name, path in required.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing required {name}: {path}")

    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
        if output_dir == root or root not in output_dir.parents:
            raise ValueError(f"Unsafe output directory for removal: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    stage2_acceptance = json.loads(stage2_acceptance_path.read_text(encoding="utf-8"))
    stage2_integrity = json.loads(stage2_integrity_path.read_text(encoding="utf-8"))
    meta = json.loads(required["meta"].read_text(encoding="utf-8"))
    if not stage2_acceptance.get("overall_accepted"):
        raise ValueError("Stage-2 fold labels and masks are not accepted")
    if not all(item["passed"] for item in stage2_acceptance["checks"]):
        raise ValueError("Stage-2 acceptance contains a failed check")

    X = np.load(required["X"], mmap_mode="r")
    valid = np.load(required["valid"], mmap_mode="r").astype(bool)
    shape = tuple(manifest["grid"]["shape"])
    if X.shape[:2] != shape or valid.shape != shape:
        raise ValueError(f"Grid mismatch: X={X.shape}, valid={valid.shape}, expected={shape}")
    if X.dtype != np.float32:
        raise ValueError(f"Expected float32 X, got {X.dtype}")

    bands = list(meta["bands"])
    seasons = list(meta["seasons"])
    channel_names = [f"{band}_{season}" for band in bands for season in seasons]
    if len(channel_names) != X.shape[2]:
        raise ValueError(
            f"Metadata defines {len(channel_names)} channels but X has {X.shape[2]}"
        )
    if not str(meta["channel_order"]).startswith("band-major, season-minor"):
        raise ValueError(f"Unexpected channel order: {meta['channel_order']}")

    source_paths = dict(required)
    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        fold_dir = dataset_v3 / fold_id
        for filename in ("y.npy", "partition.npy", "train.npy", "val.npy", "test.npy"):
            key = f"{fold_id}/{filename}"
            source_paths[key] = fold_dir / filename
            if not source_paths[key].is_file():
                raise FileNotFoundError(f"Missing stage-2 fold source: {source_paths[key]}")

    hashes_before = {name: sha256(path) for name, path in source_paths.items()}
    expected_x_hash = stage2_integrity["source_files"]["X"]["sha256_after"]
    x_hash_matches_stage2 = hashes_before["X"] == expected_x_hash
    if not x_hash_matches_stage2:
        raise ValueError("X.npy no longer matches the accepted stage-2 source hash")

    checks: list[dict] = [
        {
            "fold": "global",
            "check": "stage2_acceptance_is_clean",
            "passed": True,
        },
        {
            "fold": "global",
            "check": "X_hash_matches_stage2",
            "passed": x_hash_matches_stage2,
            "sha256": hashes_before["X"],
        },
        {
            "fold": "global",
            "check": "channel_metadata_matches_X",
            "passed": len(channel_names) == X.shape[2],
            "channels": X.shape[2],
        },
    ]
    summary_rows: list[dict] = []
    fold_records: list[dict] = []

    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        fold_source = dataset_v3 / fold_id
        fold_out = output_dir / fold_id
        fold_out.mkdir()
        train = np.load(fold_source / "train.npy", mmap_mode="r")
        val = np.load(fold_source / "val.npy", mmap_mode="r")
        test = np.load(fold_source / "test.npy", mmap_mode="r")
        y = np.load(fold_source / "y.npy", mmap_mode="r")
        partition = np.load(fold_source / "partition.npy", mmap_mode="r")
        if any(array.shape != shape for array in (train, val, test, y, partition)):
            raise ValueError(f"Fold array shape mismatch: {fold_id}")

        expected_train = (partition == 1) & valid & (y != 2)
        mask_exact = bool(np.array_equal(train, expected_train))
        masks_disjoint = bool(
            not np.any(train & val)
            and not np.any(train & test)
            and not np.any(val & test)
        )
        sample_count = int(train.sum())
        expected_count = int(fold["roles"]["train"]["counts"]["evaluation_pixels"])
        count_matches = sample_count == expected_count
        checks.extend(
            [
                {
                    "fold": fold_id,
                    "check": "train_mask_exactly_matches_partition_valid_label",
                    "passed": mask_exact,
                },
                {
                    "fold": fold_id,
                    "check": "train_validation_test_masks_disjoint",
                    "passed": masks_disjoint,
                },
                {
                    "fold": fold_id,
                    "check": "training_sample_count_matches_manifest",
                    "passed": count_matches,
                    "actual": sample_count,
                    "expected": expected_count,
                },
            ]
        )
        if not (mask_exact and masks_disjoint and count_matches):
            raise ValueError(f"Fold mask acceptance failed before fitting: {fold_id}")

        values = np.asarray(X[train], dtype=np.float64)
        finite = np.isfinite(values)
        finite_counts = finite.sum(axis=0)
        all_finite = bool(np.all(finite_counts == sample_count))
        if not all_finite:
            raise ValueError(f"Non-finite feature values in {fold_id}")
        means = values.mean(axis=0, dtype=np.float64)
        stds = values.std(axis=0, ddof=0, dtype=np.float64)
        minima = values.min(axis=0)
        maxima = values.max(axis=0)
        stats_valid = bool(
            np.all(np.isfinite(means))
            and np.all(np.isfinite(stds))
            and np.all(stds > 0.0)
            and np.all(maxima >= minima)
        )
        checks.extend(
            [
                {
                    "fold": fold_id,
                    "check": "all_selected_feature_values_finite",
                    "passed": all_finite,
                    "sample_count_per_channel": sample_count,
                },
                {
                    "fold": fold_id,
                    "check": "all_channel_statistics_finite_and_nonconstant",
                    "passed": stats_valid,
                    "minimum_std": float(stds.min()),
                },
            ]
        )
        if not stats_valid:
            raise ValueError(f"Invalid channel statistics in {fold_id}")

        norm_payload = {
            "schema_version": "fault-proxy-v3-normalization-1.0",
            "fold_id": fold_id,
            "method": "channel-wise z-score fitted only on train.npy true pixels",
            "accumulator_dtype": "float64",
            "standard_deviation": "population standard deviation (ddof=0)",
            "training_mask_semantics": "partition=train AND valid AND fold label is not ignore",
            "sample_count": sample_count,
            "channel_count": X.shape[2],
            "channel_order": meta["channel_order"],
            "channel_names": channel_names,
            "mean": [float(value) for value in means],
            "std": [float(value) for value in stds],
            "min": [float(value) for value in minima],
            "max": [float(value) for value in maxima],
            "finite_count": [int(value) for value in finite_counts],
            "input_sha256": {
                "X": hashes_before["X"],
                "train_mask": hashes_before[f"{fold_id}/train.npy"],
                "fold_label": hashes_before[f"{fold_id}/y.npy"],
                "partition": hashes_before[f"{fold_id}/partition.npy"],
                "stage1_manifest": hashes_before["stage1_manifest"],
                "stage2_acceptance": hashes_before["stage2_acceptance"],
            },
        }
        norm_path = fold_out / "norm_stats.json"
        write_json(norm_path, norm_payload)

        for index, channel_name in enumerate(channel_names):
            band_index, season_index = divmod(index, len(seasons))
            summary_rows.append(
                {
                    "fold_id": fold_id,
                    "channel_index": index,
                    "channel_name": channel_name,
                    "band": bands[band_index],
                    "season": seasons[season_index],
                    "sample_count": sample_count,
                    "finite_count": int(finite_counts[index]),
                    "mean": format(float(means[index]), ".17g"),
                    "std_ddof0": format(float(stds[index]), ".17g"),
                    "minimum": format(float(minima[index]), ".17g"),
                    "maximum": format(float(maxima[index]), ".17g"),
                }
            )
        fold_records.append(
            {
                "fold_id": fold_id,
                "stats_path": str(norm_path.relative_to(output_dir)),
                "stats_sha256": sha256(norm_path),
                "sample_count": sample_count,
                "channels": X.shape[2],
                "train_mask_sha256": hashes_before[f"{fold_id}/train.npy"],
            }
        )
        del values

    write_csv(output_dir / "normalization_stats_v3.csv", summary_rows)
    write_json(
        output_dir / "normalization_manifest_v3.json",
        {
            "schema_version": "fault-proxy-v3-normalization-manifest-1.0",
            "status": "generated_pending_stage3_acceptance",
            "feature_source": str(required["X"].relative_to(root)),
            "feature_sha256": hashes_before["X"],
            "feature_shape": list(X.shape),
            "feature_dtype": str(X.dtype),
            "bands": bands,
            "seasons": seasons,
            "channel_names": channel_names,
            "folds": fold_records,
        },
    )

    hashes_after = {name: sha256(path) for name, path in source_paths.items()}
    sources_unchanged = hashes_before == hashes_after
    checks.append(
        {
            "fold": "global",
            "check": "all_source_files_unchanged_during_stage3",
            "passed": sources_unchanged,
        }
    )
    write_json(
        output_dir / "normalization_source_integrity_v3.json",
        {
            "source_files": {
                name: {
                    "path": str(path.relative_to(root)),
                    "sha256_before": hashes_before[name],
                    "sha256_after": hashes_after[name],
                    "unchanged": hashes_before[name] == hashes_after[name],
                }
                for name, path in source_paths.items()
            },
            "all_sources_unchanged": sources_unchanged,
            "X_matches_stage2_accepted_hash": x_hash_matches_stage2,
        },
    )

    overall_accepted = all(item["passed"] for item in checks)
    write_json(
        output_dir / "stage3_acceptance_v3.json",
        {
            "overall_accepted": overall_accepted,
            "status": (
                "accepted_for_training_loader_integration"
                if overall_accepted else "rejected"
            ),
            "next_gate": (
                "Integrate fold-specific masks and normalization into a V3 loader and verify patch "
                "containment and inference coverage before any model training."
            ),
            "checks": checks,
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# normalization_v3 — stage 3",
                "",
                f"Status: {'accepted' if overall_accepted else 'rejected'}.",
                "",
                "Each fold's `norm_stats.json` contains 32-channel z-score means and population",
                "standard deviations fitted only on that fold's eligible `train.npy` pixels.",
                "",
                "The V2 feature tensor and stage-2 V3 masks are referenced by SHA-256 and were not",
                "modified. No full normalized tensor, model checkpoint, or training result is included.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "normalization_checksums_v3.sha256":
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "normalization_checksums_v3.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "overall_accepted": overall_accepted,
                "folds": len(fold_records),
                "channels_per_fold": X.shape[2],
                "checks": len(checks),
                "failed_checks": sum(not item["passed"] for item in checks),
                "next_gate": "V3 loader integration; no model training yet",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not overall_accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
