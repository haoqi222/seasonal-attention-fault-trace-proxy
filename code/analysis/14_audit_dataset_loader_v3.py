#!/usr/bin/env python3
"""Run the V3 stage-4 loader and boundary integration audit.

No model, optimizer, checkpoint, or training loop is created. The audit loads
all four folds, exercises deterministic patch extraction and augmentation,
and verifies complete validation/test window coverage without crossing role
boundaries.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch


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
    output_dir = (
        args.output_dir or root / "04_results/loader_audit_v3"
    ).resolve()
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Refusing to overwrite existing output: {output_dir}")
        if output_dir == root or root not in output_dir.parents:
            raise ValueError(f"Unsafe output directory for removal: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    modeling_dir = root / "02_scripts/modeling"
    sys.path.insert(0, str(modeling_dir))
    from dataset_common_v3 import (  # pylint: disable=import-outside-toplevel
        PATCH,
        ROLE_CODES,
        FoldTrainPatchDataset,
        available_folds,
        inference_coverage,
        load_fold,
        pos_weight_from,
    )

    fold_manifest_path = root / "04_results/split_design_v3/fold_manifest_v3.json"
    manifest = json.loads(fold_manifest_path.read_text(encoding="utf-8"))
    fold_lookup = {record["fold_id"]: record for record in manifest["folds"]}
    fold_ids = available_folds(root)
    if fold_ids != [record["fold_id"] for record in manifest["folds"]]:
        raise ValueError("Loader fold order differs from frozen manifest")

    dataset_v2 = root / "01_data/interim/dataset_v2"
    dataset_v3 = root / "01_data/interim/dataset_v3"
    normalization_v3 = root / "01_data/interim/normalization_v3"
    source_paths = {
        "X": dataset_v2 / "X.npy",
        "valid": dataset_v2 / "valid.npy",
        "meta": dataset_v2 / "meta.json",
        "fold_manifest": fold_manifest_path,
        "stage2_acceptance": dataset_v3 / "stage2_acceptance_v3.json",
        "stage3_acceptance": normalization_v3 / "stage3_acceptance_v3.json",
        "loader": modeling_dir / "dataset_common_v3.py",
    }
    for fold_id in fold_ids:
        for filename in ("y.npy", "partition.npy", "train.npy", "val.npy", "test.npy"):
            source_paths[f"{fold_id}/{filename}"] = dataset_v3 / fold_id / filename
        source_paths[f"{fold_id}/norm_stats.json"] = (
            normalization_v3 / fold_id / "norm_stats.json"
        )
    for name, path in source_paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing loader-audit source {name}: {path}")
    hashes_before = {name: sha256(path) for name, path in source_paths.items()}

    checks: list[dict] = []
    summary_rows: list[dict] = []

    def record(fold: str, check: str, passed: bool, **values) -> None:
        checks.append({"fold": fold, "check": check, "passed": bool(passed), **values})

    record("global", "four_frozen_folds_available", len(fold_ids) == 4, count=len(fold_ids))

    for fold_id in fold_ids:
        data = load_fold(fold_id, project_root=root, normalize=True, verify_hashes=True)
        fold_record = fold_lookup[fold_id]
        record(fold_id, "loader_selected_requested_fold", data["fold_id"] == fold_id)
        record(
            fold_id,
            "role_masks_match_partition_valid_label",
            np.array_equal(data["train"], (data["partition"] == 1) & data["valid"] & (data["y"] != 2))
            and np.array_equal(data["val"], (data["partition"] == 2) & data["valid"] & (data["y"] != 2))
            and np.array_equal(data["test"], (data["partition"] == 3) & data["valid"] & (data["y"] != 2)),
        )
        record(
            fold_id,
            "role_masks_pairwise_disjoint",
            not np.any(data["train"] & data["val"])
            and not np.any(data["train"] & data["test"])
            and not np.any(data["val"] & data["test"]),
        )
        record(
            fold_id,
            "loaded_features_finite_and_invalid_zero",
            np.all(np.isfinite(data["X"])) and np.all(data["X"][~data["valid"]] == 0),
        )

        train_values = np.asarray(data["X"][data["train"]], dtype=np.float64)
        normalized_mean = train_values.mean(axis=0, dtype=np.float64)
        normalized_std = train_values.std(axis=0, ddof=0, dtype=np.float64)
        max_abs_mean = float(np.max(np.abs(normalized_mean)))
        max_abs_std_error = float(np.max(np.abs(normalized_std - 1.0)))
        record(
            fold_id,
            "training_channels_normalized_to_zero_mean",
            max_abs_mean <= 5e-5,
            max_abs_mean=max_abs_mean,
            tolerance=5e-5,
        )
        record(
            fold_id,
            "training_channels_normalized_to_unit_std",
            max_abs_std_error <= 5e-5,
            max_abs_std_error=max_abs_std_error,
            tolerance=5e-5,
        )

        patch_dataset = FoldTrainPatchDataset(data, n=64, seed=20260902, patch=PATCH)
        height, width = data["partition"].shape
        expected_complete_centers = sum(
            (end - start - PATCH + 1) * (width - PATCH + 1)
            for start, end in fold_record["roles"]["train"]["row_ranges"]
        )
        actual_complete_centers = int(patch_dataset.complete_centers.sum())
        record(
            fold_id,
            "complete_training_center_count_matches_geometry",
            actual_complete_centers == expected_complete_centers,
            actual=actual_complete_centers,
            expected=expected_complete_centers,
        )
        positive_centers = len(patch_dataset.pos_centers)
        background_centers = len(patch_dataset.bg_centers)
        record(
            fold_id,
            "positive_and_background_center_pools_nonempty",
            positive_centers > 0 and background_centers > 0,
            positive_centers=positive_centers,
            background_centers=background_centers,
        )

        representative_centers = [
            tuple(map(int, patch_dataset.pos_centers[0])),
            tuple(map(int, patch_dataset.bg_centers[0])),
            tuple(map(int, patch_dataset.pos_centers[-1])),
            tuple(map(int, patch_dataset.bg_centers[-1])),
        ]
        contained = True
        for cy, cx in representative_centers:
            half = PATCH // 2
            contained &= bool(
                np.all(
                    data["partition"][cy - half:cy + half, cx - half:cx + half]
                    == ROLE_CODES["train"]
                )
            )
        record(fold_id, "representative_training_patches_fully_contained", contained)

        cy, cx = representative_centers[0]
        half = PATCH // 2
        source_y = data["y"][cy - half:cy + half, cx - half:cx + half]
        source_mask = data["train"][cy - half:cy + half, cx - half:cx + half]
        _, y_flip, mask_flip = patch_dataset._patch_at(
            cy, cx, flip_horizontal=True, flip_vertical=True
        )
        augmentation_aligned = bool(
            np.array_equal(y_flip.numpy(), source_y[::-1, ::-1])
            and np.array_equal(mask_flip.numpy().astype(bool), source_mask[::-1, ::-1])
        )
        record(
            fold_id,
            "augmentation_flips_features_labels_and_loss_mask_together",
            augmentation_aligned,
        )

        dataset_a = FoldTrainPatchDataset(data, n=16, seed=314159, patch=PATCH)
        dataset_b = FoldTrainPatchDataset(data, n=16, seed=314159, patch=PATCH)
        deterministic = True
        sampled_shapes_valid = True
        sampled_masks_valid = True
        for index in range(16):
            sample_a = dataset_a[index]
            sample_b = dataset_b[index]
            deterministic &= all(torch.equal(a, b) for a, b in zip(sample_a, sample_b))
            x_tensor, y_tensor, mask_tensor = sample_a
            sampled_shapes_valid &= (
                tuple(x_tensor.shape) == (data["X"].shape[2], PATCH, PATCH)
                and tuple(y_tensor.shape) == (PATCH, PATCH)
                and tuple(mask_tensor.shape) == (PATCH, PATCH)
            )
            sampled_masks_valid &= bool(
                torch.isfinite(x_tensor).all()
                and not torch.any((mask_tensor > 0) & (y_tensor == 2))
                and torch.any(mask_tensor > 0)
            )
        record(fold_id, "online_patch_sampling_is_seed_deterministic", deterministic)
        record(fold_id, "sampled_patch_tensor_shapes_are_valid", sampled_shapes_valid)
        record(fold_id, "sampled_loss_masks_exclude_ignore", sampled_masks_valid)

        positive = int((data["train"] & (data["y"] == 1)).sum())
        negative = int((data["train"] & (data["y"] == 0)).sum())
        expected_pos_weight = negative / positive
        pos_weight = pos_weight_from(data)
        record(
            fold_id,
            "positive_weight_uses_only_eligible_training_labels",
            np.isclose(pos_weight, expected_pos_weight, rtol=0.0, atol=0.0),
            pos_weight=pos_weight,
        )

        coverage_results = {}
        for role in ("validation", "test"):
            coverage, windows = inference_coverage(
                data, role=role, patch=PATCH, stride=64
            )
            target = data["partition"] == ROLE_CODES[role]
            coverage_ok = bool(
                len(windows) > 0
                and np.all(coverage[target] > 0)
                and not np.any(coverage[~target] > 0)
            )
            windows_contained = all(
                np.all(
                    data["partition"][y0:y0 + PATCH, x0:x0 + PATCH]
                    == ROLE_CODES[role]
                )
                for y0, x0 in windows
            )
            record(
                fold_id,
                f"{role}_windows_cover_only_complete_role",
                coverage_ok and windows_contained,
                windows=len(windows),
                min_coverage=int(coverage[target].min()),
                max_coverage=int(coverage[target].max()),
            )
            coverage_results[role] = {
                "windows": len(windows),
                "min": int(coverage[target].min()),
                "max": int(coverage[target].max()),
            }

        summary_rows.append(
            {
                "fold_id": fold_id,
                "training_pixels": int(data["train"].sum()),
                "positive_training_pixels": positive,
                "negative_training_pixels": negative,
                "pos_weight": format(pos_weight, ".17g"),
                "max_abs_normalized_train_mean": format(max_abs_mean, ".17g"),
                "max_abs_normalized_train_std_error": format(max_abs_std_error, ".17g"),
                "complete_training_centers": actual_complete_centers,
                "positive_training_centers": positive_centers,
                "background_training_centers": background_centers,
                "validation_windows": coverage_results["validation"]["windows"],
                "validation_min_coverage": coverage_results["validation"]["min"],
                "validation_max_coverage": coverage_results["validation"]["max"],
                "test_windows": coverage_results["test"]["windows"],
                "test_min_coverage": coverage_results["test"]["min"],
                "test_max_coverage": coverage_results["test"]["max"],
            }
        )
        del train_values, patch_dataset, dataset_a, dataset_b, data
        gc.collect()

    hashes_after = {name: sha256(path) for name, path in source_paths.items()}
    sources_unchanged = hashes_before == hashes_after
    record(
        "global",
        "all_loader_sources_unchanged_during_stage4",
        sources_unchanged,
    )
    write_csv(output_dir / "loader_audit_summary_v3.csv", summary_rows)
    write_json(
        output_dir / "loader_contract_v3.json",
        {
            "schema_version": "fault-proxy-v3-loader-contract-1.0",
            "loader": "02_scripts/modeling/dataset_common_v3.py",
            "loader_sha256": hashes_before["loader"],
            "fold_selection": "explicit fold_id is mandatory",
            "normalization": "fold-specific training-only mean/std",
            "training_patch": {
                "size": PATCH,
                "containment": "every input pixel must have partition code 1",
                "loss_mask": "eligible train.npy pixels only",
                "augmentation": "features, labels, and loss mask flip together",
            },
            "inference": {
                "patch": PATCH,
                "stride": 64,
                "roles": ["validation", "test"],
                "containment": "every window lies wholly in its requested role partition",
                "coverage": "every role-core pixel covered at least once; no outside coverage",
            },
            "folds": summary_rows,
        },
    )
    write_json(
        output_dir / "loader_source_integrity_v3.json",
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
        },
    )
    overall_accepted = all(item["passed"] for item in checks)
    write_json(
        output_dir / "stage4_acceptance_v3.json",
        {
            "overall_accepted": overall_accepted,
            "status": (
                "accepted_for_training_protocol_freeze"
                if overall_accepted else "rejected"
            ),
            "next_gate": (
                "Freeze the fair multi-model training protocol, seed schedule, stopping rule, and "
                "output schema before launching any training jobs."
            ),
            "checks": checks,
        },
    )
    (output_dir / "README.md").write_text(
        "\n".join(
            [
                "# V3 loader integration audit — stage 4",
                "",
                f"Status: {'accepted' if overall_accepted else 'rejected'}.",
                "",
                "The audit loads all four folds through `dataset_common_v3.py`, verifies training-only",
                "normalization, tests deterministic and synchronized patch augmentation, and proves",
                "validation/test sliding-window containment and full role-core coverage.",
                "",
                "No model, optimizer, checkpoint, or training result is created in this stage.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    checksum_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name not in {
            "loader_checksums_v3.sha256",
            "independent_verification_v3.json",
        }:
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "loader_checksums_v3.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "overall_accepted": overall_accepted,
                "folds": len(fold_ids),
                "checks": len(checks),
                "failed_checks": sum(not item["passed"] for item in checks),
                "models_created": 0,
                "next_gate": "training protocol freeze; do not train yet",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not overall_accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
