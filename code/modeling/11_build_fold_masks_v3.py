#!/usr/bin/env python3
"""Build and verify the frozen V3 fold-specific labels and masks.

This is V3 experiment stage 2. It consumes the accepted stage-1 fold manifest
and the immutable V2 source data. It creates a new ``dataset_v3`` directory;
it never writes into ``dataset_v2`` and refuses to overwrite an existing V3
directory unless ``--overwrite`` is explicitly supplied.

No normalization statistics are fitted and no model is trained in this stage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
from shapely.geometry import box


PARTITION_CODES = {"excluded_or_buffer": 0, "train": 1, "validation": 2, "test": 3}
ROLE_MASK_FILES = {"train": "train.npy", "validation": "val.npy", "test": "test.npy"}


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


def rows_mask(shape: tuple[int, int], row_ranges: list[list[int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for start, end in row_ranges:
        if not (0 <= start < end <= shape[0]):
            raise ValueError(f"Invalid row range [{start}, {end}) for shape {shape}")
        if mask[start:end].any():
            raise ValueError(f"Overlapping row range [{start}, {end})")
        mask[start:end] = True
    return mask


def rasterize_ids(
    faults: gpd.GeoDataFrame,
    fault_ids: set[str],
    shape: tuple[int, int],
    transform: rasterio.Affine,
) -> np.ndarray:
    selected = faults[faults["fault_id_text"].isin(fault_ids)]
    if selected.empty:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize(
        [(geometry, 1) for geometry in selected.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=True,
        dtype="uint8",
    )


def write_raster(
    path: Path,
    array: np.ndarray,
    reference_profile: dict,
    *,
    description: str,
) -> None:
    profile = reference_profile.copy()
    profile.update(
        driver="GTiff",
        count=1,
        dtype="uint8",
        nodata=255,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **profile) as dataset:
        dataset.write(array.astype(np.uint8, copy=False), 1)
        dataset.set_band_description(1, description)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = args.project_root.resolve()
    manifest_path = (
        args.manifest or root / "04_results/split_design_v3/fold_manifest_v3.json"
    ).resolve()
    output_dir = (
        args.output_dir or root / "01_data/interim/dataset_v3"
    ).resolve()
    source_paths = {
        "X": root / "01_data/interim/dataset_v2/X.npy",
        "y": root / "01_data/interim/dataset_v2/y.npy",
        "valid": root / "01_data/interim/dataset_v2/valid.npy",
        "vector": root / "01_data/interim/nm_rifft_concealed.gpkg",
        "reference": root / "01_data/interim/label_inferred_fault_v2.tif",
        "manifest": manifest_path,
    }
    for name, path in source_paths.items():
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
    if manifest.get("status") != "accepted_for_stage_2_mask_generation":
        raise ValueError(f"Stage-1 manifest is not accepted: {manifest.get('status')}")

    hashes_before = {name: sha256(path) for name, path in source_paths.items()}
    frozen_hashes = manifest["input_sha256"]
    frozen_matches = {
        name: hashes_before[name] == frozen_hashes[name]
        for name in ("y", "valid", "vector", "reference")
    }
    if not all(frozen_matches.values()):
        raise ValueError(f"Frozen V2 input hash mismatch: {frozen_matches}")

    y_v2 = np.load(source_paths["y"])
    valid = np.load(source_paths["valid"]).astype(bool)
    shape = tuple(manifest["grid"]["shape"])
    if y_v2.shape != shape or valid.shape != shape:
        raise ValueError(f"Grid mismatch: manifest={shape}, y={y_v2.shape}, valid={valid.shape}")
    if set(np.unique(y_v2).tolist()) - {0, 1, 2}:
        raise ValueError(f"Unexpected V2 labels: {np.unique(y_v2)}")

    with rasterio.open(source_paths["reference"]) as reference:
        if reference.shape != shape:
            raise ValueError(f"Reference grid mismatch: {reference.shape} != {shape}")
        transform = reference.transform
        bounds = reference.bounds
        crs = reference.crs
        reference_profile = reference.profile.copy()
    pixel_size_m = abs(float(transform.a))
    if not np.isclose(pixel_size_m, manifest["grid"]["pixel_size_m"]):
        raise ValueError("Reference pixel size differs from the frozen manifest")

    positive_distance_m = float(manifest["label_policy"]["positive_distance_m"])
    ignore_distance_m = float(manifest["label_policy"]["ignore_distance_m"])
    faults = gpd.read_file(source_paths["vector"]).to_crs(crs)
    for field in ("fault_id", "linetype"):
        if field not in faults:
            raise ValueError(f"Vector is missing required field: {field}")
    faults = faults[faults["linetype"].astype(str).str.casefold().eq("inferred")].copy()
    faults = faults[
        faults.geometry.intersects(box(*bounds).buffer(ignore_distance_m))
    ].copy()
    faults["fault_id_text"] = faults["fault_id"].astype(str)
    all_fault_ids = set(faults["fault_id_text"])

    selected_blocks = manifest["selected_blocks"]
    excluded_buffers = manifest["excluded_buffers"]
    core_union = rows_mask(shape, list(selected_blocks.values()))
    buffer_union = rows_mask(shape, list(excluded_buffers.values()))
    if np.any(core_union & buffer_union) or not np.all(core_union | buffer_union):
        raise ValueError("Frozen core blocks and buffers do not partition every grid row")

    buffer_rows = int(manifest["grid"]["buffer_rows"])
    buffer_geometry_ok = all(
        end - start == buffer_rows for start, end in excluded_buffers.values()
    ) and all(
        list(selected_blocks.values())[i + 1][0] - list(selected_blocks.values())[i][1]
        == buffer_rows
        for i in range(len(selected_blocks) - 1)
    )

    label_cache: dict[frozenset[str], np.ndarray] = {}

    def label_for(allowed_ids: set[str]) -> np.ndarray:
        cache_key = frozenset(allowed_ids)
        if cache_key in label_cache:
            return label_cache[cache_key]
        if not allowed_ids:
            raise ValueError("A role has no allowed fault IDs")
        unknown = allowed_ids - all_fault_ids
        if unknown:
            raise ValueError(f"Manifest contains unknown fault IDs: {sorted(unknown)}")

        included_mask = rasterize_ids(faults, allowed_ids, shape, transform)
        included_distance = ndimage.distance_transform_edt(included_mask == 0) * pixel_size_m
        label = np.zeros(shape, dtype=np.uint8)
        label[
            (included_distance > positive_distance_m)
            & (included_distance <= ignore_distance_m)
        ] = 2
        label[included_distance <= positive_distance_m] = 1

        excluded_mask = rasterize_ids(faults, all_fault_ids - allowed_ids, shape, transform)
        if excluded_mask.any():
            excluded_distance = (
                ndimage.distance_transform_edt(excluded_mask == 0) * pixel_size_m
            )
            label[excluded_distance <= ignore_distance_m] = 2
        label_cache[cache_key] = label
        return label

    stats_rows: list[dict] = []
    id_records: list[dict] = []
    acceptance_checks: list[dict] = []
    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        fold_dir = output_dir / fold_id
        fold_dir.mkdir()
        fold_label = np.full(shape, 2, dtype=np.uint8)
        partition = np.zeros(shape, dtype=np.uint8)
        eligible_masks: dict[str, np.ndarray] = {}

        role_allowed = {
            role: set(fold["roles"][role]["allowed_fault_ids"])
            for role in ("train", "validation", "test")
        }
        ids_disjoint = all(
            not (role_allowed[left] & role_allowed[right])
            for left, right in (
                ("train", "validation"), ("train", "test"), ("validation", "test")
            )
        )
        acceptance_checks.append(
            {"fold": fold_id, "check": "allowed_fault_ids_disjoint", "passed": ids_disjoint}
        )

        for role in ("train", "validation", "test"):
            role_record = fold["roles"][role]
            role_core = rows_mask(shape, role_record["row_ranges"])
            if np.any(partition[role_core] != 0):
                raise ValueError(f"Overlapping role cores in {fold_id}: {role}")
            partition[role_core] = PARTITION_CODES[role]
            role_label = label_for(role_allowed[role])
            fold_label[role_core] = role_label[role_core]
            eligible = role_core & valid & (fold_label != 2)
            eligible_masks[role] = eligible

            actual = {
                "valid_pixels": int((role_core & valid).sum()),
                "evaluation_pixels": int(eligible.sum()),
                "positive_pixels": int((eligible & (fold_label == 1)).sum()),
                "negative_pixels": int((eligible & (fold_label == 0)).sum()),
                "ignored_valid_pixels": int((role_core & valid & (fold_label == 2)).sum()),
            }
            expected = role_record["counts"]
            count_matches = all(actual[key] == int(expected[key]) for key in actual)
            acceptance_checks.append(
                {
                    "fold": fold_id,
                    "role": role,
                    "check": "counts_match_stage1_manifest",
                    "passed": count_matches,
                    "actual": actual,
                    "expected": {key: int(expected[key]) for key in actual},
                }
            )

            purged_ids = set(role_record["purged_fault_ids"])
            purge_violation_count = 0
            if purged_ids:
                purged_trace = rasterize_ids(faults, purged_ids, shape, transform)
                purged_distance = (
                    ndimage.distance_transform_edt(purged_trace == 0) * pixel_size_m
                )
                purge_influence = role_core & (purged_distance <= ignore_distance_m)
                purge_violation_count = int((purge_influence & (fold_label != 2)).sum())
            acceptance_checks.append(
                {
                    "fold": fold_id,
                    "role": role,
                    "check": "purged_id_influence_is_ignore",
                    "passed": purge_violation_count == 0,
                    "violation_pixels": purge_violation_count,
                }
            )
            test_match = None
            if role == "test":
                test_match = bool(np.array_equal(fold_label[role_core], y_v2[role_core]))
                acceptance_checks.append(
                    {
                        "fold": fold_id,
                        "role": role,
                        "check": "test_label_matches_v2",
                        "passed": test_match,
                    }
                )

            stats_rows.append(
                {
                    "fold_id": fold_id,
                    "role": role,
                    "blocks": ";".join(role_record["blocks"]),
                    **actual,
                    "allowed_fault_ids": len(role_allowed[role]),
                    "purged_fault_ids": len(purged_ids),
                    "purge_violation_pixels": purge_violation_count,
                    "test_label_matches_v2": test_match,
                }
            )
            id_records.append(
                {
                    "fold_id": fold_id,
                    "role": role,
                    "blocks": role_record["blocks"],
                    "allowed_fault_ids": sorted(role_allowed[role]),
                    "purged_fault_ids": sorted(purged_ids),
                }
            )

        pairwise_disjoint = not (
            np.any(eligible_masks["train"] & eligible_masks["validation"])
            or np.any(eligible_masks["train"] & eligible_masks["test"])
            or np.any(eligible_masks["validation"] & eligible_masks["test"])
        )
        role_mask_exact = all(
            np.array_equal(
                eligible_masks[role],
                (partition == PARTITION_CODES[role]) & valid & (fold_label != 2),
            )
            for role in ("train", "validation", "test")
        )
        core_partition_ok = bool(np.all(partition[core_union] != 0))
        buffer_ok = bool(
            np.all(partition[buffer_union] == 0)
            and np.all(fold_label[buffer_union] == 2)
            and all(not mask[buffer_union].any() for mask in eligible_masks.values())
        )
        for check, passed in (
            ("eligible_masks_pairwise_disjoint", pairwise_disjoint),
            ("eligible_masks_match_partition_valid_label", role_mask_exact),
            ("every_core_pixel_has_one_partition", core_partition_ok),
            ("buffers_are_excluded_and_ignored", buffer_ok),
        ):
            acceptance_checks.append({"fold": fold_id, "check": check, "passed": passed})

        np.save(fold_dir / "y.npy", fold_label, allow_pickle=False)
        np.save(fold_dir / "partition.npy", partition, allow_pickle=False)
        for role, filename in ROLE_MASK_FILES.items():
            np.save(fold_dir / filename, eligible_masks[role], allow_pickle=False)
        write_raster(
            fold_dir / "label.tif", fold_label, reference_profile,
            description="V3 fold-specific label: 0 background, 1 positive, 2 ignore",
        )
        write_raster(
            fold_dir / "partition.tif", partition, reference_profile,
            description="V3 partition: 0 excluded/buffer, 1 train, 2 validation, 3 test",
        )
        write_json(
            fold_dir / "fold_stats.json",
            {
                "fold_id": fold_id,
                "partition_codes": PARTITION_CODES,
                "label_codes": {"background": 0, "positive": 1, "ignore": 2},
                "roles": [row for row in stats_rows if row["fold_id"] == fold_id],
            },
        )

    acceptance_checks.append(
        {
            "fold": "global",
            "check": "buffer_geometry_is_50_rows_5km",
            "passed": bool(
                buffer_geometry_ok
                and np.isclose(buffer_rows * pixel_size_m, 5000.0)
            ),
            "buffer_rows": buffer_rows,
            "buffer_distance_m": buffer_rows * pixel_size_m,
        }
    )
    write_csv(output_dir / "fold_mask_stats_v3.csv", stats_rows)
    write_json(
        output_dir / "fold_id_assignments_v3.json",
        {
            "policy": manifest["label_policy"]["purge"],
            "fold_roles": id_records,
        },
    )

    hashes_after = {name: sha256(path) for name, path in source_paths.items()}
    source_unchanged = hashes_before == hashes_after
    acceptance_checks.append(
        {
            "fold": "global",
            "check": "all_source_files_unchanged_during_stage2",
            "passed": source_unchanged,
        }
    )
    write_json(
        output_dir / "source_integrity_v3.json",
        {
            "source_files": {
                name: {
                    "path": str(path.relative_to(root)),
                    "sha256_before": hashes_before[name],
                    "sha256_after": hashes_after[name],
                    "unchanged": hashes_before[name] == hashes_after[name],
                    "matches_stage1_frozen_hash": frozen_matches.get(name),
                }
                for name, path in source_paths.items()
            },
            "all_sources_unchanged": source_unchanged,
            "all_stage1_frozen_inputs_match": all(frozen_matches.values()),
        },
    )

    overall_accepted = all(check["passed"] for check in acceptance_checks)
    write_json(
        output_dir / "stage2_acceptance_v3.json",
        {
            "overall_accepted": overall_accepted,
            "status": (
                "accepted_for_fold_specific_normalization"
                if overall_accepted else "rejected"
            ),
            "next_gate": (
                "Fit normalization statistics independently on each fold's eligible training pixels; "
                "do not train models until those statistics and their provenance pass review."
            ),
            "checks": acceptance_checks,
        },
    )

    readme_lines = [
        "# dataset_v3 — stage 2 fold labels and masks",
        "",
        f"Status: {'accepted' if overall_accepted else 'rejected'}.",
        "",
        "This directory was generated from the accepted stage-1 spatial-fold manifest. "
        "It does not contain fitted normalization statistics or model outputs.",
        "",
        "Each fold directory contains:",
        "",
        "- `y.npy` / `label.tif`: 0 background, 1 positive, 2 ignore.",
        "- `partition.npy` / `partition.tif`: 0 excluded/buffer, 1 train, 2 validation, 3 test.",
        "- `train.npy`, `val.npy`, `test.npy`: eligible boolean evaluation masks.",
        "- `fold_stats.json`: role counts and purge audit values.",
        "",
        "All three internal buffers are 50 rows on a 100 m grid (5 km). Test labels "
        "match the frozen V2 labels; held-out fault-ID influence zones are ignored in development roles.",
        "",
    ]
    (output_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")

    checksum_lines = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "file_checksums_v3.sha256":
            checksum_lines.append(f"{sha256(path)}  {path.relative_to(output_dir).as_posix()}")
    (output_dir / "file_checksums_v3.sha256").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "overall_accepted": overall_accepted,
                "folds": len(manifest["folds"]),
                "checks": len(acceptance_checks),
                "failed_checks": sum(not item["passed"] for item in acceptance_checks),
                "next_gate": "fold-specific normalization only",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    if not overall_accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
