#!/usr/bin/env python3
"""Independently verify V3 stage-2 fold labels and masks.

This verifier does not modify source data or the V3 dataset. It does not import
the stage-2 generator and recomputes the delivery checks from the saved arrays,
rasters, frozen manifest, source data, and QFFD vector. An optional JSON audit
report may be written outside the dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from scipy import ndimage


ROLE_CODES = {"train": 1, "validation": 2, "test": 3}
ROLE_FILES = {"train": "train.npy", "validation": "val.npy", "test": "test.npy"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_mask(shape: tuple[int, int], ranges: list[list[int]]) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    for start, end in ranges:
        result[start:end] = True
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    dataset = (args.dataset_dir or root / "01_data/interim/dataset_v3").resolve()
    manifest_path = root / "04_results/split_design_v3/fold_manifest_v3.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    acceptance = json.loads(
        (dataset / "stage2_acceptance_v3.json").read_text(encoding="utf-8")
    )
    integrity = json.loads(
        (dataset / "source_integrity_v3.json").read_text(encoding="utf-8")
    )
    shape = tuple(manifest["grid"]["shape"])
    y_v2_path = root / "01_data/interim/dataset_v2/y.npy"
    valid_path = root / "01_data/interim/dataset_v2/valid.npy"
    y_v2 = np.load(y_v2_path)
    valid = np.load(valid_path).astype(bool)

    failures: list[str] = []
    check_count = 0

    def check(condition: bool, description: str) -> None:
        nonlocal check_count
        check_count += 1
        if not bool(condition):
            failures.append(description)

    check(acceptance["overall_accepted"], "saved stage2 acceptance is false")
    check(
        all(item["passed"] for item in acceptance["checks"]),
        "saved stage2 acceptance contains a failed check",
    )
    check(integrity["all_sources_unchanged"], "source-integrity record says sources changed")
    check(
        integrity["all_stage1_frozen_inputs_match"],
        "source-integrity record does not match stage1 inputs",
    )

    source_map = {
        "X": root / "01_data/interim/dataset_v2/X.npy",
        "y": y_v2_path,
        "valid": valid_path,
        "vector": root / "01_data/interim/nm_rifft_concealed.gpkg",
        "reference": root / "01_data/interim/label_inferred_fault_v2.tif",
        "manifest": manifest_path,
    }
    for name, path in source_map.items():
        actual_hash = sha256(path)
        recorded = integrity["source_files"][name]
        check(actual_hash == recorded["sha256_before"], f"{name} differs from before hash")
        check(actual_hash == recorded["sha256_after"], f"{name} differs from after hash")
        if name in manifest["input_sha256"]:
            check(
                actual_hash == manifest["input_sha256"][name],
                f"{name} differs from frozen stage1 hash",
            )

    checksum_entries = {}
    for line in (dataset / "file_checksums_v3.sha256").read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_entries[relative] = digest
    actual_files = {
        path.relative_to(dataset).as_posix()
        for path in dataset.rglob("*")
        if path.is_file() and path.name != "file_checksums_v3.sha256"
    }
    check(set(checksum_entries) == actual_files, "checksum inventory differs from output files")
    for relative, expected_hash in checksum_entries.items():
        check(sha256(dataset / relative) == expected_hash, f"checksum mismatch: {relative}")

    reference_path = source_map["reference"]
    with rasterio.open(reference_path) as ref:
        reference_transform = ref.transform
        reference_crs = ref.crs
        reference_bounds = ref.bounds
    pixel_size_m = abs(float(reference_transform.a))
    ignore_distance_m = float(manifest["label_policy"]["ignore_distance_m"])

    faults = gpd.read_file(source_map["vector"]).to_crs(reference_crs)
    faults = faults[faults["linetype"].astype(str).str.casefold().eq("inferred")].copy()
    faults["fault_id_text"] = faults["fault_id"].astype(str)

    core_union = row_mask(shape, list(manifest["selected_blocks"].values()))
    buffer_union = row_mask(shape, list(manifest["excluded_buffers"].values()))
    check(not np.any(core_union & buffer_union), "core and buffer masks overlap")
    check(np.all(core_union | buffer_union), "core and buffer masks do not cover all rows")
    check(
        all(end - start == 50 for start, end in manifest["excluded_buffers"].values()),
        "a frozen buffer is not 50 rows",
    )
    check(np.isclose(50 * pixel_size_m, 5000.0), "50 rows are not 5 km")

    role_count_rows = 0
    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        fold_dir = dataset / fold_id
        label = np.load(fold_dir / "y.npy")
        partition = np.load(fold_dir / "partition.npy")
        masks = {role: np.load(fold_dir / filename) for role, filename in ROLE_FILES.items()}

        check(label.shape == shape and label.dtype == np.uint8, f"{fold_id} label shape/dtype")
        check(
            partition.shape == shape and partition.dtype == np.uint8,
            f"{fold_id} partition shape/dtype",
        )
        check(set(np.unique(label)).issubset({0, 1, 2}), f"{fold_id} unexpected labels")
        check(
            set(np.unique(partition)).issubset({0, 1, 2, 3}),
            f"{fold_id} unexpected partition codes",
        )

        for filename, expected in (("label.tif", label), ("partition.tif", partition)):
            with rasterio.open(fold_dir / filename) as raster:
                check(raster.shape == shape, f"{fold_id}/{filename} shape mismatch")
                check(raster.transform == reference_transform, f"{fold_id}/{filename} transform")
                check(raster.crs == reference_crs, f"{fold_id}/{filename} CRS")
                check(
                    np.array_equal(raster.read(1), expected),
                    f"{fold_id}/{filename} differs from NPY",
                )

        check(
            not np.any(masks["train"] & masks["validation"])
            and not np.any(masks["train"] & masks["test"])
            and not np.any(masks["validation"] & masks["test"]),
            f"{fold_id} eligible masks overlap",
        )
        check(
            np.all(partition[core_union] != 0), f"{fold_id} unassigned core pixels"
        )
        check(
            np.all(partition[buffer_union] == 0)
            and np.all(label[buffer_union] == 2)
            and all(not mask[buffer_union].any() for mask in masks.values()),
            f"{fold_id} buffer exclusion failure",
        )

        allowed_sets = {
            role: set(fold["roles"][role]["allowed_fault_ids"])
            for role in ROLE_CODES
        }
        check(
            not (allowed_sets["train"] & allowed_sets["validation"])
            and not (allowed_sets["train"] & allowed_sets["test"])
            and not (allowed_sets["validation"] & allowed_sets["test"]),
            f"{fold_id} allowed fault IDs overlap",
        )

        for role, code in ROLE_CODES.items():
            role_count_rows += 1
            record = fold["roles"][role]
            role_core = row_mask(shape, record["row_ranges"])
            expected_mask = (partition == code) & valid & (label != 2)
            check(np.array_equal(masks[role], expected_mask), f"{fold_id}/{role} mask rule")
            actual_counts = {
                "valid_pixels": int((role_core & valid).sum()),
                "evaluation_pixels": int(expected_mask.sum()),
                "positive_pixels": int((expected_mask & (label == 1)).sum()),
                "negative_pixels": int((expected_mask & (label == 0)).sum()),
                "ignored_valid_pixels": int((role_core & valid & (label == 2)).sum()),
            }
            check(
                all(actual_counts[key] == int(record["counts"][key]) for key in actual_counts),
                f"{fold_id}/{role} counts differ from manifest",
            )
            if role == "test":
                check(
                    np.array_equal(label[role_core], y_v2[role_core]),
                    f"{fold_id} test labels differ from V2",
                )

            purged_ids = set(record["purged_fault_ids"])
            if purged_ids:
                purged = faults[faults["fault_id_text"].isin(purged_ids)]
                trace = rasterize(
                    [(geometry, 1) for geometry in purged.geometry],
                    out_shape=shape,
                    transform=reference_transform,
                    fill=0,
                    all_touched=True,
                    dtype="uint8",
                )
                distance = ndimage.distance_transform_edt(trace == 0) * pixel_size_m
                check(
                    np.all(label[role_core & (distance <= ignore_distance_m)] == 2),
                    f"{fold_id}/{role} purged influence is not entirely ignored",
                )

    check(role_count_rows == 12, "expected 12 fold-role count records")
    check(
        reference_bounds is not None and shape == y_v2.shape == valid.shape,
        "source arrays do not match reference grid",
    )

    result = {
        "independent_verification_passed": not failures,
        "checks": check_count,
        "failed_checks": len(failures),
        "failures": failures,
        "folds": len(manifest["folds"]),
        "fold_role_records": role_count_rows,
    }
    if args.report:
        report_path = args.report.resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
