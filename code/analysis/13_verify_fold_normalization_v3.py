#!/usr/bin/env python3
"""Independently verify V3 stage-3 normalization statistics.

The verifier is read-only with respect to V2, dataset_v3, and normalization_v3.
It recomputes moments with a numerically stable parallel-variance merge across
row chunks, rather than calling the fitting implementation. An optional JSON
report may be written outside the normalization dataset.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--normalization-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    norm_dir = (
        args.normalization_dir or root / "01_data/interim/normalization_v3"
    ).resolve()
    dataset_v2 = root / "01_data/interim/dataset_v2"
    dataset_v3 = root / "01_data/interim/dataset_v3"
    manifest = json.loads(
        (root / "04_results/split_design_v3/fold_manifest_v3.json").read_text(
            encoding="utf-8"
        )
    )
    acceptance = json.loads(
        (norm_dir / "stage3_acceptance_v3.json").read_text(encoding="utf-8")
    )
    integrity = json.loads(
        (norm_dir / "normalization_source_integrity_v3.json").read_text(
            encoding="utf-8"
        )
    )
    norm_manifest = json.loads(
        (norm_dir / "normalization_manifest_v3.json").read_text(encoding="utf-8")
    )
    meta = json.loads((dataset_v2 / "meta.json").read_text(encoding="utf-8"))
    X = np.load(dataset_v2 / "X.npy", mmap_mode="r")
    valid = np.load(dataset_v2 / "valid.npy", mmap_mode="r").astype(bool)
    channel_names = [
        f"{band}_{season}" for band in meta["bands"] for season in meta["seasons"]
    ]

    failures: list[str] = []
    check_count = 0

    def check(condition: bool, description: str) -> None:
        nonlocal check_count
        check_count += 1
        if not bool(condition):
            failures.append(description)

    check(acceptance["overall_accepted"], "saved stage3 acceptance is false")
    check(
        all(item["passed"] for item in acceptance["checks"]),
        "saved stage3 acceptance contains a failed check",
    )
    check(integrity["all_sources_unchanged"], "saved source-integrity status is false")
    check(
        integrity["X_matches_stage2_accepted_hash"],
        "X hash did not match stage2 in saved integrity record",
    )
    check(list(X.shape) == norm_manifest["feature_shape"], "feature shape mismatch")
    check(str(X.dtype) == norm_manifest["feature_dtype"], "feature dtype mismatch")
    check(channel_names == norm_manifest["channel_names"], "channel order mismatch")

    checksum_entries: dict[str, str] = {}
    checksum_path = norm_dir / "normalization_checksums_v3.sha256"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_entries[relative] = digest
    actual_output_files = {
        path.relative_to(norm_dir).as_posix()
        for path in norm_dir.rglob("*")
        if path.is_file() and path.name != checksum_path.name
    }
    check(
        set(checksum_entries) == actual_output_files,
        "normalization checksum inventory differs from output files",
    )
    for relative, expected in checksum_entries.items():
        check(sha256(norm_dir / relative) == expected, f"checksum mismatch: {relative}")

    for name, record in integrity["source_files"].items():
        path = root / record["path"]
        actual = sha256(path)
        check(actual == record["sha256_before"], f"source before hash mismatch: {name}")
        check(actual == record["sha256_after"], f"source after hash mismatch: {name}")

    with (norm_dir / "normalization_stats_v3.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        summary_rows = list(csv.DictReader(handle))
    check(len(summary_rows) == len(manifest["folds"]) * X.shape[2], "summary row count")
    summary_lookup = {
        (row["fold_id"], int(row["channel_index"])): row for row in summary_rows
    }

    fold_manifest_lookup = {record["fold_id"]: record for record in norm_manifest["folds"]}
    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        fold_data = dataset_v3 / fold_id
        train_path = fold_data / "train.npy"
        train = np.load(train_path, mmap_mode="r")
        val = np.load(fold_data / "val.npy", mmap_mode="r")
        test = np.load(fold_data / "test.npy", mmap_mode="r")
        y = np.load(fold_data / "y.npy", mmap_mode="r")
        partition = np.load(fold_data / "partition.npy", mmap_mode="r")
        stats_path = norm_dir / fold_id / "norm_stats.json"
        stats = json.loads(stats_path.read_text(encoding="utf-8"))

        check(stats["fold_id"] == fold_id, f"{fold_id} identifier mismatch")
        check(stats["channel_count"] == X.shape[2], f"{fold_id} channel count")
        check(stats["channel_names"] == channel_names, f"{fold_id} channel names")
        check(stats["accumulator_dtype"] == "float64", f"{fold_id} accumulator dtype")
        check("ddof=0" in stats["standard_deviation"], f"{fold_id} ddof")
        check(
            np.array_equal(train, (partition == 1) & valid & (y != 2)),
            f"{fold_id} train mask rule mismatch",
        )
        check(
            not np.any(train & val)
            and not np.any(train & test)
            and not np.any(val & test),
            f"{fold_id} fold masks overlap",
        )
        n = int(train.sum())
        check(n == stats["sample_count"], f"{fold_id} sample count in norm stats")
        check(
            n == int(fold["roles"]["train"]["counts"]["evaluation_pixels"]),
            f"{fold_id} sample count in frozen manifest",
        )
        check(
            sha256(train_path) == stats["input_sha256"]["train_mask"],
            f"{fold_id} train-mask input hash",
        )
        check(
            sha256(dataset_v2 / "X.npy") == stats["input_sha256"]["X"],
            f"{fold_id} X input hash",
        )
        record = fold_manifest_lookup[fold_id]
        check(sha256(stats_path) == record["stats_sha256"], f"{fold_id} stats hash")
        check(record["sample_count"] == n, f"{fold_id} manifest sample count")

        merged_mean = np.zeros(X.shape[2], dtype=np.float64)
        merged_m2 = np.zeros(X.shape[2], dtype=np.float64)
        minima = np.full(X.shape[2], np.inf, dtype=np.float64)
        maxima = np.full(X.shape[2], -np.inf, dtype=np.float64)
        finite_counts = np.zeros(X.shape[2], dtype=np.int64)
        accumulated_n = 0
        for row_start in range(0, X.shape[0], 128):
            row_end = min(row_start + 128, X.shape[0])
            chunk_mask = train[row_start:row_end]
            if not chunk_mask.any():
                continue
            values = np.asarray(X[row_start:row_end][chunk_mask], dtype=np.float64)
            chunk_n = values.shape[0]
            finite = np.isfinite(values)
            finite_counts += finite.sum(axis=0)
            chunk_mean = values.mean(axis=0, dtype=np.float64)
            chunk_m2 = np.square(values - chunk_mean).sum(axis=0, dtype=np.float64)
            combined_n = accumulated_n + chunk_n
            delta = chunk_mean - merged_mean
            merged_m2 += (
                chunk_m2
                + np.square(delta) * accumulated_n * chunk_n / combined_n
            )
            merged_mean += delta * chunk_n / combined_n
            accumulated_n = combined_n
            minima = np.minimum(minima, values.min(axis=0))
            maxima = np.maximum(maxima, values.max(axis=0))
        check(accumulated_n == n, f"{fold_id} chunked sample count")

        recomputed_mean = merged_mean
        recomputed_std = np.sqrt(np.maximum(merged_m2 / n, 0.0))
        saved_mean = np.asarray(stats["mean"], dtype=np.float64)
        saved_std = np.asarray(stats["std"], dtype=np.float64)
        saved_min = np.asarray(stats["min"], dtype=np.float64)
        saved_max = np.asarray(stats["max"], dtype=np.float64)
        saved_finite = np.asarray(stats["finite_count"], dtype=np.int64)
        normalized_mean = (recomputed_mean - saved_mean) / saved_std
        normalized_std = recomputed_std / saved_std
        check(
            np.max(np.abs(normalized_mean)) <= 1e-8,
            f"{fold_id} normalized means are not approximately zero",
        )
        check(
            np.max(np.abs(normalized_std - 1.0)) <= 1e-8,
            f"{fold_id} normalized stds are not approximately one",
        )

        for index, channel_name in enumerate(channel_names):
            prefix = f"{fold_id}/{index}:{channel_name}"
            check(
                finite_counts[index] == n == saved_finite[index],
                f"{prefix} finite count",
            )
            check(
                np.isclose(recomputed_mean[index], saved_mean[index], rtol=1e-12, atol=1e-10),
                f"{prefix} mean",
            )
            check(
                np.isclose(recomputed_std[index], saved_std[index], rtol=1e-10, atol=1e-10),
                f"{prefix} std",
            )
            check(minima[index] == saved_min[index], f"{prefix} minimum")
            check(maxima[index] == saved_max[index], f"{prefix} maximum")
            summary = summary_lookup[(fold_id, index)]
            check(int(summary["sample_count"]) == n, f"{prefix} CSV sample count")
            check(
                np.isclose(float(summary["mean"]), saved_mean[index], rtol=0.0, atol=0.0),
                f"{prefix} CSV mean",
            )
            check(
                np.isclose(float(summary["std_ddof0"]), saved_std[index], rtol=0.0, atol=0.0),
                f"{prefix} CSV std",
            )

    result = {
        "independent_verification_passed": not failures,
        "checks": check_count,
        "failed_checks": len(failures),
        "failures": failures,
        "folds": len(manifest["folds"]),
        "channels_per_fold": X.shape[2],
        "recomputation": "row-chunk parallel variance merge in float64",
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
