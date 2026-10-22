#!/usr/bin/env python3
"""Independently verify the V3 stage-4 loader audit.

This verifier does not import or execute ``dataset_common_v3``. It reconstructs
normalization, eligible patch centers, and inference coverage directly from the
frozen manifests and arrays, then compares those results with the saved audit.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


ROLE_CODES = {"train": 1, "validation": 2, "test": 3}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def axis_starts(start: int, end: int, patch: int, stride: int) -> list[int]:
    starts = list(range(start, end - patch + 1, stride))
    if not starts or starts[-1] != end - patch:
        starts.append(end - patch)
    return starts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    audit_dir = (args.audit_dir or root / "04_results/loader_audit_v3").resolve()
    manifest = json.loads(
        (root / "04_results/split_design_v3/fold_manifest_v3.json").read_text(
            encoding="utf-8"
        )
    )
    acceptance = json.loads(
        (audit_dir / "stage4_acceptance_v3.json").read_text(encoding="utf-8")
    )
    contract = json.loads(
        (audit_dir / "loader_contract_v3.json").read_text(encoding="utf-8")
    )
    integrity = json.loads(
        (audit_dir / "loader_source_integrity_v3.json").read_text(encoding="utf-8")
    )
    with (audit_dir / "loader_audit_summary_v3.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    summary = {row["fold_id"]: row for row in rows}
    contract_folds = {row["fold_id"]: row for row in contract["folds"]}

    failures: list[str] = []
    check_count = 0

    def check(condition: bool, description: str) -> None:
        nonlocal check_count
        check_count += 1
        if not bool(condition):
            failures.append(description)

    check(acceptance["overall_accepted"], "saved stage4 acceptance is false")
    check(
        all(item["passed"] for item in acceptance["checks"]),
        "saved stage4 acceptance contains a failed check",
    )
    check(integrity["all_sources_unchanged"], "saved source integrity is false")
    check(len(manifest["folds"]) == 4, "frozen manifest does not contain four folds")
    check(len(rows) == 4, "loader audit summary does not contain four rows")

    checksum_entries: dict[str, str] = {}
    checksum_file = audit_dir / "loader_checksums_v3.sha256"
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_entries[relative] = digest
    actual_files = {
        path.relative_to(audit_dir).as_posix()
        for path in audit_dir.rglob("*")
        if path.is_file()
        and path.name not in {
            "loader_checksums_v3.sha256",
            "independent_verification_v3.json",
        }
    }
    check(set(checksum_entries) == actual_files, "loader checksum inventory mismatch")
    for relative, expected in checksum_entries.items():
        check(sha256(audit_dir / relative) == expected, f"checksum mismatch: {relative}")

    for name, record in integrity["source_files"].items():
        source_path = root / record["path"]
        actual = sha256(source_path)
        check(actual == record["sha256_before"], f"source before hash mismatch: {name}")
        check(actual == record["sha256_after"], f"source after hash mismatch: {name}")

    loader_path = root / contract["loader"]
    check(sha256(loader_path) == contract["loader_sha256"], "loader contract hash mismatch")
    loader_tree = ast.parse(loader_path.read_text(encoding="utf-8"))
    function_names = {
        node.name for node in loader_tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }
    for required_name in (
        "load_fold",
        "FoldTrainPatchDataset",
        "inference_window_starts",
        "inference_coverage",
        "sliding_predict_fold",
    ):
        check(required_name in function_names, f"loader is missing {required_name}")
    load_fold_node = next(
        node for node in loader_tree.body if isinstance(node, ast.FunctionDef) and node.name == "load_fold"
    )
    positional_names = [argument.arg for argument in load_fold_node.args.args]
    check(positional_names == ["fold_id"], "load_fold does not require explicit fold_id")
    loader_text = loader_path.read_text(encoding="utf-8")
    check("torch.optim" not in loader_text, "loader unexpectedly references an optimizer")
    check("dataset_v2/norm_stats.json" not in loader_text, "loader references V2 normalization")

    X = np.load(root / "01_data/interim/dataset_v2/X.npy", mmap_mode="r")
    valid = np.load(root / "01_data/interim/dataset_v2/valid.npy", mmap_mode="r").astype(bool)
    patch = int(contract["training_patch"]["size"])
    stride = int(contract["inference"]["stride"])
    half = patch // 2
    height, width = valid.shape

    for fold in manifest["folds"]:
        fold_id = fold["fold_id"]
        row = summary[fold_id]
        contract_row = contract_folds[fold_id]
        fold_dir = root / "01_data/interim/dataset_v3" / fold_id
        y = np.load(fold_dir / "y.npy", mmap_mode="r")
        partition = np.load(fold_dir / "partition.npy", mmap_mode="r")
        train = np.load(fold_dir / "train.npy", mmap_mode="r")
        validation = np.load(fold_dir / "val.npy", mmap_mode="r")
        test = np.load(fold_dir / "test.npy", mmap_mode="r")
        norm = json.loads(
            (
                root
                / "01_data/interim/normalization_v3"
                / fold_id
                / "norm_stats.json"
            ).read_text(encoding="utf-8")
        )

        check(
            np.array_equal(train, (partition == 1) & valid & (y != 2)),
            f"{fold_id} train mask rule",
        )
        check(
            np.array_equal(validation, (partition == 2) & valid & (y != 2)),
            f"{fold_id} validation mask rule",
        )
        check(
            np.array_equal(test, (partition == 3) & valid & (y != 2)),
            f"{fold_id} test mask rule",
        )
        training_pixels = int(train.sum())
        positive = int((train & (y == 1)).sum())
        negative = int((train & (y == 0)).sum())
        pos_weight = negative / positive
        for saved in (row, contract_row):
            check(int(saved["training_pixels"]) == training_pixels, f"{fold_id} training pixels")
            check(
                int(saved["positive_training_pixels"]) == positive,
                f"{fold_id} positive training pixels",
            )
            check(
                int(saved["negative_training_pixels"]) == negative,
                f"{fold_id} negative training pixels",
            )
            check(
                float(saved["pos_weight"]) == pos_weight,
                f"{fold_id} positive weight",
            )

        mean = np.asarray(norm["mean"], dtype=np.float32)
        std = np.asarray(norm["std"], dtype=np.float32)
        values = np.asarray(X[train], dtype=np.float32)
        normalized = np.empty_like(values)
        np.subtract(values, mean, out=normalized)
        np.divide(normalized, std, out=normalized)
        max_abs_mean = float(
            np.max(np.abs(normalized.mean(axis=0, dtype=np.float64)))
        )
        max_abs_std_error = float(
            np.max(np.abs(normalized.std(axis=0, ddof=0, dtype=np.float64) - 1.0))
        )
        check(
            np.isclose(
                float(row["max_abs_normalized_train_mean"]),
                max_abs_mean,
                rtol=0.0,
                atol=0.0,
            ),
            f"{fold_id} normalized mean audit value",
        )
        check(
            np.isclose(
                float(row["max_abs_normalized_train_std_error"]),
                max_abs_std_error,
                rtol=0.0,
                atol=0.0,
            ),
            f"{fold_id} normalized std audit value",
        )
        check(max_abs_mean <= 5e-5, f"{fold_id} normalized mean tolerance")
        check(max_abs_std_error <= 5e-5, f"{fold_id} normalized std tolerance")

        center_mask = np.zeros((height, width), dtype=bool)
        for start, end in fold["roles"]["train"]["row_ranges"]:
            center_mask[start + half:end - half + 1, half:width - half + 1] = True
        expected_complete = sum(
            (end - start - patch + 1) * (width - patch + 1)
            for start, end in fold["roles"]["train"]["row_ranges"]
        )
        positive_centers = int((center_mask & train & (y == 1)).sum())
        background_centers = int((center_mask & train & (y == 0)).sum())
        for saved in (row, contract_row):
            check(
                int(saved["complete_training_centers"]) == expected_complete,
                f"{fold_id} complete center count",
            )
            check(
                int(saved["positive_training_centers"]) == positive_centers,
                f"{fold_id} positive center count",
            )
            check(
                int(saved["background_training_centers"]) == background_centers,
                f"{fold_id} background center count",
            )
        check(positive_centers > 0 and background_centers > 0, f"{fold_id} empty center pool")

        for role, mask_name in (("validation", "val"), ("test", "test")):
            coverage = np.zeros((height, width), dtype=np.uint16)
            windows = []
            x_starts = axis_starts(0, width, patch, stride)
            for start, end in fold["roles"][role]["row_ranges"]:
                for y0 in axis_starts(start, end, patch, stride):
                    for x0 in x_starts:
                        windows.append((y0, x0))
                        coverage[y0:y0 + patch, x0:x0 + patch] += 1
                        check(
                            np.all(
                                partition[y0:y0 + patch, x0:x0 + patch]
                                == ROLE_CODES[role]
                            ),
                            f"{fold_id}/{role} window boundary at {(y0, x0)}",
                        )
            target = partition == ROLE_CODES[role]
            check(np.all(coverage[target] > 0), f"{fold_id}/{role} uncovered target")
            check(not np.any(coverage[~target] > 0), f"{fold_id}/{role} outside coverage")
            check(
                np.all((validation if mask_name == "val" else test) <= target),
                f"{fold_id}/{role} eligible mask outside role",
            )
            for saved in (row, contract_row):
                check(int(saved[f"{role}_windows"]) == len(windows), f"{fold_id}/{role} windows")
                check(
                    int(saved[f"{role}_min_coverage"]) == int(coverage[target].min()),
                    f"{fold_id}/{role} minimum coverage",
                )
                check(
                    int(saved[f"{role}_max_coverage"]) == int(coverage[target].max()),
                    f"{fold_id}/{role} maximum coverage",
                )
        del values, normalized

    f1 = summary["F1_NorthTest"]
    f4 = summary["F4_SouthTest"]
    for key in (
        "training_pixels",
        "positive_training_pixels",
        "negative_training_pixels",
        "pos_weight",
        "complete_training_centers",
        "positive_training_centers",
        "background_training_centers",
    ):
        check(f1[key] == f4[key], f"F1/F4 shared-training value differs: {key}")

    result = {
        "independent_verification_passed": not failures,
        "checks": check_count,
        "failed_checks": len(failures),
        "failures": failures,
        "folds": len(manifest["folds"]),
        "loader_imported": False,
        "models_created": 0,
    }
    report_path = (
        args.report.resolve()
        if args.report
        else audit_dir / "independent_verification_v3.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
