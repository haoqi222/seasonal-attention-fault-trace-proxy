#!/usr/bin/env python3
"""Independently verify the locked V3 stage-5 protocol.

This verifier does not import the stage-5 audit implementation. It checks the
protocol hash, source hashes, capacity records, full Cartesian run matrix,
fold-derived sampler/weight values, environment snapshot, and the absence of
V3 training outputs.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from itertools import product
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    root = args.project_root.resolve()
    audit_dir = (
        args.audit_dir or root / "04_results/training_protocol_v3"
    ).resolve()
    protocol_path = root / "03_configs/experiment_v3_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    lock = json.loads((audit_dir / "protocol_lock_v3.json").read_text(encoding="utf-8"))
    acceptance = json.loads(
        (audit_dir / "stage5_acceptance_v3.json").read_text(encoding="utf-8")
    )
    integrity = json.loads(
        (audit_dir / "protocol_source_integrity_v3.json").read_text(encoding="utf-8")
    )
    environment = json.loads(
        (audit_dir / "environment_snapshot_v3.json").read_text(encoding="utf-8")
    )
    with (audit_dir / "model_capacity_v3.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        capacity_rows = list(csv.DictReader(handle))
    with (audit_dir / "run_matrix_v3.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        run_rows = list(csv.DictReader(handle))

    failures: list[str] = []
    check_count = 0

    def check(condition: bool, description: str) -> None:
        nonlocal check_count
        check_count += 1
        if not bool(condition):
            failures.append(description)

    check(acceptance["overall_accepted"], "saved stage5 acceptance is false")
    check(
        all(item["passed"] for item in acceptance["checks"]),
        "saved stage5 acceptance contains a failed check",
    )
    check(lock["status"] == "locked", "protocol lock is not locked")
    check(not lock["training_started"], "protocol lock says training started")
    check(integrity["all_sources_unchanged"], "saved source integrity is false")
    actual_protocol_hash = sha256(protocol_path)
    check(actual_protocol_hash == lock["protocol_sha256"], "protocol SHA-256 mismatch")
    check(
        lock["source_sha256"]["protocol"] == actual_protocol_hash,
        "locked protocol source hash mismatch",
    )

    checksum_file = audit_dir / "training_protocol_checksums_v3.sha256"
    checksum_entries = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_entries[relative] = digest
    actual_audit_files = {
        path.relative_to(audit_dir).as_posix()
        for path in audit_dir.rglob("*")
        if path.is_file()
        and path.name not in {
            "training_protocol_checksums_v3.sha256",
            "independent_verification_v3.json",
        }
    }
    check(set(checksum_entries) == actual_audit_files, "audit checksum inventory mismatch")
    for relative, expected in checksum_entries.items():
        check(sha256(audit_dir / relative) == expected, f"checksum mismatch: {relative}")

    for name, record in integrity["source_files"].items():
        source_path = root / record["path"]
        actual = sha256(source_path)
        check(actual == record["sha256_before"], f"source before hash mismatch: {name}")
        check(actual == record["sha256_after"], f"source after hash mismatch: {name}")
        check(actual == lock["source_sha256"][name], f"lock source hash mismatch: {name}")

    modeling_dir = root / "02_scripts/modeling"
    sys.path.insert(0, str(modeling_dir))
    from models_v3 import build_deep_model, trainable_parameters  # noqa: PLC0415

    configured_deep = {entry["model_id"]: entry for entry in protocol["deep_models"]}
    saved_capacity = {row["model_id"]: row for row in capacity_rows}
    check(set(saved_capacity) == set(configured_deep), "capacity model IDs mismatch")
    actual_counts = {}
    dummy = torch.zeros(1, 32, 128, 128)
    for model_id, entry in configured_deep.items():
        model = build_deep_model(model_id).eval()
        count = trainable_parameters(model)
        actual_counts[model_id] = count
        check(count == entry["expected_trainable_parameters"], f"{model_id} config parameters")
        check(
            count == int(saved_capacity[model_id]["trainable_parameters"]),
            f"{model_id} capacity CSV parameters",
        )
        with torch.no_grad():
            output = model(dummy)
        check(tuple(output.shape) == (1, 1, 128, 128), f"{model_id} forward shape")
        check(torch.isfinite(output).all(), f"{model_id} non-finite forward output")
        del model, output
    primary_ids = [
        model_id
        for model_id, entry in configured_deep.items()
        if entry["analysis_tier"] == "primary_capacity_matched"
    ]
    primary_ratio = max(actual_counts[key] for key in primary_ids) / min(
        actual_counts[key] for key in primary_ids
    )
    check(len(primary_ids) == 3, "primary capacity group does not contain three models")
    check(primary_ratio <= 1.5, "primary capacity ratio exceeds 1.5")
    check(
        np.isclose(primary_ratio, lock["model_capacity_ratio_primary"], rtol=0.0, atol=0.0),
        "locked primary capacity ratio mismatch",
    )

    models_tree = ast.parse((modeling_dir / "models_v3.py").read_text(encoding="utf-8"))
    called_attributes = {
        node.func.attr
        for node in ast.walk(models_tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    check("backward" not in called_attributes, "models_v3 contains a backward call")
    check("step" not in called_attributes, "models_v3 contains an optimizer-like step call")
    check("optim" not in (modeling_dir / "models_v3.py").read_text(encoding="utf-8"),
          "models_v3 references an optimizer")

    with (root / "04_results/loader_audit_v3/loader_audit_summary_v3.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        loader_rows = {row["fold_id"]: row for row in csv.DictReader(handle)}
    fractions = protocol["deep_training"]["sampler"]["fold_fault_center_fraction"]
    weights = protocol["deep_training"]["loss"]["fold_positive_weight"]
    for fold_id in protocol["data"]["folds"]:
        row = loader_rows[fold_id]
        pos_centers = int(row["positive_training_centers"])
        bg_centers = int(row["background_training_centers"])
        natural_fraction = pos_centers / (pos_centers + bg_centers)
        exact_weight = int(row["negative_training_pixels"]) / int(
            row["positive_training_pixels"]
        )
        check(
            np.isclose(fractions[fold_id], natural_fraction, rtol=0.0, atol=1e-15),
            f"{fold_id} sampler fraction",
        )
        check(
            np.isclose(weights[fold_id], exact_weight, rtol=0.0, atol=1e-15),
            f"{fold_id} positive weight",
        )

    model_entries = [
        (entry["model_id"], "deep", entry["analysis_tier"])
        for entry in protocol["deep_models"]
    ] + [
        (entry["model_id"], "random_forest", entry["analysis_tier"])
        for entry in protocol["random_forest_models"]
    ]
    expected_cells = {
        (model_id, kind, tier, fold_id, str(seed))
        for (model_id, kind, tier), fold_id, seed in product(
            model_entries, protocol["data"]["folds"], protocol["seeds"]
        )
    }
    actual_cells = {
        (
            row["model_id"],
            row["model_kind"],
            row["analysis_tier"],
            row["fold_id"],
            row["seed"],
        )
        for row in run_rows
    }
    check(actual_cells == expected_cells, "run matrix is not the full Cartesian product")
    check(len(run_rows) == 128, "run matrix does not contain 128 rows")
    check(
        len({row["run_id"] for row in run_rows}) == len(run_rows),
        "run IDs are not unique",
    )
    check(
        len({row["output_directory"] for row in run_rows}) == len(run_rows),
        "run output directories are not unique",
    )
    check(
        all(row["status"] == "planned_not_started" for row in run_rows),
        "a planned run is not marked planned_not_started",
    )
    check(
        all(actual_protocol_hash in row["output_directory"] for row in run_rows),
        "a run output is not bound to the protocol hash",
    )
    deep_runs = sum(row["model_kind"] == "deep" for row in run_rows)
    rf_runs = sum(row["model_kind"] == "random_forest" for row in run_rows)
    check(deep_runs == 96 and rf_runs == 32, "deep/RF run counts mismatch")
    check(lock["planned_runs"]["total"] == 128, "locked planned-run total mismatch")
    check(
        not (root / "01_data/interim/results_v3").exists(),
        "results_v3 exists before runner smoke testing",
    )

    deep = protocol["deep_training"]
    check(deep["maximum_epochs"] == 60, "maximum epochs changed")
    check(deep["minimum_epochs_before_early_stop"] == 15, "minimum epochs changed")
    check(deep["checkpoint"]["monitor"] == "validation_average_precision", "checkpoint metric")
    check(not deep["checkpoint"]["test_data_used"], "test used for checkpoint selection")
    check(not deep["primary_test_time_augmentation"], "primary TTA unexpectedly enabled")
    check(not protocol["evaluation"]["test_threshold_tuning"], "test threshold tuning enabled")
    check(
        protocol["random_forest_models"][1]["analysis_tier"]
        == "secondary_context_sensitivity",
        "RF context design is not secondary",
    )
    check(
        protocol["failure_policy"]["failed_seed_replacement"] == "prohibited",
        "failed seed replacement is not prohibited",
    )
    check(
        protocol["change_control"]["mixing_results_across_protocol_hashes"]
        == "prohibited",
        "cross-protocol result mixing is not prohibited",
    )

    check(environment["cuda_available"], "environment snapshot says CUDA unavailable")
    check(environment["torch_cuda_device_count"] == 2, "environment snapshot GPU count")
    check(len(environment["gpus"]) == 2, "environment snapshot GPU records")
    check(all("RTX 5090" in gpu["name"] for gpu in environment["gpus"]), "GPU model")
    check(environment["project_disk_free_bytes"] >= 500 * 1024 ** 3, "free disk below 500GB")
    check(torch.cuda.is_available() and torch.cuda.device_count() == 2, "live CUDA state")
    check(
        all("RTX 5090" in torch.cuda.get_device_name(index) for index in range(2)),
        "live GPU model",
    )

    result = {
        "independent_verification_passed": not failures,
        "checks": check_count,
        "failed_checks": len(failures),
        "failures": failures,
        "protocol_sha256": actual_protocol_hash,
        "planned_runs": len(run_rows),
        "training_started": False,
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
