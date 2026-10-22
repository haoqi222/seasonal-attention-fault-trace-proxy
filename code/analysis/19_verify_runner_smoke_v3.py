#!/usr/bin/env python3
"""Independently verify V3 stage-6 runner smoke artifacts."""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import sys
from pathlib import Path

import joblib
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
    audit_dir = (args.audit_dir or root / "04_results/runner_smoke_v3").resolve()
    acceptance = json.loads(
        (audit_dir / "stage6_acceptance_v3.json").read_text(encoding="utf-8")
    )
    artifact_manifest = json.loads(
        (audit_dir / "smoke_artifact_manifest_v3.json").read_text(encoding="utf-8")
    )
    integrity = json.loads(
        (audit_dir / "smoke_source_integrity_v3.json").read_text(encoding="utf-8")
    )
    with (audit_dir / "smoke_summary_v3.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    summary = {row["smoke_id"]: row for row in rows}

    failures: list[str] = []
    check_count = 0

    def check(condition: bool, description: str) -> None:
        nonlocal check_count
        check_count += 1
        if not bool(condition):
            failures.append(description)

    check(acceptance["overall_accepted"], "saved stage6 acceptance is false")
    check(
        all(item["passed"] for item in acceptance["checks"]),
        "saved stage6 acceptance contains a failed check",
    )
    check(not acceptance["formal_training_started"], "acceptance says formal training started")
    check(integrity["all_sources_unchanged"], "saved smoke source integrity is false")
    check(
        artifact_manifest["formal_results_absent_before"]
        and artifact_manifest["formal_results_absent_after"],
        "artifact manifest reports formal results presence",
    )
    check(
        not (root / artifact_manifest["formal_results_root"]).exists(),
        "formal results_v3 exists during independent verification",
    )

    protocol_path = root / "03_configs/experiment_v3_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = sha256(protocol_path)
    check(protocol_hash == artifact_manifest["protocol_sha256"], "protocol hash mismatch")

    checksum_file = audit_dir / "runner_smoke_checksums_v3.sha256"
    checksum_entries = {}
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", 1)
        checksum_entries[relative] = digest
    actual_audit_files = {
        path.relative_to(audit_dir).as_posix()
        for path in audit_dir.rglob("*")
        if path.is_file()
        and path.name not in {
            "runner_smoke_checksums_v3.sha256",
            "independent_verification_v3.json",
        }
    }
    check(set(checksum_entries) == actual_audit_files, "audit checksum inventory mismatch")
    for relative, expected in checksum_entries.items():
        check(sha256(audit_dir / relative) == expected, f"audit checksum mismatch: {relative}")

    for name, record in integrity["source_files"].items():
        source_path = root / record["path"]
        actual = sha256(source_path)
        check(actual == record["sha256_before"], f"source before hash mismatch: {name}")
        check(actual == record["sha256_after"], f"source after hash mismatch: {name}")

    check(len(rows) == 9, "smoke summary does not contain nine cells")
    check(all(row["status"] == "passed" for row in rows), "a smoke row is not passed")
    check(
        {row["device"] for row in rows} >= {"cpu", "cuda:0", "cuda:1"},
        "CPU and both GPUs were not represented",
    )
    gpu_deep = {
        row["model_id"]
        for row in rows
        if row["model_kind"] == "deep"
        and row["device"].startswith("cuda")
        and row["batch_size"] == "16"
    }
    configured_deep = {entry["model_id"] for entry in protocol["deep_models"]}
    check(gpu_deep == configured_deep, "not all deep models passed GPU batch-16 smoke")
    check(
        {
            row["model_id"]
            for row in rows
            if row["model_kind"] == "random_forest"
        }
        == {entry["model_id"] for entry in protocol["random_forest_models"]},
        "RF smoke model set mismatch",
    )
    check(all(row["roundtrip_equal"] == "True" for row in rows), "round-trip flag false")
    check(
        all(row["no_validation_or_test_access"] == "True" for row in rows),
        "role-access flag false",
    )

    modeling_dir = root / "02_scripts/modeling"
    sys.path.insert(0, str(modeling_dir))
    from models_v3 import build_deep_model  # noqa: PLC0415

    manifest_lookup = {record["smoke_id"]: record for record in artifact_manifest["artifacts"]}
    check(set(manifest_lookup) == set(summary), "artifact and summary smoke IDs differ")
    required = set(protocol["output_schema"]["required_run_files"])
    for smoke_id, row in summary.items():
        record = manifest_lookup[smoke_id]
        run_dir = root / record["directory"]
        check(run_dir.is_dir(), f"missing artifact directory: {smoke_id}")
        actual_names = {path.name for path in run_dir.iterdir() if path.is_file()}
        kind_file = "checkpoint.pt" if row["model_kind"] == "deep" else "model.joblib"
        check(required <= actual_names, f"common schema missing: {smoke_id}")
        check(kind_file in actual_names, f"kind-specific schema missing: {smoke_id}")
        check(set(record["files"]) == actual_names, f"artifact file inventory: {smoke_id}")
        for filename, file_record in record["files"].items():
            path = run_dir / filename
            check(path.stat().st_size == file_record["bytes"], f"artifact size: {smoke_id}/{filename}")
            check(sha256(path) == file_record["sha256"], f"artifact hash: {smoke_id}/{filename}")

        validation_sentinel = np.load(run_dir / "validation_probability.npy")
        test_sentinel = np.load(run_dir / "test_probability.npy")
        check(
            validation_sentinel.shape == (1, 1)
            and validation_sentinel.dtype == np.float32
            and not validation_sentinel.any(),
            f"validation sentinel: {smoke_id}",
        )
        check(
            test_sentinel.shape == (1, 1)
            and test_sentinel.dtype == np.float32
            and not test_sentinel.any(),
            f"test sentinel: {smoke_id}",
        )
        curve = np.load(run_dir / "pr_curve.npz")
        check(
            curve["precision"].size == 0
            and curve["recall"].size == 0
            and curve["thresholds"].size == 0
            and bool(curve["smoke_no_role_access"][0]),
            f"PR smoke sentinel: {smoke_id}",
        )
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        run_manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        check(result["scope"] == "stage6_smoke_only", f"result scope: {smoke_id}")
        check(
            result["smoke_no_validation_or_test_access"]
            and not result["formal_training_started"],
            f"result access/training flags: {smoke_id}",
        )
        check(run_manifest["protocol_sha256"] == protocol_hash, f"manifest protocol: {smoke_id}")
        for filename, expected_hash in run_manifest["file_sha256"].items():
            check(sha256(run_dir / filename) == expected_hash, f"run manifest hash: {smoke_id}/{filename}")

        if row["model_kind"] == "deep":
            checkpoint = torch.load(
                run_dir / "checkpoint.pt", map_location="cpu", weights_only=False
            )
            check(checkpoint["scope"] == "stage6_smoke_only", f"checkpoint scope: {smoke_id}")
            check(checkpoint["protocol_sha256"] == protocol_hash, f"checkpoint protocol: {smoke_id}")
            check(checkpoint["smoke_no_validation_or_test_access"], f"checkpoint access: {smoke_id}")
            model = build_deep_model(row["model_id"])
            load_result = model.load_state_dict(checkpoint["model_state"])
            check(not load_result.missing_keys and not load_result.unexpected_keys, f"state keys: {smoke_id}")
            check(bool(checkpoint["optimizer_state"]["state"]), f"optimizer state empty: {smoke_id}")
            check(
                np.isfinite(float(row["loss"])) and np.isfinite(float(row["gradient_norm"])),
                f"loss or gradient non-finite: {smoke_id}",
            )
        else:
            classifier = joblib.load(run_dir / "model.joblib")
            expected_features = 32 if row["model_id"] == "rf_direct" else 56
            check(classifier.n_features_in_ == expected_features, f"RF feature count: {smoke_id}")
            check(len(classifier.estimators_) == 5, f"RF smoke trees: {smoke_id}")

    runner_path = root / "02_scripts/modeling/13_run_experiment_v3.py"
    runner_text = runner_path.read_text(encoding="utf-8")
    runner_tree = ast.parse(runner_text)
    check("--confirm-protocol-sha" in runner_text, "formal SHA confirmation option missing")
    check("planned_not_started" not in runner_text, "runner rewrites planned matrix status")
    formal_function = next(
        node
        for node in runner_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "execute_formal"
    )
    compare_nodes = [node for node in ast.walk(formal_function) if isinstance(node, ast.Compare)]
    check(bool(compare_nodes), "formal execution has no protocol comparison")
    smoke_script = (root / "02_scripts/analysis/18_smoke_test_runner_v3.py").read_text(
        encoding="utf-8"
    )
    check('data["val"]' not in smoke_script, "smoke orchestrator references validation mask")
    check('data["test"]' not in smoke_script, "smoke orchestrator references test mask")
    check(torch.cuda.is_available() and torch.cuda.device_count() == 2, "live two-GPU state")

    result = {
        "independent_verification_passed": not failures,
        "checks": check_count,
        "failed_checks": len(failures),
        "failures": failures,
        "smoke_cells": len(rows),
        "protocol_sha256": protocol_hash,
        "formal_training_started": False,
        "formal_results_present": (root / "01_data/interim/results_v3").exists(),
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
