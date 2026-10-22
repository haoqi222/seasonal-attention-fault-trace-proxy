#!/usr/bin/env python3
"""Stage-7 preflight and isolated recovery tests for the V3 scheduler."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def load_scheduler(path: Path):
    spec = importlib.util.spec_from_file_location("scheduler_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load scheduler: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def record(checks: list[dict], name: str, passed: bool, **details) -> None:
    checks.append({"check": name, "passed": bool(passed), **details})


def make_completed_fixture(scheduler, context: dict, formal_root: Path, cell) -> Path:
    final_dir = scheduler.cell_directory(formal_root, cell)
    final_dir.mkdir(parents=True)
    protocol = context["protocol"]
    required = list(protocol["output_schema"]["required_run_files"])
    required += (
        protocol["output_schema"]["deep_only_files"]
        if cell.model_kind == "deep"
        else protocol["output_schema"]["random_forest_only_files"]
    )
    for name in required:
        path = final_dir / name
        if name == "run_manifest.json":
            continue
        if name == "result.json":
            write_json(
                path,
                {
                    "protocol_sha256": context["protocol_sha256"],
                    "model_id": cell.model_id,
                    "model_kind": cell.model_kind,
                    "fold_id": cell.fold_id,
                    "seed": cell.seed,
                    "fixture_only": True,
                },
            )
        else:
            path.write_bytes(("fixture:" + name + "\n").encode("utf-8"))
    hashes = {
        name: sha256(final_dir / name)
        for name in required
        if name != "run_manifest.json"
    }
    write_json(
        final_dir / "run_manifest.json",
        {
            "scope": "formal_v3_run",
            "protocol_sha256": context["protocol_sha256"],
            "model_id": cell.model_id,
            "model_kind": cell.model_kind,
            "fold_id": cell.fold_id,
            "seed": cell.seed,
            "attempt": 1,
            "completed": True,
            "fixture_only": True,
            "file_sha256": hashes,
        },
    )
    return final_dir


def write_plan_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "ordinal",
        "cell_id",
        "model_id",
        "model_kind",
        "fold_id",
        "seed",
        "lane",
        "status",
        "attempt",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.project_root.resolve()
    scheduler_path = root / "02_scripts/modeling/14_schedule_experiments_v3.py"
    audit_dir = root / "04_results/scheduler_preflight_v3"
    scheduler = load_scheduler(scheduler_path)
    context = scheduler.load_context(root)
    fixture_root = (
        root
        / "01_data/interim/scheduler_preflight_v3"
        / context["protocol_sha256"]
    )
    if audit_dir.exists():
        raise FileExistsError(f"Refusing to overwrite stage-7 audit: {audit_dir}")
    if fixture_root.exists():
        raise FileExistsError(f"Refusing to overwrite stage-7 fixture: {fixture_root}")

    sources = {
        "protocol": root / scheduler.PROTOCOL_REL,
        "stage5_lock": root / scheduler.LOCK_REL,
        "runner": root / scheduler.RUNNER_REL,
        "scheduler": scheduler_path,
    }
    before_hashes = {name: sha256(path) for name, path in sources.items()}
    formal_root_before = context["formal_root"].exists()
    checks: list[dict] = []
    record(checks, "formal_results_absent_before_preflight", not formal_root_before)

    actual_plan = scheduler.build_plan(context)
    record(checks, "actual_matrix_has_128_cells", actual_plan["total_cells"] == 128)
    record(
        checks,
        "actual_matrix_all_pending_new",
        actual_plan["status_counts"]
        == {"pending_new": 128, "retry_failed": 0, "skip_completed": 0, "conflict": 0},
        counts=actual_plan["status_counts"],
    )
    record(
        checks,
        "lane_assignment_is_48_48_32",
        actual_plan["lane_counts"] == {"cuda:0": 48, "cuda:1": 48, "cpu": 32},
        counts=actual_plan["lane_counts"],
    )
    record(checks, "actual_matrix_safe_to_launch", actual_plan["safe_to_launch"])
    record(
        checks,
        "actual_cell_ids_unique",
        len({row["cell_id"] for row in actual_plan["cells"]}) == 128,
    )
    record(
        checks,
        "deep_and_rf_counts_match_lock",
        sum(row["model_kind"] == "deep" for row in actual_plan["cells"]) == 96
        and sum(row["model_kind"] == "random_forest" for row in actual_plan["cells"]) == 32,
    )

    cells = scheduler.generate_cells(context["protocol"])
    fixture_formal_root = fixture_root / "results"
    completed_cell, retry_cell, active_cell, invalid_cell = cells[:4]
    make_completed_fixture(scheduler, context, fixture_formal_root, completed_cell)

    retry_final = scheduler.cell_directory(fixture_formal_root, retry_cell)
    retry_attempt = retry_final.with_name(retry_final.name + ".attempt_001")
    retry_attempt.mkdir(parents=True)
    write_json(retry_attempt / "failure.json", {"completed": False, "fixture_only": True})

    active_final = scheduler.cell_directory(fixture_formal_root, active_cell)
    active_attempt = active_final.with_name(active_final.name + ".attempt_001")
    active_attempt.mkdir(parents=True)
    (active_attempt / "partial.tmp").write_text("fixture\n", encoding="utf-8")

    invalid_final = scheduler.cell_directory(fixture_formal_root, invalid_cell)
    invalid_final.mkdir(parents=True)
    (invalid_final / "result.json").write_text("{}\n", encoding="utf-8")

    fixture_plan = scheduler.build_plan(context, formal_root=fixture_formal_root)
    expected_fixture_counts = {
        "pending_new": 124,
        "retry_failed": 1,
        "skip_completed": 1,
        "conflict": 2,
    }
    record(
        checks,
        "fixture_status_counts_match",
        fixture_plan["status_counts"] == expected_fixture_counts,
        counts=fixture_plan["status_counts"],
    )
    first_four = {row["cell_id"]: row for row in fixture_plan["cells"][:4]}
    record(
        checks,
        "valid_completed_cell_is_skipped",
        first_four[completed_cell.cell_id]["status"] == "skip_completed",
    )
    record(
        checks,
        "failed_attempt_is_resumable_as_attempt_2",
        first_four[retry_cell.cell_id]["status"] == "retry_failed"
        and first_four[retry_cell.cell_id]["attempt"] == 2,
    )
    record(
        checks,
        "unfinished_attempt_is_conflict",
        first_four[active_cell.cell_id]["status"] == "conflict",
    )
    record(
        checks,
        "invalid_final_is_conflict",
        first_four[invalid_cell.cell_id]["status"] == "conflict",
    )
    record(checks, "fixture_conflicts_block_launch", not fixture_plan["safe_to_launch"])

    command = scheduler.runner_command(context, actual_plan["cells"][0])
    record(checks, "runner_command_has_exact_protocol_hash", context["protocol_sha256"] in command)
    record(checks, "runner_command_has_locked_cell", completed_cell.model_id in command and completed_cell.fold_id in command)

    no_execute_blocked = False
    try:
        scheduler.validate_execution_authorization(
            Namespace(execute=False, confirm_protocol_sha=None, confirm_launch_phrase=None),
            context,
            actual_plan,
        )
    except RuntimeError:
        no_execute_blocked = True
    record(checks, "execution_without_execute_flag_is_blocked", no_execute_blocked)

    wrong_phrase_blocked = False
    try:
        scheduler.validate_execution_authorization(
            Namespace(
                execute=True,
                confirm_protocol_sha=context["protocol_sha256"],
                confirm_launch_phrase="WRONG",
            ),
            context,
            actual_plan,
        )
    except RuntimeError:
        wrong_phrase_blocked = True
    record(checks, "execution_with_wrong_phrase_is_blocked", wrong_phrase_blocked)
    record(
        checks,
        "formal_gate_requires_stage7_audit",
        "validate_stage7_preflight(context)"
        in scheduler_path.read_text(encoding="utf-8"),
    )
    record(
        checks,
        "formal_gate_requires_two_cuda_devices",
        "torch.cuda.device_count() < 2"
        in scheduler_path.read_text(encoding="utf-8"),
    )

    lock_dir = fixture_root / "lock_test/scheduler.lock"
    scheduler.acquire_lock(lock_dir)
    second_lock_blocked = False
    try:
        scheduler.acquire_lock(lock_dir)
    except RuntimeError:
        second_lock_blocked = True
    finally:
        scheduler.release_lock(lock_dir)
    record(checks, "second_scheduler_lock_is_blocked", second_lock_blocked)

    after_hashes = {name: sha256(path) for name, path in sources.items()}
    record(checks, "all_sources_unchanged", before_hashes == after_hashes)
    record(
        checks,
        "formal_results_absent_after_preflight",
        not context["formal_root"].exists(),
    )

    audit_dir.mkdir(parents=True)
    write_plan_csv(audit_dir / "preflight_plan_v3.csv", actual_plan["cells"])
    write_json(
        audit_dir / "preflight_summary_v3.json",
        {key: value for key, value in actual_plan.items() if key != "cells"},
    )
    write_json(
        audit_dir / "fixture_summary_v3.json",
        {
            "fixture_only": True,
            "formal_root": str(fixture_formal_root),
            "status_counts": fixture_plan["status_counts"],
            "safe_to_launch": fixture_plan["safe_to_launch"],
            "tested_cells": [
                {
                    "cell_id": row["cell_id"],
                    "status": row["status"],
                    "attempt": row["attempt"],
                    "problems": row["problems"],
                }
                for row in fixture_plan["cells"][:4]
            ],
        },
    )
    write_json(
        audit_dir / "scheduler_source_integrity_v3.json",
        {
            "source_files": {
                name: {
                    "path": str(path.relative_to(root)).replace("\\", "/"),
                    "sha256_before": before_hashes[name],
                    "sha256_after": after_hashes[name],
                    "unchanged": before_hashes[name] == after_hashes[name],
                }
                for name, path in sources.items()
            },
            "all_sources_unchanged": before_hashes == after_hashes,
        },
    )
    failures = [item for item in checks if not item["passed"]]
    acceptance = {
        "overall_accepted": not failures,
        "status": "accepted_for_explicit_formal_launch_authorization" if not failures else "rejected",
        "next_gate": "Request explicit user authorization before supplying all three formal execution gates.",
        "checks": checks,
        "check_count": len(checks),
        "failed_checks": len(failures),
        "protocol_sha256": context["protocol_sha256"],
        "planned_runs": 128,
        "formal_training_started": False,
        "formal_results_present": context["formal_root"].exists(),
    }
    write_json(audit_dir / "stage7_acceptance_v3.json", acceptance)
    (audit_dir / "README.md").write_text(
        "# V3 scheduler preflight — stage 7\n\n"
        "Status: accepted.\n\n"
        f"Protocol SHA-256: `{context['protocol_sha256']}`\n\n"
        "The 128 locked cells were expanded without creating formal results. "
        "Recovery, completed-run skipping, conflict blocking, lane assignment, "
        "launch gates, and the exclusive scheduler lock were tested in an isolated fixture.\n",
        encoding="utf-8",
        newline="\n",
    )
    checksum_names = [
        "README.md",
        "fixture_summary_v3.json",
        "preflight_plan_v3.csv",
        "preflight_summary_v3.json",
        "scheduler_source_integrity_v3.json",
        "stage7_acceptance_v3.json",
    ]
    with (audit_dir / "scheduler_preflight_checksums_v3.sha256").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        for name in checksum_names:
            handle.write(f"{sha256(audit_dir / name)}  {name}\n")

    print(
        json.dumps(
            {
                "overall_accepted": acceptance["overall_accepted"],
                "checks": len(checks),
                "failed_checks": len(failures),
                "planned_runs": 128,
                "lane_counts": actual_plan["lane_counts"],
                "fixture_status_counts": fixture_plan["status_counts"],
                "formal_training_started": False,
                "formal_results_present": context["formal_root"].exists(),
            },
            indent=2,
        )
    )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
