#!/usr/bin/env python3
"""Independent verification of V3 scheduler preflight artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import subprocess
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


def load_scheduler(path: Path):
    spec = importlib.util.spec_from_file_location("scheduler_verify_v3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


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
        / "results"
    )
    checks: list[dict] = []

    def check(name: str, condition: bool, **details) -> None:
        checks.append({"check": name, "passed": bool(condition), **details})

    required_audit = [
        "README.md",
        "fixture_summary_v3.json",
        "preflight_plan_v3.csv",
        "preflight_summary_v3.json",
        "scheduler_preflight_checksums_v3.sha256",
        "scheduler_source_integrity_v3.json",
        "stage7_acceptance_v3.json",
    ]
    for name in required_audit:
        check(f"audit_file_exists:{name}", (audit_dir / name).is_file())

    acceptance = read_json(audit_dir / "stage7_acceptance_v3.json")
    check("stage7_acceptance_true", acceptance.get("overall_accepted") is True)
    check("stage7_acceptance_zero_failures", acceptance.get("failed_checks") == 0)
    check("stage7_acceptance_plans_128", acceptance.get("planned_runs") == 128)
    check("stage7_records_no_training", acceptance.get("formal_training_started") is False)
    check("stage7_records_no_formal_results", acceptance.get("formal_results_present") is False)
    for item in acceptance.get("checks", []):
        check(f"embedded_acceptance:{item.get('check')}", item.get("passed") is True)

    checksum_lines = (audit_dir / "scheduler_preflight_checksums_v3.sha256").read_text(
        encoding="utf-8"
    ).splitlines()
    checksum_map = {}
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        checksum_map[name] = digest
    check("checksum_manifest_has_six_entries", len(checksum_map) == 6)
    for name, expected in checksum_map.items():
        check(f"audit_sha256:{name}", sha256(audit_dir / name) == expected)

    integrity = read_json(audit_dir / "scheduler_source_integrity_v3.json")
    check("source_integrity_accepted", integrity.get("all_sources_unchanged") is True)
    for name, item in integrity.get("source_files", {}).items():
        source = root / item["path"]
        check(f"source_exists:{name}", source.is_file())
        check(f"source_current_sha256:{name}", sha256(source) == item["sha256_after"])
        check(f"source_unchanged:{name}", item.get("unchanged") is True)

    formal_absent_before = not context["formal_root"].exists()
    check("formal_results_absent_before_independent_checks", formal_absent_before)
    plan = scheduler.build_plan(context)
    check("live_plan_128", plan["total_cells"] == 128)
    check(
        "live_status_counts_all_pending",
        plan["status_counts"]
        == {"pending_new": 128, "retry_failed": 0, "skip_completed": 0, "conflict": 0},
    )
    check("live_lane_counts", plan["lane_counts"] == {"cuda:0": 48, "cuda:1": 48, "cpu": 32})
    check("live_plan_safe", plan["safe_to_launch"] is True)
    check("live_fingerprint_is_sha256", len(plan["state_fingerprint"]) == 64)

    with (audit_dir / "preflight_plan_v3.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as handle:
        csv_rows = list(csv.DictReader(handle))
    check("csv_has_128_rows", len(csv_rows) == 128)
    check("csv_ordinals_are_1_to_128", [int(row["ordinal"]) for row in csv_rows] == list(range(1, 129)))
    check("csv_cell_ids_unique", len({row["cell_id"] for row in csv_rows}) == 128)
    check(
        "csv_matches_live_cell_ids",
        [row["cell_id"] for row in csv_rows] == [row["cell_id"] for row in plan["cells"]],
    )
    check("csv_all_pending_new", {row["status"] for row in csv_rows} == {"pending_new"})
    check("csv_all_attempt_1", {row["attempt"] for row in csv_rows} == {"1"})

    stored_summary = read_json(audit_dir / "preflight_summary_v3.json")
    for key in (
        "protocol_sha256",
        "total_cells",
        "status_counts",
        "lane_counts",
        "actionable_cells",
        "safe_to_launch",
        "state_fingerprint",
        "formal_root_exists",
    ):
        check(f"stored_summary_matches_live:{key}", stored_summary.get(key) == plan.get(key))

    fixture_plan = scheduler.build_plan(context, formal_root=fixture_root)
    check(
        "fixture_live_counts",
        fixture_plan["status_counts"]
        == {"pending_new": 124, "retry_failed": 1, "skip_completed": 1, "conflict": 2},
    )
    check("fixture_live_blocks_launch", fixture_plan["safe_to_launch"] is False)
    fixture_stored = read_json(audit_dir / "fixture_summary_v3.json")
    check("fixture_stored_counts_match", fixture_stored["status_counts"] == fixture_plan["status_counts"])
    for expected, row in zip(
        ("skip_completed", "retry_failed", "conflict", "conflict"),
        fixture_plan["cells"][:4],
        strict=True,
    ):
        check(f"fixture_status:{row['cell_id']}", row["status"] == expected)
    check("fixture_retry_uses_attempt_2", fixture_plan["cells"][1]["attempt"] == 2)

    source_text = scheduler_path.read_text(encoding="utf-8")
    check("scheduler_has_execute_flag", 'add_argument("--execute", action="store_true")' in source_text)
    check("scheduler_has_exact_launch_phrase", scheduler.LAUNCH_PHRASE in source_text)
    check("scheduler_has_exclusive_lock", "lock_dir.mkdir()" in source_text)
    check("scheduler_rechecks_state_fingerprint", 'fresh_plan["state_fingerprint"]' in source_text)
    check("scheduler_calls_runner_without_shell", "subprocess.run(command" in source_text and "shell=True" not in source_text)
    check("scheduler_revalidates_stage7_sources", "validate_stage7_preflight(context)" in source_text)
    check("scheduler_requires_two_cuda_devices", "torch.cuda.device_count() < 2" in source_text)

    base_command = [sys.executable, str(scheduler_path), "--project-root", str(root)]
    default_call = subprocess.run(base_command, text=True, capture_output=True, check=False)
    check("default_cli_preflight_succeeds", default_call.returncode == 0)
    default_payload = json.loads(default_call.stdout) if default_call.returncode == 0 else {}
    check("default_cli_is_read_only_mode", default_payload.get("mode") == "preflight_read_only")
    check("default_cli_expands_128", default_payload.get("total_cells") == 128)

    missing_gates = subprocess.run(
        [*base_command, "--execute"], text=True, capture_output=True, check=False
    )
    check("execute_without_confirmations_fails", missing_gates.returncode != 0)
    wrong_phrase = subprocess.run(
        [
            *base_command,
            "--execute",
            "--confirm-protocol-sha",
            context["protocol_sha256"],
            "--confirm-launch-phrase",
            "WRONG",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    check("execute_with_wrong_phrase_fails", wrong_phrase.returncode != 0)
    check("negative_gate_tests_create_no_results", not context["formal_root"].exists())
    check("negative_gate_tests_create_no_logs", not context["log_root"].exists())

    try:
        import torch

        gpu_count = torch.cuda.device_count()
    except Exception:
        gpu_count = 0
    check("two_cuda_devices_available", gpu_count >= 2, gpu_count=gpu_count)
    correct_gate_validates = False
    try:
        scheduler.validate_execution_authorization(
            Namespace(
                execute=True,
                confirm_protocol_sha=context["protocol_sha256"],
                confirm_launch_phrase=scheduler.LAUNCH_PHRASE,
            ),
            context,
            plan,
        )
        correct_gate_validates = True
    except RuntimeError:
        correct_gate_validates = False
    check("all_three_gates_validate_without_launching", correct_gate_validates)
    check("formal_results_absent_after_independent_checks", not context["formal_root"].exists())

    failures = [item for item in checks if not item["passed"]]
    output = {
        "independent_verification_passed": not failures,
        "checks": len(checks),
        "failed_checks": len(failures),
        "failures": failures,
        "planned_runs": 128,
        "protocol_sha256": context["protocol_sha256"],
        "formal_training_started": False,
        "formal_results_present": context["formal_root"].exists(),
    }
    with (audit_dir / "independent_verification_v3.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps(output, indent=2, ensure_ascii=False))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
