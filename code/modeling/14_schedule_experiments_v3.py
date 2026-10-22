#!/usr/bin/env python3
"""Resumable scheduler for the frozen V3 experiment matrix.

The default CLI is read-only preflight. Formal subprocesses can only be launched
when all three execution gates are supplied explicitly.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import threading
from dataclasses import asdict, dataclass
from pathlib import Path


PROTOCOL_REL = Path("03_configs/experiment_v3_protocol.json")
LOCK_REL = Path("04_results/training_protocol_v3/protocol_lock_v3.json")
RUNNER_REL = Path("02_scripts/modeling/13_run_experiment_v3.py")
FORMAL_ROOT_REL = Path("01_data/interim/results_v3")
LOG_ROOT_REL = Path("99_logs/experiment_v3")
PREFLIGHT_AUDIT_REL = Path("04_results/scheduler_preflight_v3")
LAUNCH_PHRASE = "START_V3_128_LOCKED_CELLS"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def load_context(project_root: Path) -> dict:
    root = project_root.resolve()
    protocol_path = root / PROTOCOL_REL
    lock_path = root / LOCK_REL
    runner_path = root / RUNNER_REL
    for required in (protocol_path, lock_path, runner_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    protocol = read_json(protocol_path)
    protocol_sha = sha256(protocol_path)
    lock = read_json(lock_path)
    if lock.get("status") != "locked":
        raise RuntimeError("Stage-5 protocol lock is not locked")
    if lock.get("training_started") is not False:
        raise RuntimeError("Stage-5 lock no longer records a pre-training state")
    if lock.get("protocol_sha256") != protocol_sha:
        raise RuntimeError("Protocol hash differs from the stage-5 lock")
    if protocol["run_matrix"] != {
        "deep_runs": 96,
        "random_forest_runs": 32,
        "total_runs": 128,
        "required_completion": "all configured model x fold x seed cells before aggregate comparison",
    }:
        raise RuntimeError("Frozen run-matrix declaration is not the accepted 96+32=128 matrix")
    return {
        "root": root,
        "protocol": protocol,
        "protocol_sha256": protocol_sha,
        "lock": lock,
        "runner": runner_path,
        "formal_root": root / FORMAL_ROOT_REL / protocol_sha,
        "log_root": root / LOG_ROOT_REL / protocol_sha,
    }


@dataclass(frozen=True)
class Cell:
    ordinal: int
    cell_id: str
    model_id: str
    model_kind: str
    fold_id: str
    seed: int
    lane: str


def generate_cells(protocol: dict) -> list[Cell]:
    cells: list[Cell] = []
    ordinal = 0
    deep_index = 0
    for model in protocol["deep_models"]:
        for fold_id in protocol["data"]["folds"]:
            for seed in protocol["seeds"]:
                ordinal += 1
                lane = f"cuda:{deep_index % 2}"
                cells.append(
                    Cell(
                        ordinal=ordinal,
                        cell_id=f"{model['model_id']}__{fold_id}__seed_{seed}",
                        model_id=model["model_id"],
                        model_kind="deep",
                        fold_id=fold_id,
                        seed=int(seed),
                        lane=lane,
                    )
                )
                deep_index += 1
    for model in protocol["random_forest_models"]:
        for fold_id in protocol["data"]["folds"]:
            for seed in protocol["seeds"]:
                ordinal += 1
                cells.append(
                    Cell(
                        ordinal=ordinal,
                        cell_id=f"{model['model_id']}__{fold_id}__seed_{seed}",
                        model_id=model["model_id"],
                        model_kind="random_forest",
                        fold_id=fold_id,
                        seed=int(seed),
                        lane="cpu",
                    )
                )
    if len(cells) != 128 or ordinal != 128:
        raise RuntimeError(f"Generated matrix has {len(cells)} cells instead of 128")
    if len({cell.cell_id for cell in cells}) != 128:
        raise RuntimeError("Generated matrix contains duplicate cell identifiers")
    return cells


def cell_directory(formal_root: Path, cell: Cell) -> Path:
    return formal_root / cell.model_id / cell.fold_id / f"seed_{cell.seed}"


def attempt_directories(final_dir: Path) -> list[Path]:
    pattern = final_dir.name + ".attempt_*"
    return sorted(final_dir.parent.glob(pattern)) if final_dir.parent.exists() else []


def parse_attempt_number(path: Path, final_name: str) -> int:
    prefix = final_name + ".attempt_"
    suffix = path.name[len(prefix) :]
    if not suffix.isdigit() or int(suffix) < 1:
        raise ValueError(f"Invalid attempt directory name: {path.name}")
    return int(suffix)


def validate_completed_run(
    final_dir: Path, cell: Cell, protocol: dict, protocol_sha: str
) -> list[str]:
    problems: list[str] = []
    required = list(protocol["output_schema"]["required_run_files"])
    required += (
        protocol["output_schema"]["deep_only_files"]
        if cell.model_kind == "deep"
        else protocol["output_schema"]["random_forest_only_files"]
    )
    for name in required:
        if not (final_dir / name).is_file():
            problems.append(f"missing:{name}")
    if problems:
        return problems
    try:
        manifest = read_json(final_dir / "run_manifest.json")
        result = read_json(final_dir / "result.json")
    except Exception as error:
        return [f"invalid_json:{type(error).__name__}:{error}"]
    expected_manifest = {
        "scope": "formal_v3_run",
        "protocol_sha256": protocol_sha,
        "model_id": cell.model_id,
        "model_kind": cell.model_kind,
        "fold_id": cell.fold_id,
        "seed": cell.seed,
        "completed": True,
    }
    for key, expected in expected_manifest.items():
        if manifest.get(key) != expected:
            problems.append(f"manifest_{key}:{manifest.get(key)!r}")
    for key, expected in {
        "protocol_sha256": protocol_sha,
        "model_id": cell.model_id,
        "model_kind": cell.model_kind,
        "fold_id": cell.fold_id,
        "seed": cell.seed,
    }.items():
        if result.get(key) != expected:
            problems.append(f"result_{key}:{result.get(key)!r}")
    hashes = manifest.get("file_sha256")
    if not isinstance(hashes, dict):
        problems.append("manifest_file_sha256_missing")
    else:
        expected_hashed = sorted(name for name in required if name != "run_manifest.json")
        if sorted(hashes) != expected_hashed:
            problems.append("manifest_file_list_mismatch")
        else:
            for name in expected_hashed:
                if hashes[name] != sha256(final_dir / name):
                    problems.append(f"hash_mismatch:{name}")
    return problems


def inspect_cell(cell: Cell, context: dict, formal_root: Path | None = None) -> dict:
    output_root = formal_root if formal_root is not None else context["formal_root"]
    final_dir = cell_directory(output_root, cell)
    attempts = attempt_directories(final_dir)
    attempt_numbers: list[int] = []
    invalid_attempt_names: list[str] = []
    active_attempts: list[str] = []
    failed_attempts: list[str] = []
    for attempt_dir in attempts:
        try:
            attempt_numbers.append(parse_attempt_number(attempt_dir, final_dir.name))
        except ValueError:
            invalid_attempt_names.append(attempt_dir.name)
            continue
        if (attempt_dir / "failure.json").is_file():
            failed_attempts.append(attempt_dir.name)
        else:
            active_attempts.append(attempt_dir.name)
    next_attempt = max(attempt_numbers, default=0) + 1
    problems: list[str] = []
    if invalid_attempt_names:
        problems.extend(f"invalid_attempt_name:{name}" for name in invalid_attempt_names)
    if final_dir.exists():
        if not final_dir.is_dir():
            problems.append("final_path_is_not_directory")
        else:
            problems.extend(
                validate_completed_run(
                    final_dir, cell, context["protocol"], context["protocol_sha256"]
                )
            )
        if active_attempts:
            problems.append("active_attempt_exists_beside_completed_path")
        status = "skip_completed" if not problems else "conflict"
    elif active_attempts or problems:
        if active_attempts:
            problems.append("unfinished_attempt_requires_manual_review")
        status = "conflict"
    elif failed_attempts:
        status = "retry_failed"
    else:
        status = "pending_new"
    return {
        **asdict(cell),
        "status": status,
        "attempt": next_attempt,
        "final_dir": str(final_dir),
        "failed_attempts": failed_attempts,
        "active_attempts": active_attempts,
        "problems": problems,
    }


def build_plan(context: dict, formal_root: Path | None = None) -> dict:
    rows = [inspect_cell(cell, context, formal_root) for cell in generate_cells(context["protocol"])]
    counts = {
        status: sum(row["status"] == status for row in rows)
        for status in ("pending_new", "retry_failed", "skip_completed", "conflict")
    }
    lane_counts = {
        lane: sum(row["lane"] == lane for row in rows)
        for lane in ("cuda:0", "cuda:1", "cpu")
    }
    state_payload = [
        {
            "cell_id": row["cell_id"],
            "status": row["status"],
            "attempt": row["attempt"],
            "failed_attempts": row["failed_attempts"],
            "active_attempts": row["active_attempts"],
            "problems": row["problems"],
        }
        for row in rows
    ]
    state_fingerprint = hashlib.sha256(
        json.dumps(state_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": "fault-proxy-v3-scheduler-plan-1.0",
        "mode": "preflight_read_only",
        "protocol_sha256": context["protocol_sha256"],
        "total_cells": len(rows),
        "status_counts": counts,
        "lane_counts": lane_counts,
        "actionable_cells": counts["pending_new"] + counts["retry_failed"],
        "safe_to_launch": counts["conflict"] == 0,
        "state_fingerprint": state_fingerprint,
        "formal_root_exists": context["formal_root"].exists(),
        "cells": rows,
    }


def runner_command(context: dict, row: dict) -> list[str]:
    return [
        sys.executable,
        str(context["runner"]),
        "--project-root",
        str(context["root"]),
        "--model-id",
        row["model_id"],
        "--fold-id",
        row["fold_id"],
        "--seed",
        str(row["seed"]),
        "--device",
        row["lane"] if row["model_kind"] == "deep" else "cpu",
        "--attempt",
        str(row["attempt"]),
        "--confirm-protocol-sha",
        context["protocol_sha256"],
    ]


def validate_stage7_preflight(context: dict) -> None:
    audit_dir = context["root"] / PREFLIGHT_AUDIT_REL
    acceptance_path = audit_dir / "stage7_acceptance_v3.json"
    integrity_path = audit_dir / "scheduler_source_integrity_v3.json"
    if not acceptance_path.is_file() or not integrity_path.is_file():
        raise RuntimeError("Formal execution requires the accepted stage-7 audit package")
    acceptance = read_json(acceptance_path)
    if acceptance.get("overall_accepted") is not True or acceptance.get("failed_checks") != 0:
        raise RuntimeError("Stage-7 preflight is not accepted")
    if acceptance.get("protocol_sha256") != context["protocol_sha256"]:
        raise RuntimeError("Stage-7 preflight belongs to a different protocol")
    integrity = read_json(integrity_path)
    if integrity.get("all_sources_unchanged") is not True:
        raise RuntimeError("Stage-7 source-integrity record is not accepted")
    for name, item in integrity.get("source_files", {}).items():
        source = context["root"] / item["path"]
        if not source.is_file() or sha256(source) != item.get("sha256_after"):
            raise RuntimeError(f"Stage-7 source changed after preflight: {name}")


def validate_execution_authorization(args: argparse.Namespace, context: dict, plan: dict) -> None:
    if not args.execute:
        raise RuntimeError("Execution mode was not requested")
    if args.confirm_protocol_sha != context["protocol_sha256"]:
        raise RuntimeError("Execution requires the exact frozen protocol SHA-256")
    if args.confirm_launch_phrase != LAUNCH_PHRASE:
        raise RuntimeError("Execution requires the exact launch phrase")
    if plan["total_cells"] != 128 or not plan["safe_to_launch"]:
        raise RuntimeError("The 128-cell plan contains a conflict or is incomplete")
    if plan["actionable_cells"] == 0:
        raise RuntimeError("No pending or retry cells remain")
    validate_stage7_preflight(context)
    try:
        import torch
    except Exception as error:
        raise RuntimeError("PyTorch is required for the formal dual-GPU launch") from error
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        raise RuntimeError("Formal execution requires both configured CUDA devices")


def acquire_lock(lock_dir: Path) -> None:
    lock_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        lock_dir.mkdir()
    except FileExistsError as error:
        raise RuntimeError(f"Scheduler lock already exists: {lock_dir}") from error
    write_json(lock_dir / "owner.json", {"pid": os.getpid(), "argv": sys.argv})


def release_lock(lock_dir: Path) -> None:
    owner = lock_dir / "owner.json"
    if owner.exists():
        owner.unlink()
    lock_dir.rmdir()


def run_lane(context: dict, lane: str, rows: list[dict], stop: threading.Event) -> list[dict]:
    outcomes: list[dict] = []
    lane_log = context["log_root"] / lane.replace(":", "_")
    lane_log.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if stop.is_set():
            outcomes.append({"cell_id": row["cell_id"], "status": "not_started_after_failure"})
            continue
        command = runner_command(context, row)
        log_path = lane_log / f"{row['ordinal']:03d}_{row['cell_id']}_attempt_{row['attempt']:03d}.log"
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        status = "completed" if completed.returncode == 0 else "failed"
        outcomes.append(
            {
                "cell_id": row["cell_id"],
                "lane": lane,
                "attempt": row["attempt"],
                "returncode": completed.returncode,
                "status": status,
                "log": str(log_path),
            }
        )
        if completed.returncode != 0:
            stop.set()
    return outcomes


def execute_plan(context: dict, plan: dict) -> dict:
    lock_dir = context["log_root"] / "scheduler.lock"
    acquire_lock(lock_dir)
    try:
        fresh_plan = build_plan(context)
        if fresh_plan["state_fingerprint"] != plan["state_fingerprint"]:
            raise RuntimeError("Formal state changed between authorization and lock acquisition")
        actionable = [
            row
            for row in fresh_plan["cells"]
            if row["status"] in {"pending_new", "retry_failed"}
        ]
        grouped = {lane: [row for row in actionable if row["lane"] == lane] for lane in ("cuda:0", "cuda:1", "cpu")}
        stop = threading.Event()
        outcomes: list[dict] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(run_lane, context, lane, rows, stop) for lane, rows in grouped.items() if rows]
            for future in concurrent.futures.as_completed(futures):
                outcomes.extend(future.result())
        summary = {
            "schema_version": "fault-proxy-v3-scheduler-execution-1.0",
            "protocol_sha256": context["protocol_sha256"],
            "completed": sum(item["status"] == "completed" for item in outcomes),
            "failed": sum(item["status"] == "failed" for item in outcomes),
            "not_started_after_failure": sum(item["status"] == "not_started_after_failure" for item in outcomes),
            "outcomes": sorted(outcomes, key=lambda item: item["cell_id"]),
        }
        write_json(context["log_root"] / "scheduler_execution_summary.json", summary)
        return summary
    finally:
        release_lock(lock_dir)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--project-root", type=Path, default=Path.cwd())
    result.add_argument("--execute", action="store_true")
    result.add_argument("--confirm-protocol-sha")
    result.add_argument("--confirm-launch-phrase")
    return result


def main() -> None:
    args = parser().parse_args()
    context = load_context(args.project_root)
    plan = build_plan(context)
    if not args.execute:
        print(json.dumps(plan, indent=2, ensure_ascii=False))
        return
    validate_execution_authorization(args, context, plan)
    print(json.dumps(execute_plan(context, plan), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
