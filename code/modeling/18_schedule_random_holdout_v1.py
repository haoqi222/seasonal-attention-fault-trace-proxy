#!/usr/bin/env python3
"""Two-GPU resumable scheduler for the 48-cell random-holdout diagnostic."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from random_holdout_common_v1 import protocol_context  # noqa: E402


LAUNCH_PHRASE = "START_RANDOM_HOLDOUT_V1_48_CELLS"


def run_lane(root: Path, digest: str, rows: list[dict], log_root: Path) -> list[dict]:
    outcomes = []
    runner = root / "02_scripts/modeling/17_run_random_holdout_v1.py"
    for row in rows:
        final_dir = (root / "01_data/interim/results_random_holdout_v1" / digest /
                     row["model_id"] / row["split_id"] / f"seed_{row['seed']}")
        if final_dir.is_dir() and (final_dir / "run_manifest.json").is_file():
            outcomes.append({**row, "status": "skipped_existing"})
            continue
        if final_dir.exists() or final_dir.with_name(final_dir.name + ".attempt").exists():
            raise RuntimeError(f"Conflicting output requires review: {final_dir}")
        log_path = log_root / f"{row['ordinal']:02d}_{row['model_id']}__{row['split_id']}__seed_{row['seed']}.log"
        command = [sys.executable, str(runner), "--project-root", str(root),
                   "--confirm-protocol-sha", digest, "--model-id", row["model_id"],
                   "--split-id", row["split_id"], "--seed", str(row["seed"]),
                   "--device", row["device"]]
        with log_path.open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Cell failed; see {log_path}")
        outcomes.append({**row, "status": "completed", "log": str(log_path)})
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirm-protocol-sha", required=True)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--launch-phrase")
    args = parser.parse_args()
    root = args.project_root.resolve()
    context = protocol_context(root, args.confirm_protocol_sha)
    protocol, digest = context["protocol"], context["sha256"]
    rows, ordinal = [], 0
    for model_id in protocol["models"]:
        for split in protocol["splits"]:
            for seed in protocol["seeds"]:
                ordinal += 1
                rows.append({"ordinal": ordinal, "model_id": model_id,
                             "split_id": split["split_id"], "seed": seed,
                             "device": f"cuda:{(ordinal - 1) % 2}"})
    if len(rows) != 48 or len({(r["model_id"], r["split_id"], r["seed"]) for r in rows}) != 48:
        raise RuntimeError("Frozen matrix is not exactly 48 unique cells")
    prepared = root / "01_data/interim/random_holdout_v1" / digest / "build_acceptance.json"
    if not prepared.is_file() or json.loads(prepared.read_text(encoding="utf-8")).get("status") != "accepted":
        raise RuntimeError("Prepared random splits are not accepted")
    pending = [r for r in rows if not (root / "01_data/interim/results_random_holdout_v1" / digest /
               r["model_id"] / r["split_id"] / f"seed_{r['seed']}" / "run_manifest.json").is_file()]
    print(json.dumps({"protocol_sha256": digest, "total": 48, "pending": len(pending),
                      "launch_requested": args.launch}, indent=2))
    if not args.launch:
        return
    if args.launch_phrase != LAUNCH_PHRASE:
        raise ValueError("Launch phrase mismatch")
    log_root = root / "99_logs/random_holdout_v1" / digest
    log_root.mkdir(parents=True, exist_ok=True)
    lanes = [[row for row in rows if row["device"] == device] for device in ("cuda:0", "cuda:1")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(run_lane, root, digest, lane, log_root) for lane in lanes]
        outcomes = [item for future in futures for item in future.result()]
    summary = {"status": "complete", "protocol_sha256": digest, "cells": len(outcomes),
               "completed_now": sum(x["status"] == "completed" for x in outcomes),
               "skipped_existing": sum(x["status"] == "skipped_existing" for x in outcomes)}
    (log_root / "scheduler_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
