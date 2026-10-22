#!/usr/bin/env python3
"""Run every frozen M5 input-mask cell without changing the parent checkpoints."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirm-protocol-sha", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol = json.loads((root / "03_configs/random_holdout_input_mask_v1_protocol.json").read_text(encoding="utf-8"))
    script = root / "02_scripts/analysis/43_run_input_masked_random_holdout_v1.py"
    progress_path = root / "04_results/random_holdout_input_mask_v1" / args.confirm_protocol_sha / "matrix_progress.json"
    progress_path.parent.mkdir(parents=True, exist_ok=True)
    cells = [(model, split, seed) for model in protocol["models"] for split in protocol["splits"] for seed in protocol["seeds"]]
    records = []
    for model, split, seed in cells:
        destination = root / "04_results/random_holdout_input_mask_v1" / args.confirm_protocol_sha / model / split / f"seed_{seed}"
        if destination.is_dir() and (destination / "run_manifest.json").is_file():
            records.append({"model_id": model, "split_id": split, "seed": seed, "status": "already_complete"})
            continue
        command = [sys.executable, str(script), "--project-root", str(root),
                   "--confirm-protocol-sha", args.confirm_protocol_sha,
                   "--model-id", model, "--split-id", split, "--seed", str(seed), "--device", args.device]
        completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
        record = {"model_id": model, "split_id": split, "seed": seed, "returncode": completed.returncode,
                  "stdout": completed.stdout[-2000:], "stderr": completed.stderr[-4000:]}
        records.append(record)
        progress_path.write_text(json.dumps({"status": "running", "completed_records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if completed.returncode != 0:
            raise SystemExit(f"M5 cell failed: {model} {split} seed={seed}\n{completed.stderr}")
    progress_path.write_text(json.dumps({"status": "complete", "cells": len(cells), "completed_records": records}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "cells": len(cells), "progress": str(progress_path)}))


if __name__ == "__main__":
    main()
