#!/usr/bin/env python3
"""Run one M5 model block so independent model blocks can be scheduled in parallel."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path.cwd()); parser.add_argument("--confirm-protocol-sha", required=True); parser.add_argument("--model-id", required=True); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); root = args.project_root.resolve(); script = root / "02_scripts/analysis/43_run_input_masked_random_holdout_v1.py"
    for split in ("R1_MatchedF1", "R2_MatchedF2", "R3_MatchedF3", "R4_MatchedF4"):
        for seed in range(4):
            destination = root / "04_results/random_holdout_input_mask_v1" / args.confirm_protocol_sha / args.model_id / split / f"seed_{seed}"
            if destination.is_dir() and (destination / "run_manifest.json").is_file(): continue
            result = subprocess.run([sys.executable, str(script), "--project-root", str(root), "--confirm-protocol-sha", args.confirm_protocol_sha, "--model-id", args.model_id, "--split-id", split, "--seed", str(seed), "--device", args.device], cwd=root, text=True)
            if result.returncode: raise SystemExit(result.returncode)
    print(f"complete: {args.model_id}")


if __name__ == "__main__": main()
