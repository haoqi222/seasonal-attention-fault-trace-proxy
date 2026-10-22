#!/usr/bin/env python3
"""Run all 48 V1.1 inference cells on two GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from pathlib import Path


PHRASE = "START_RANDOM_V1_1_INFERENCE_48"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lane(root: Path, digest: str, cells: list[dict], logdir: Path) -> list[dict]:
    outcomes = []
    for cell in cells:
        destination = root / "01_data/interim/results_random_holdout_v1_1" / digest / cell["model"] / cell["split"] / f"seed_{cell['seed']}"
        if (destination / "run_manifest.json").is_file():
            outcomes.append({**cell, "status": "skipped"}); continue
        if destination.exists() or destination.with_name(destination.name + ".attempt").exists():
            raise RuntimeError(f"Conflicting output: {destination}")
        command = [sys.executable, str(root / "02_scripts/modeling/19_rerun_random_holdout_inference_v1_1.py"),
                   "--project-root", str(root), "--confirm-protocol-sha", digest,
                   "--model-id", cell["model"], "--split-id", cell["split"],
                   "--seed", str(cell["seed"]), "--device", cell["device"]]
        log = logdir / f"{cell['ordinal']:02d}_{cell['model']}__{cell['split']}__seed_{cell['seed']}.log"
        with log.open("w", encoding="utf-8") as handle:
            completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"Inference failed: {log}")
        outcomes.append({**cell, "status": "completed"})
    return outcomes


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--confirm-protocol-sha", required=True); parser.add_argument("--launch-phrase", required=True)
    args = parser.parse_args(); root = args.project_root.resolve()
    path = root / "03_configs/random_holdout_v1_1_inference_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8")); digest = sha256(path)
    if args.confirm_protocol_sha != digest or args.launch_phrase != PHRASE:
        raise ValueError("Launch gate mismatch")
    cells=[]; ordinal=0
    for model in protocol["models"]:
        for split in protocol["splits"]:
            for seed in protocol["seeds"]:
                ordinal += 1; cells.append({"ordinal":ordinal,"model":model,"split":split,"seed":seed,
                                            "device":f"cuda:{(ordinal-1)%2}"})
    if len(cells) != 48: raise RuntimeError("Matrix is not 48 cells")
    logdir = root / "99_logs/random_holdout_v1_1" / digest; logdir.mkdir(parents=True, exist_ok=True)
    lanes = [[c for c in cells if c["device"] == device] for device in ("cuda:0", "cuda:1")]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = [item for future in [executor.submit(lane, root, digest, x, logdir) for x in lanes]
                   for item in future.result()]
    summary={"status":"complete","protocol_sha256":digest,"cells":len(results),
             "completed_now":sum(x["status"]=="completed" for x in results),
             "skipped":sum(x["status"]=="skipped" for x in results)}
    (logdir/"scheduler_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__ == "__main__": main()
