#!/usr/bin/env python3
"""Aggregate and independently verify the M5 input-level masking matrix."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
from random_holdout_common_v1 import load_random_split  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def score(y: np.ndarray, probability: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    truth = (y[mask] == 1).astype(np.int8); values = np.asarray(probability[mask], dtype=np.float64)
    return {"n": int(truth.size), "positive": int(truth.sum()), "positive_rate": float(truth.mean()),
            "average_precision": float(average_precision_score(truth, values)),
            "roc_auc": float(roc_auc_score(truth, values))}


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--project-root", type=Path, default=Path.cwd()); parser.add_argument("--protocol-sha", required=True)
    args = parser.parse_args(); root = args.project_root.resolve(); protocol = json.loads((root / "03_configs/random_holdout_input_mask_v1_protocol.json").read_text(encoding="utf-8"))
    output = root / "04_results/random_holdout_input_mask_v1" / args.protocol_sha; rows = []; ensemble_rows = []; checks = []
    def check(name: str, passed: bool, **evidence) -> None: checks.append({"name": name, "passed": bool(passed), **evidence})
    for model in protocol["models"]:
        for split in protocol["splits"]:
            data = load_random_split(root, split, "6391a2680c2173f0deda638f2b8c317d1d681def25b73e7da9df9ccf92148627")
            probabilities = []; baseline_probabilities = []
            for seed in protocol["seeds"]:
                cell = output / model / split / f"seed_{seed}"; result_path = cell / "result.json"; array_path = cell / "masked_probability.npy"; manifest_path = cell / "run_manifest.json"
                present = result_path.is_file() and array_path.is_file() and manifest_path.is_file()
                check(f"cell_present:{model}:{split}:{seed}", present)
                if not present: continue
                result = json.loads(result_path.read_text(encoding="utf-8")); manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                check(f"cell_protocol:{model}:{split}:{seed}", result.get("protocol_sha256") == args.protocol_sha)
                check(f"cell_status:{model}:{split}:{seed}", manifest.get("status") == "complete")
                check(f"cell_hash:{model}:{split}:{seed}", sha256(array_path) == manifest.get("file_sha256", {}).get("masked_probability.npy"))
                probability = np.load(array_path, mmap_mode="r"); probabilities.append(np.asarray(probability))
                baseline_path = root / "01_data/interim/results_random_holdout_v1_1" / protocol["parent_inference_protocol_sha256"] / model / split / f"seed_{seed}" / "probability.npy"
                check(f"baseline_present:{model}:{split}:{seed}", baseline_path.is_file())
                if baseline_path.is_file(): baseline_probabilities.append(np.asarray(np.load(baseline_path, mmap_mode="r")))
                recomputed = score(data["y"], probability, data["test"]); stored = result["masked_test"]
                check(f"cell_metric:{model}:{split}:{seed}", all(abs(float(recomputed[key]) - float(stored[key])) <= 1e-10 for key in ("n", "positive", "positive_rate", "average_precision", "roc_auc")))
                rows.append({"model_id": model, "split_id": split, "source_fold": data["source_fold"], "seed": seed,
                             "baseline_ap": result["unmasked_test"]["average_precision"], "masked_ap": stored["average_precision"],
                             "delta_ap": result["delta_masked_minus_unmasked"]["average_precision"],
                             "baseline_auc": result["unmasked_test"]["roc_auc"], "masked_auc": stored["roc_auc"]})
            if len(probabilities) == len(protocol["seeds"]):
                ensemble = np.mean(np.stack(probabilities), axis=0)
                ensemble_score = score(data["y"], ensemble, data["test"])
                baseline_values = [r["baseline_ap"] for r in rows if r["model_id"] == model and r["split_id"] == split]
                baseline_ensemble_score = score(data["y"], np.mean(np.stack(baseline_probabilities), axis=0), data["test"]) if len(baseline_probabilities) == len(protocol["seeds"]) else {}
                ensemble_rows.append({"model_id": model, "split_id": split, "source_fold": data["source_fold"],
                                      "baseline_seed_mean_ap": float(np.mean(baseline_values)), "baseline_ensemble_ap": baseline_ensemble_score.get("average_precision"),
                                      "masked_ensemble_ap": ensemble_score["average_precision"], "delta_ensemble_ap": ensemble_score["average_precision"] - baseline_ensemble_score.get("average_precision", float("nan")),
                                      "baseline_ensemble_auc": baseline_ensemble_score.get("roc_auc"), "masked_ensemble_auc": ensemble_score["roc_auc"], "prevalence": ensemble_score["positive_rate"]})
    write_csv(output / "input_mask_cell_metrics_v1.csv", rows)
    write_csv(output / "input_mask_ensemble_metrics_v1.csv", ensemble_rows)
    failures = [item for item in checks if not item["passed"]]
    report = {"status": "accepted" if not failures and len(rows) == 48 else "rejected", "protocol_sha256": args.protocol_sha,
              "cells_present": len(rows), "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures}
    (output / "independent_verification_input_mask_v1.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "summary_input_mask_v1.json").write_text(json.dumps({"status": report["status"], "protocol_sha256": args.protocol_sha,
        "cells": len(rows), "models": protocol["models"], "splits": protocol["splits"], "seeds": protocol["seeds"]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False)); raise SystemExit(1 if failures or len(rows) != 48 else 0)


if __name__ == "__main__": main()
