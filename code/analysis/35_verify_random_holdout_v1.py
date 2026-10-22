#!/usr/bin/env python3
"""Independent verification of random-holdout aggregates."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


MODELS = ["unet_matched", "temporal_attn", "temporal_concat"]
SEEDS = [0, 1, 2, 3]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / "03_configs/random_holdout_v1_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256(protocol_path)
    raw = root / "01_data/interim/results_random_holdout_v1" / protocol_sha
    prepared = root / "01_data/interim/random_holdout_v1" / protocol_sha
    aggregate = root / "04_results/random_holdout_v1_aggregate"
    checks = []

    def check(name: str, passed: bool, **evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    checksum_lines = (aggregate / "aggregate_checksums_random_v1.sha256").read_text(encoding="ascii").splitlines()
    check("aggregate_checksum_count", len(checksum_lines) == 10, count=len(checksum_lines))
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        check("aggregate_hash:" + name, sha256(aggregate / name) == digest)

    split_records = {x["split_id"]: x for x in protocol["splits"]}
    ensemble_table = {(x["model_id"], x["split_id"]): x for x in rows(aggregate / "split_seed_ensemble_metrics_random_holdout_v1.csv")}
    spatial_table = {(x["model_id"], x["fold_id"]): x for x in rows(root / "04_results/experiment_v4_aggregate/fold_seed_ensemble_metrics_v4.csv")}
    paired_table = {(x["model_id"], x["split_id"]): x for x in rows(aggregate / "spatial_random_paired_comparison_v1.csv")}
    summary_table = {x["model_id"]: x for x in rows(aggregate / "model_summary_random_holdout_v1.csv")}
    all_aps = {}
    for model in MODELS:
        model_aps = []
        for split_id, record in split_records.items():
            folder = prepared / split_id
            y = np.load(root / "01_data/interim/dataset_v3" / record["source_fold"] / "y.npy", mmap_mode="r")
            test = np.load(folder / "test.npy", mmap_mode="r").astype(bool)
            total = np.zeros(y.shape, dtype=np.float64)
            for seed in SEEDS:
                cell = raw / model / split_id / f"seed_{seed}"
                manifest = json.loads((cell / "run_manifest.json").read_text(encoding="utf-8"))
                check(f"identity:{model}:{split_id}:{seed}",
                      manifest.get("status") == "complete" and manifest.get("protocol_sha256") == protocol_sha and
                      manifest.get("model_id") == model and manifest.get("split_id") == split_id and manifest.get("seed") == seed)
                for name, digest in manifest["file_sha256"].items():
                    check(f"run_hash:{model}:{split_id}:{seed}:{name}", sha256(cell / name) == digest)
                probability = np.load(cell / "probability.npy", mmap_mode="r")
                check(f"probability:{model}:{split_id}:{seed}",
                      probability.shape == y.shape and np.isfinite(probability).all() and probability.min() >= 0 and probability.max() <= 1)
                total += probability
            ensemble = total / 4
            truth = (y[test] == 1).astype(np.int8)
            ap = float(average_precision_score(truth, ensemble[test]))
            auc = float(roc_auc_score(truth, ensemble[test]))
            stored = ensemble_table[(model, split_id)]
            check(f"ensemble_ap:{model}:{split_id}", abs(ap - float(stored["test_average_precision"])) <= 1e-12)
            check(f"ensemble_auc:{model}:{split_id}", abs(auc - float(stored["test_roc_auc"])) <= 1e-12)
            spatial_ap = float(spatial_table[(model, record["source_fold"])]["test_average_precision"])
            paired = paired_table[(model, split_id)]
            check(f"paired_delta:{model}:{split_id}",
                  abs((ap - spatial_ap) - float(paired["random_minus_spatial_ap"])) <= 1e-12)
            model_aps.append(ap)
        all_aps[model] = model_aps
        stored_summary = summary_table[model]
        check(f"summary_mean:{model}",
              abs(float(np.mean(model_aps)) - float(stored_summary["split_ensemble_average_precision_mean"])) <= 1e-12)
        check(f"summary_sd:{model}",
              abs(float(np.std(model_aps, ddof=1)) - float(stored_summary["split_ensemble_average_precision_sd"])) <= 1e-12)

    pair_summary = {x["comparison_id"]: x for x in rows(aggregate / "architecture_pairwise_summary_random_v1.csv")}
    for name, left, right in [("temporal_attn_vs_unet", "temporal_attn", "unet_matched"),
                              ("temporal_attn_vs_concat", "temporal_attn", "temporal_concat")]:
        differences = np.asarray(all_aps[left]) - np.asarray(all_aps[right])
        stored = pair_summary[name]
        check("pair_mean:" + name, abs(float(differences.mean()) - float(stored["mean_ap_difference_left_minus_right"])) <= 1e-12)
        check("pair_sd:" + name, abs(float(differences.std(ddof=1)) - float(stored["sd_ap_difference"])) <= 1e-12)
        check("pair_wins:" + name, int((differences > 0).sum()) == int(stored["left_wins"]) and
              int((differences < 0).sum()) == int(stored["right_wins"]))

    failures = [x for x in checks if not x["passed"]]
    report = {"status": "accepted" if not failures else "rejected",
              "protocol_sha256": protocol_sha, "checks_passed": len(checks) - len(failures),
              "checks_total": len(checks), "failures": failures}
    (aggregate / "independent_verification_random_v1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
