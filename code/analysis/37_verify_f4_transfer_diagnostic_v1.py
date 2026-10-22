#!/usr/bin/env python3
"""Independent verification of the frozen F4 transfer diagnostic."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


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
    output = root / "04_results/f4_transfer_diagnostic_v1"
    protocol_path = root / "03_configs/f4_transfer_diagnostic_v1_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256(protocol_path)
    summary = json.loads((output / "summary_f4_transfer_diagnostic_v1.json").read_text(encoding="utf-8"))
    checks = []

    def check(name: str, passed: bool, **evidence) -> None:
        checks.append({"name": name, "passed": bool(passed), **evidence})

    checksum_lines = (output / "checksums_f4_transfer_v1.sha256").read_text(encoding="ascii").splitlines()
    check("checksum_count", len(checksum_lines) == 8, count=len(checksum_lines))
    for line in checksum_lines:
        digest, name = line.split("  ", 1)
        check("hash:" + name, sha256(output / name) == digest)
    check("protocol_identity", summary["protocol_sha256"] == protocol_sha and
          summary["fold_id"] == protocol["fold_id"])

    fold = root / "01_data/interim/dataset_v3/F4_SouthTest"
    y = np.load(fold / "y.npy", mmap_mode="r")
    train = np.load(fold / "train.npy", mmap_mode="r").astype(bool)
    test = np.load(fold / "test.npy", mmap_mode="r").astype(bool)
    X_source = np.load(root / "01_data/interim/dataset_v2/X.npy", mmap_mode="r")
    norm = json.loads((root / "01_data/interim/normalization_v3/F4_SouthTest/norm_stats.json").read_text(encoding="utf-8"))
    X = (
        np.asarray(X_source, dtype=np.float32)
        - np.asarray(norm["mean"], dtype=np.float32)
    ) / np.asarray(norm["std"], dtype=np.float32)
    meta = json.loads((root / "01_data/interim/dataset_v2/meta.json").read_text(encoding="utf-8"))
    channels = [f"{band}_{season}" for band in meta["bands"] for season in meta["seasons"]]
    reversal_table = {r["channel"]: r for r in rows(output / "feature_association_reversals_f4.csv")}
    shift_table = {r["channel"]: r for r in rows(output / "feature_shift_train_vs_test_f4.csv")}
    reversals = material = 0
    shifts = []
    for index, channel in enumerate(channels):
        train_auc = float(roc_auc_score((y[train] == 1).astype(np.int8), X[..., index][train]))
        test_auc = float(roc_auc_score((y[test] == 1).astype(np.int8), X[..., index][test]))
        reversed_direction = (train_auc - 0.5) * (test_auc - 0.5) < 0
        material_direction = reversed_direction and abs(train_auc - 0.5) >= 0.05 and abs(test_auc - 0.5) >= 0.05
        reversals += reversed_direction; material += material_direction
        stored = reversal_table[channel]
        check("association:" + channel,
              abs(train_auc - float(stored["train_feature_auc"])) <= 1e-12 and
              abs(test_auc - float(stored["test_feature_auc"])) <= 1e-12 and
              (stored["auc_direction_reversed"] == "True") == reversed_direction and
              (stored["material_reversal"] == "True") == material_direction)
        train_feature, test_feature = X[..., index][train], X[..., index][test]
        shift = float((test_feature.mean() - train_feature.mean()) / train_feature.std())
        shifts.append((abs(shift), channel, shift))
        check("shift:" + channel,
              abs(shift - float(shift_table[channel]["test_minus_train_mean_in_f4_train_sd"])) <= 1e-7)
    check("reversal_counts", reversals == summary["auc_direction_reversals"] and
          material == summary["material_auc_direction_reversals"], reversals=reversals, material=material)
    largest = max(shifts)
    check("largest_shift", largest[1] == summary["largest_absolute_standardized_test_shift"]["channel"] and
          abs(largest[2] - summary["largest_absolute_standardized_test_shift"]["test_minus_train_mean_in_f4_train_sd"]) <= 1e-7)

    correlation_table = {(r["model_id"], r["variable"]): r for r in rows(output / "model_score_correlations_f4.csv")}
    row_grid = np.indices(y.shape)[0][test]
    for model in protocol["models"]:
        total = np.zeros(y.shape, dtype=np.float64)
        for seed in protocol["model_seeds"]:
            cell = (
                root
                / "01_data/interim/results_v4"
                / protocol["v4_2_protocol_sha256"]
                / model
                / "F4_SouthTest"
                / f"seed_{seed}/test_probability.npy"
            )
            total += np.load(cell, mmap_mode="r")
        probability = total / 4
        truth, score = (y[test] == 1).astype(np.int8), probability[test]
        ap, auc = float(average_precision_score(truth, score)), float(roc_auc_score(truth, score))
        stored = summary["models"][model]
        check("model_metrics:" + model,
              abs(ap - stored["average_precision"]) <= 1e-12 and abs(auc - stored["roc_auc"]) <= 1e-12)
        rho = float(spearmanr(score, row_grid).statistic)
        check("row_correlation:" + model,
              abs(rho - float(correlation_table[(model, "row_coordinate")]["spearman_rho"])) <= 1e-12)

    failures = [item for item in checks if not item["passed"]]
    report = {"status": "accepted" if not failures else "rejected", "protocol_sha256": protocol_sha,
              "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures}
    (output / "independent_verification_f4_transfer_v1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
