#!/usr/bin/env python3
"""Independently recompute the four-fold coordinate-only negative control."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
from dataset_common_v3 import load_fold  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def evaluate(y: np.ndarray, score: np.ndarray) -> dict:
    truth = (y == 1).astype(np.int8); predicted = (score >= 0.5).astype(np.int8)
    return {"n": int(truth.size), "positive": int(truth.sum()), "prevalence": float(truth.mean()),
            "average_precision": float(average_precision_score(truth, score)),
            "roc_auc": float(roc_auc_score(truth, score)),
            "precision_at_0_5": float(precision_score(truth, predicted, zero_division=0)),
            "recall_at_0_5": float(recall_score(truth, predicted, zero_division=0)),
            "f1_at_0_5": float(f1_score(truth, predicted, zero_division=0))}


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "04_results/coordinate_baseline_v1"
    protocol_path = root / "03_configs/coordinate_baseline_v1_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")); protocol_sha = sha256(protocol_path)
    stored = {(row["fold_id"], row["model_id"], row["role"]): row for row in rows(output / "coordinate_baseline_metrics_v1.csv")}
    checks: list[dict] = []
    def check(name: str, passed: bool, **evidence) -> None: checks.append({"name": name, "passed": bool(passed), **evidence})
    lines = (output / "checksums_coordinate_baseline_v1.sha256").read_text(encoding="ascii").splitlines()
    check("checksum_count", len(lines) == 4, count=len(lines))
    for line in lines:
        digest, name = line.split("  ", 1); check("hash:" + name, sha256(output / name) == digest)
    summary = json.loads((output / "summary_coordinate_baseline_v1.json").read_text(encoding="utf-8"))
    check("protocol_identity", summary["protocol_sha256"] == protocol_sha)
    recomputed = 0
    for fold_id in protocol["folds"]:
        data = load_fold(fold_id, project_root=root, normalize=False, verify_hashes=True)
        y = np.asarray(data["y"]); rows_grid, cols_grid = np.indices(y.shape)
        coordinates = np.column_stack((rows_grid.ravel(), cols_grid.ravel())).astype(np.float64)
        flat_y = y.ravel(); roles = {"train": np.asarray(data["train"], dtype=bool).ravel(),
                                      "validation": np.asarray(data["val"], dtype=bool).ravel(),
                                      "test": np.asarray(data["test"], dtype=bool).ravel()}
        scaler = StandardScaler().fit(coordinates[roles["train"]]); scaled = scaler.transform(coordinates)
        for model_id, transformer in (("coordinate_linear", None),
                                      ("coordinate_polynomial_degree2", PolynomialFeatures(degree=2, include_bias=False))):
            features = scaled if transformer is None else transformer.fit_transform(scaled)
            cfg = protocol["model"]
            model = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], solver=cfg["solver"],
                                       max_iter=cfg["max_iter"], random_state=cfg["random_state"])
            model.fit(features[roles["train"]], (flat_y[roles["train"]] == 1).astype(np.int8))
            for role, mask in roles.items():
                current = evaluate(flat_y[mask], model.predict_proba(features[mask])[:, 1]); expected = stored[(fold_id, model_id, role)]
                ok = all(abs(float(expected[key]) - float(current[key])) <= 1e-12 for key in
                         ["n", "positive", "prevalence", "average_precision", "roc_auc", "precision_at_0_5", "recall_at_0_5", "f1_at_0_5"])
                check(f"metrics:{fold_id}:{model_id}:{role}", ok); recomputed += 1
    check("metric_cell_count", recomputed == 24, count=recomputed)
    failures = [item for item in checks if not item["passed"]]
    report = {"status": "accepted" if not failures else "rejected", "protocol_sha256": protocol_sha,
              "checks_passed": len(checks) - len(failures), "checks_total": len(checks), "failures": failures}
    (output / "independent_verification_coordinate_baseline_v1.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures: raise SystemExit(1)


if __name__ == "__main__": main()
