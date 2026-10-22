#!/usr/bin/env python3
"""Run the frozen four-fold coordinate-only negative control."""

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

PROTOCOL = Path("03_configs/coordinate_baseline_v1_protocol.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def evaluate(y: np.ndarray, score: np.ndarray) -> dict:
    truth = (y == 1).astype(np.int8)
    predicted = (score >= 0.5).astype(np.int8)
    return {
        "n": int(truth.size),
        "positive": int(truth.sum()),
        "prevalence": float(truth.mean()),
        "average_precision": float(average_precision_score(truth, score)),
        "roc_auc": float(roc_auc_score(truth, score)),
        "precision_at_0_5": float(precision_score(truth, predicted, zero_division=0)),
        "recall_at_0_5": float(recall_score(truth, predicted, zero_division=0)),
        "f1_at_0_5": float(f1_score(truth, predicted, zero_division=0)),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256(protocol_path)
    if protocol["status"] != "frozen_before_analysis":
        raise RuntimeError("Coordinate baseline protocol is not frozen")
    output = root / protocol["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)

    metric_rows: list[dict] = []
    coefficient_rows: list[dict] = []
    fold_summaries: list[dict] = []
    for fold_id in protocol["folds"]:
        data = load_fold(fold_id, project_root=root, normalize=False, verify_hashes=True)
        y = np.asarray(data["y"])
        roles = {"train": np.asarray(data["train"], dtype=bool),
                 "validation": np.asarray(data["val"], dtype=bool),
                 "test": np.asarray(data["test"], dtype=bool)}
        rows_grid, cols_grid = np.indices(y.shape)
        coordinates = np.column_stack((rows_grid.ravel(), cols_grid.ravel())).astype(np.float64)
        flat_y = y.ravel()
        flat_roles = {role: mask.ravel() for role, mask in roles.items()}
        scaler = StandardScaler().fit(coordinates[flat_roles["train"]])
        scaled = scaler.transform(coordinates)
        transformers = {"coordinate_linear": None,
                        "coordinate_polynomial_degree2": PolynomialFeatures(degree=2, include_bias=False)}
        fold_summary = {"fold_id": fold_id}
        for model_id, transformer in transformers.items():
            if transformer is None:
                features = scaled
                feature_names = ["row", "column"]
            else:
                features = transformer.fit_transform(scaled)
                feature_names = list(transformer.get_feature_names_out(["row", "column"]))
            cfg = protocol["model"]
            model = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"],
                                       solver=cfg["solver"], max_iter=cfg["max_iter"],
                                       random_state=cfg["random_state"])
            train_mask = flat_roles["train"]
            model.fit(features[train_mask], (flat_y[train_mask] == 1).astype(np.int8))
            for role, mask in flat_roles.items():
                scores = model.predict_proba(features[mask])[:, 1]
                metrics = evaluate(flat_y[mask], scores)
                metric_rows.append({"fold_id": fold_id, "model_id": model_id, "role": role, **metrics})
                if role == "test":
                    fold_summary[model_id] = metrics
            for feature_name, coefficient in zip(feature_names, model.coef_[0]):
                coefficient_rows.append({"fold_id": fold_id, "model_id": model_id,
                                         "feature": feature_name, "coefficient": float(coefficient)})
        fold_summaries.append(fold_summary)

    write_csv(output / "coordinate_baseline_metrics_v1.csv", metric_rows)
    write_csv(output / "coordinate_baseline_coefficients_v1.csv", coefficient_rows)
    summary = {
        "status": "completed",
        "protocol_sha256": protocol_sha,
        "fold_count": len(protocol["folds"]),
        "model_count": len(protocol["models"]),
        "test_summary_by_fold": fold_summaries,
        "interpretation_boundaries": protocol["interpretation_boundaries"],
    }
    (output / "summary_coordinate_baseline_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# Coordinate baseline V1\n\n"
        "Four-fold descriptive row/column negative control with training-only standardization. "
        "Coordinates are not remote-sensing evidence and no causal decomposition is permitted.\n",
        encoding="utf-8", newline="\n")
    names = sorted(p.name for p in output.iterdir() if p.is_file() and p.name != "checksums_coordinate_baseline_v1.sha256")
    (output / "checksums_coordinate_baseline_v1.sha256").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in names), encoding="ascii", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
