#!/usr/bin/env python3
"""Analyze F4 transfer failure under a frozen descriptive diagnostic protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "modeling"))
from dataset_common_v3 import load_fold  # noqa: E402


PROTOCOL = Path("03_configs/f4_transfer_diagnostic_v1_protocol.json")


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


def classification(y: np.ndarray, score: np.ndarray) -> dict:
    truth = (y == 1).astype(np.int8)
    return {"n": int(len(truth)), "positive": int(truth.sum()), "prevalence": float(truth.mean()),
            "average_precision": float(average_precision_score(truth, score)),
            "roc_auc": float(roc_auc_score(truth, score))}


def ensemble(root: Path, v4_sha: str, model: str, fold: str) -> np.ndarray:
    total = None
    for seed in range(4):
        value = np.load(root / "01_data/interim/results_v4" / v4_sha / model / fold /
                        f"seed_{seed}/test_probability.npy", mmap_mode="r")
        if total is None:
            total = np.zeros(value.shape, dtype=np.float64)
        total += value
    return total / 4


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    protocol_path = root / PROTOCOL
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_sha = sha256(protocol_path)
    if protocol["status"] != "frozen_before_analysis":
        raise RuntimeError("F4 diagnostic protocol is not frozen")
    output = root / protocol["output"]
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True)
    fold = protocol["fold_id"]
    data = load_fold(fold, project_root=root, normalize=True, verify_hashes=True)
    X, y = data["X"], data["y"]
    roles = {"train": data["train"], "validation": data["val"], "test": data["test"]}
    channels = data["channel_names"]
    probabilities = {model: ensemble(root, protocol["v4_2_protocol_sha256"], model, fold)
                     for model in protocol["models"]}

    shift_rows = []
    train_values = X[roles["train"]]
    test_values = X[roles["test"]]
    for index, channel in enumerate(channels):
        train_channel, test_channel = train_values[:, index], test_values[:, index]
        ks = ks_2samp(train_channel, test_channel, alternative="two-sided", method="asymp")
        shift_rows.append({"channel_index": index, "channel": channel,
                           "train_mean": float(train_channel.mean()), "train_sd": float(train_channel.std()),
                           "test_mean": float(test_channel.mean()), "test_sd": float(test_channel.std()),
                           "test_minus_train_mean_in_f4_train_sd": float((test_channel.mean() - train_channel.mean()) / train_channel.std()),
                           "ks_statistic": float(ks.statistic), "ks_pvalue_descriptive_only": float(ks.pvalue)})
    write_csv(output / "feature_shift_train_vs_test_f4.csv", shift_rows)

    association_rows = []
    association_lookup = {}
    for role, mask in roles.items():
        truth = (y[mask] == 1).astype(np.int8)
        values = X[mask]
        for index, channel in enumerate(channels):
            feature = values[:, index]
            positive, negative = feature[truth == 1], feature[truth == 0]
            pooled = math.sqrt(((len(positive) - 1) * positive.var(ddof=1) +
                                (len(negative) - 1) * negative.var(ddof=1)) /
                               max(len(positive) + len(negative) - 2, 1))
            auc = float(roc_auc_score(truth, feature))
            row = {"role": role, "channel_index": index, "channel": channel,
                   "n": len(truth), "positive": int(truth.sum()), "feature_auc": auc,
                   "positive_mean": float(positive.mean()), "negative_mean": float(negative.mean()),
                   "standardized_positive_minus_negative": float((positive.mean() - negative.mean()) / pooled)}
            association_rows.append(row); association_lookup[(role, channel)] = row
    write_csv(output / "feature_class_association_by_role_f4.csv", association_rows)
    reversal_rows = []
    for channel in channels:
        train, test = association_lookup[("train", channel)], association_lookup[("test", channel)]
        reversal = (train["feature_auc"] - 0.5) * (test["feature_auc"] - 0.5) < 0
        material = reversal and abs(train["feature_auc"] - 0.5) >= 0.05 and abs(test["feature_auc"] - 0.5) >= 0.05
        reversal_rows.append({"channel": channel, "train_feature_auc": train["feature_auc"],
                              "test_feature_auc": test["feature_auc"],
                              "train_standardized_contrast": train["standardized_positive_minus_negative"],
                              "test_standardized_contrast": test["standardized_positive_minus_negative"],
                              "auc_direction_reversed": reversal, "material_reversal": material})
    write_csv(output / "feature_association_reversals_f4.csv", reversal_rows)

    test_mask = roles["test"]
    rows_grid, cols_grid = np.indices(y.shape)
    correlates = [("row_coordinate", rows_grid[test_mask]), ("column_coordinate", cols_grid[test_mask])]
    correlates += [(channel, X[..., index][test_mask]) for index, channel in enumerate(channels)]
    correlation_rows = []
    for model, probability in probabilities.items():
        score = probability[test_mask]
        for name, values in correlates:
            correlation, pvalue = spearmanr(score, values)
            correlation_rows.append({"model_id": model, "variable": name,
                                     "spearman_rho": float(correlation),
                                     "pvalue_descriptive_only": float(pvalue)})
    write_csv(output / "model_score_correlations_f4.csv", correlation_rows)

    block_rows = []
    for start in range(1250, 1840, 50):
        end = min(start + 50, 1840)
        block = test_mask.copy()
        block[:start] = False; block[end:] = False
        block_y = y[block]
        if not np.any(block_y == 1) or not np.any(block_y == 0):
            continue
        for model, probability in probabilities.items():
            block_rows.append({"model_id": model, "row_start": start, "row_end": end,
                               **classification(block_y, probability[block])})
    write_csv(output / "model_metrics_by_5km_block_f4.csv", block_rows)

    coords = np.column_stack((rows_grid.ravel(), cols_grid.ravel())).astype(np.float64)
    flat_y = y.ravel()
    flat_masks = {role: mask.ravel() for role, mask in roles.items()}
    scaler = StandardScaler().fit(coords[flat_masks["train"]])
    scaled = scaler.transform(coords)
    coordinate_rows = []
    for name, transformer in (("coordinate_linear", None),
                              ("coordinate_polynomial_degree2", PolynomialFeatures(degree=2, include_bias=False))):
        features = scaled if transformer is None else transformer.fit_transform(scaled)
        cfg = protocol["coordinate_model"]
        model = LogisticRegression(C=cfg["C"], class_weight=cfg["class_weight"], solver=cfg["solver"],
                                   max_iter=cfg["max_iter"], random_state=cfg["random_state"])
        model.fit(features[flat_masks["train"]], (flat_y[flat_masks["train"]] == 1).astype(np.int8))
        for role in ("train", "validation", "test"):
            mask = flat_masks[role]
            score = model.predict_proba(features[mask])[:, 1]
            coordinate_rows.append({"model_id": name, "role": role,
                                    **classification(flat_y[mask], score)})
    write_csv(output / "coordinate_baselines_f4.csv", coordinate_rows)

    model_summary = {}
    for model, probability in probabilities.items():
        overall = classification(y[test_mask], probability[test_mask])
        model_blocks = [row for row in block_rows if row["model_id"] == model]
        model_summary[model] = {**overall,
            "blocks_with_both_classes": len(model_blocks),
            "blocks_auc_below_0_5": sum(row["roc_auc"] < 0.5 for row in model_blocks),
            "block_auc_mean": float(np.mean([row["roc_auc"] for row in model_blocks])),
            "block_auc_sd": float(np.std([row["roc_auc"] for row in model_blocks], ddof=1))}
    summary = {"status": "completed", "protocol_sha256": protocol_sha, "fold_id": fold,
               "test_prevalence": float((y[test_mask] == 1).mean()),
               "feature_channels": len(channels),
               "auc_direction_reversals": sum(row["auc_direction_reversed"] for row in reversal_rows),
               "material_auc_direction_reversals": sum(row["material_reversal"] for row in reversal_rows),
               "largest_absolute_standardized_test_shift": max(shift_rows, key=lambda row: abs(row["test_minus_train_mean_in_f4_train_sd"])),
               "models": model_summary, "interpretation_boundaries": protocol["interpretation_boundaries"]}
    (output / "summary_f4_transfer_diagnostic_v1.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        "# F4 transfer diagnostic V1\n\nDescriptive audit of feature shift, association reversal, "
        "5 km block dispersion, model-score correlates, and coordinate-only controls. Causal attribution is prohibited.\n",
        encoding="utf-8", newline="\n")
    names = sorted(p.name for p in output.iterdir() if p.is_file() and p.name != "checksums_f4_transfer_v1.sha256")
    (output / "checksums_f4_transfer_v1.sha256").write_text(
        "".join(f"{sha256(output / name)}  {name}\n" for name in names), encoding="ascii", newline="\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
