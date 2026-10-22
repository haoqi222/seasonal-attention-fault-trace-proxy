#!/usr/bin/env python3
"""Recompute M7 partial score-label association with categorical distance bins.

This is an evaluation-only sensitivity analysis. It reuses the accepted
fold/seed probability arrays and frozen labels; it does not retrain models or
select checkpoints. Distance is represented by the same bins used in Figure 6
instead of a monotone log-distance term.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
V4_SHA = "43315ba42f2686b339644e0e89afd964bab32ef4647aa9189bfd4eea96118b1b"
V3_SHA = "7e9844b006fc6024cd02b332a23576eb9e070492e113f514561fdebd67a58569"
SHAPE = (1840, 539)
FOLDS = ["F1_NorthTest", "F2_NorthCentralTest", "F3_SouthCentralTest", "F4_SouthTest"]
MODELS = {
    "unet_matched": "v4", "temporal_attn": "v4", "temporal_concat": "v4",
    "rf_direct": "v3", "rf_context": "v3", "summer_only": "v4",
    "thermal_only": "v4", "optical_only": "v4",
}
BIN_EDGES_KM = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 80.0, np.inf])
BIN_LABELS = ["0-5", "5-10", "10-20", "20-40", "40-80", ">=80"]


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def ensemble_probability(model: str, fold: str) -> np.ndarray:
    tier = MODELS[model]
    root = ROOT / "01_data/interim" / ("results_v4" if tier == "v4" else "results_v3") / (V4_SHA if tier == "v4" else V3_SHA)
    arrays = [np.load(root / model / fold / f"seed_{seed}" / "test_probability.npy", mmap_mode="r") for seed in range(4)]
    return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float32)


def categorical_partial_corr(y: np.ndarray, score: np.ndarray, distance_bin: np.ndarray, classes: np.ndarray) -> float:
    """Pearson association between residuals after distance-bin and NLCD indicators."""
    x = [np.ones(y.size)]
    for code in range(len(BIN_LABELS))[1:]:
        x.append((distance_bin == code).astype(float))
    for code in np.unique(classes)[1:]:
        x.append((classes == code).astype(float))
    design = np.column_stack(x)
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    rs = score - design @ np.linalg.lstsq(design, score, rcond=None)[0]
    den = float(np.linalg.norm(ry) * np.linalg.norm(rs))
    return float(np.dot(ry, rs) / den) if den else float("nan")


def main() -> None:
    valid = np.load(ROOT / "01_data/interim/dataset_v2/valid.npy").astype(bool)
    external = ROOT / "01_data/external"
    import rasterio
    with rasterio.open(ROOT / "01_data/interim/m7_covariates_v1/distance_rio_grande_m.tif") as ds:
        distance_m = ds.read(1)
    with rasterio.open(external / "nlcd_2021/nlcd_2021_100m_epsg32613.tif") as ds:
        nlcd = ds.read(1)
    rows: list[dict] = []
    for fold in FOLDS:
        fold_dir = ROOT / "01_data/interim/dataset_v3" / fold
        y = np.load(fold_dir / "y.npy").astype(np.uint8)
        test = np.load(fold_dir / "test.npy").astype(bool)
        base = valid & test & np.isfinite(distance_m) & (nlcd > 0)
        yb = y[base].astype(float)
        dkm = distance_m[base].astype(float) / 1000.0
        cb = nlcd[base].astype(int)
        db = np.digitize(dkm, BIN_EDGES_KM[1:-1], right=False)
        for model in MODELS:
            score = ensemble_probability(model, fold)[base].astype(float)
            pc = categorical_partial_corr(yb, score, db, cb)
            rho, p = spearmanr(score, dkm)
            rows.append({"fold": fold, "model": model, "n": int(base.sum()),
                         "categorical_distance_nlcd_partial_corr": pc,
                         "spearman_score_distance_km": float(rho),
                         "spearman_p_distance": float(p),
                         "distance_bins": ";".join(BIN_LABELS)})
    out = ROOT / "04_results/m7_covariate_audit_v2_categorical_distance"
    write_csv(out / "m7_partial_association_categorical_distance.csv", rows)
    manifest = {
        "status": "completed",
        "scope": "evaluation-only; no retraining; no checkpoint selection",
        "distance_control": "categorical indicators using Figure 6 distance bins",
        "land_cover_control": "NLCD class indicators",
        "folds": FOLDS,
        "models": list(MODELS),
        "input_m7_distance_raster": "01_data/interim/m7_covariates_v1/distance_rio_grande_m.tif",
        "output": str(out.relative_to(ROOT)),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "rows": len(rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
