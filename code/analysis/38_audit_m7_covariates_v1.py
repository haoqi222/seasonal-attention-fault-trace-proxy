#!/usr/bin/env python3
"""M7 descriptive audit: Rio Grande distance and NLCD-2021 covariates.

This is an evaluation-only audit.  It does not retrain a model, select a
checkpoint, or reinterpret proxy scores as geological probabilities.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import rasterio
from scipy.stats import spearmanr
from shapely import distance as shapely_distance
from shapely import points as shapely_points
from shapely.geometry import shape
from shapely.ops import unary_union
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[2]
V4_SHA = "43315ba42f2686b339644e0e89afd964bab32ef4647aa9189bfd4eea96118b1b"
V3_SHA = "7e9844b006fc6024cd02b332a23576eb9e070492e113f514561fdebd67a58569"
M7_AMENDMENT = "03_configs/m7_river_nlcd_covariate_v1_1_amendment.json"
SHAPE = (1840, 539)
MODELS = {
    "unet_matched": ("v4", "primary_capacity_matched"),
    "temporal_attn": ("v4", "primary_capacity_matched"),
    "temporal_concat": ("v4", "primary_capacity_matched"),
    "rf_direct": ("v3", "primary_nonneural_baseline"),
    "rf_context": ("v3", "secondary_context_sensitivity"),
    "summer_only": ("v4", "secondary_input_ablation"),
    "thermal_only": ("v4", "secondary_input_ablation"),
    "optical_only": ("v4", "secondary_input_ablation"),
}
FOLDS = ["F1_NorthTest", "F2_NorthCentralTest", "F3_SouthCentralTest", "F4_SouthTest"]
NLCD_LABELS = {
    11: "Open water", 21: "Developed, open space", 22: "Developed, low intensity",
    23: "Developed, medium intensity", 24: "Developed, high intensity",
    31: "Barren land", 41: "Deciduous forest", 42: "Evergreen forest",
    43: "Mixed forest", 52: "Shrub/scrub", 71: "Grassland/herbaceous",
    81: "Pasture/hay", 82: "Cultivated crops", 90: "Woody wetlands",
    95: "Emergent herbaceous wetlands",
}
DISTANCE_BINS_KM = [(0, 5), (5, 10), (10, 20), (20, 40), (40, 80), (80, np.inf)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for row in rows for k in row})
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def river_distance_raster(river_path: Path, template_path: Path, out_path: Path) -> np.ndarray:
    geo = json.loads(river_path.read_text(encoding="utf-8"))
    union = unary_union([shape(feat["geometry"]) for feat in geo["features"]])
    with rasterio.open(template_path) as ds:
        transform = ds.transform
        height, width = ds.height, ds.width
        xs = transform.c + (np.arange(width) + 0.5) * transform.a
        ys = transform.f + (np.arange(height) + 0.5) * transform.e
        out = np.empty((height, width), dtype=np.float32)
        for start in range(0, height, 100):
            stop = min(start + 100, height)
            xx, yy = np.meshgrid(xs, ys[start:stop])
            pts = shapely_points(xx.ravel(), yy.ravel())
            out[start:stop] = np.asarray(shapely_distance(pts, union), dtype=np.float32).reshape(stop - start, width)
        profile = ds.profile.copy()
        profile.update(dtype="float32", count=1, nodata=-9999.0, compress="deflate", predictor=3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, "Euclidean distance to NHDPlus HR Rio Grande flowlines (m)")
    return out


def ensemble_probability(model: str, fold: str) -> np.ndarray:
    tier, _ = MODELS[model]
    root = ROOT / "01_data/interim" / ("results_v4" if tier == "v4" else "results_v3") / (V4_SHA if tier == "v4" else V3_SHA)
    arrays = [np.load(root / model / fold / f"seed_{seed}" / "test_probability.npy", mmap_mode="r") for seed in range(4)]
    return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float32)


def safe_ap(y: np.ndarray, score: np.ndarray) -> float:
    return float(average_precision_score(y, score)) if y.size and np.unique(y).size == 2 else float("nan")


def partial_corr(y: np.ndarray, score: np.ndarray, distance_km: np.ndarray, classes: np.ndarray) -> float:
    """Residual Pearson association after distance + NLCD one-hot adjustment."""
    cls = np.unique(classes)
    x = [np.ones(y.size), np.log1p(distance_km)]
    for code in cls[1:]:
        x.append((classes == code).astype(float))
    design = np.column_stack(x)
    ry = y - design @ np.linalg.lstsq(design, y, rcond=None)[0]
    rs = score - design @ np.linalg.lstsq(design, score, rcond=None)[0]
    den = float(np.linalg.norm(ry) * np.linalg.norm(rs))
    return float(np.dot(ry, rs) / den) if den else float("nan")


def main() -> None:
    external = ROOT / "01_data/external"
    river_path = external / "nhdplus_hr_rio_grande.geojson"
    nlcd_path = external / "nlcd_2021/nlcd_2021_100m_epsg32613.tif"
    valid = np.load(ROOT / "01_data/interim/dataset_v2/valid.npy").astype(bool)
    if valid.shape != SHAPE:
        raise ValueError(f"valid shape {valid.shape} != {SHAPE}")
    cov_dir = ROOT / "01_data/interim/m7_covariates_v1"
    dist_path = cov_dir / "distance_rio_grande_m.tif"
    distance_m = river_distance_raster(river_path, nlcd_path, dist_path) if not dist_path.exists() else rasterio.open(dist_path).read(1)
    with rasterio.open(nlcd_path) as ds:
        nlcd = ds.read(1)
        transform = tuple(ds.transform)
        crs = str(ds.crs)
    if nlcd.shape != SHAPE or distance_m.shape != SHAPE:
        raise ValueError("covariate shape mismatch")

    summary, bins, class_rows, partial_rows = [], [], [], []
    for fold in FOLDS:
        fold_dir = ROOT / "01_data/interim/dataset_v3" / fold
        y = np.load(fold_dir / "y.npy").astype(np.uint8)
        test = np.load(fold_dir / "test.npy").astype(bool)
        base = valid & test & np.isfinite(distance_m) & (nlcd > 0)
        yb = y[base]
        dkm = distance_m[base].astype(float) / 1000.0
        cb = nlcd[base].astype(int)
        random_ap = float(np.mean(yb)) if yb.size else float("nan")
        for model, (_, tier) in MODELS.items():
            score = ensemble_probability(model, fold)[base].astype(float)
            ap = safe_ap(yb, score)
            rho_d, p_d = spearmanr(score, dkm) if score.size > 2 else (float("nan"), float("nan"))
            rho_y, p_y = spearmanr(score, yb) if score.size > 2 else (float("nan"), float("nan"))
            pc = partial_corr(yb.astype(float), score, dkm, cb) if score.size > 20 and np.unique(cb).size > 1 else float("nan")
            summary.append({"fold": fold, "model": model, "analysis_tier": tier, "n_test_valid_nlcd": int(base.sum()),
                           "positive_rate": random_ap, "ap": ap, "ap_minus_random": ap - random_ap,
                           "spearman_score_distance_km": float(rho_d), "spearman_p_distance": float(p_d),
                           "spearman_score_label": float(rho_y), "spearman_p_label": float(p_y),
                           "partial_residual_corr_score_label_distance_nlcd": pc})
            for lo, hi in DISTANCE_BINS_KM:
                m = (dkm >= lo) & (dkm < hi)
                if not np.any(m):
                    continue
                bins.append({"fold": fold, "model": model, "distance_bin_km": f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">={lo:g}",
                             "n": int(m.sum()), "positive_rate": float(yb[m].mean()), "mean_score": float(score[m].mean()),
                             "ap": safe_ap(yb[m], score[m])})
            for code in sorted(np.unique(cb)):
                m = cb == code
                class_rows.append({"fold": fold, "model": model, "nlcd_code": int(code), "nlcd_label": NLCD_LABELS.get(int(code), "Unknown"),
                                   "n": int(m.sum()), "positive_rate": float(yb[m].mean()), "mean_score": float(score[m].mean()),
                                   "ap": safe_ap(yb[m], score[m])})
            partial_rows.append({"fold": fold, "model": model, "n": int(base.sum()), "random_ap": random_ap, "ap": ap,
                                 "partial_corr": pc, "distance_median_km": float(np.median(dkm)),
                                 "distance_q25_km": float(np.quantile(dkm, .25)), "distance_q75_km": float(np.quantile(dkm, .75))})

    out = ROOT / "04_results/m7_covariate_audit_v1"
    write_csv(out / "m7_fold_model_summary.csv", summary)
    write_csv(out / "m7_distance_bins.csv", bins)
    write_csv(out / "m7_nlcd_classes.csv", class_rows)
    write_csv(out / "m7_partial_association.csv", partial_rows)
    manifest = {"status": "descriptive_audit_complete", "scope": "no retraining; no checkpoint selection; no geological probability interpretation",
                "m7_protocol_amendment": M7_AMENDMENT,
                "v4_protocol_sha256": V4_SHA, "v3_protocol_sha256_for_rf": V3_SHA, "shape": SHAPE, "crs": crs,
                "nlcd_transform": transform, "sources": {str(river_path.relative_to(ROOT)): sha256(river_path), str(nlcd_path.relative_to(ROOT)): sha256(nlcd_path), str(dist_path.relative_to(ROOT)): sha256(dist_path)},
                "models": MODELS, "folds": FOLDS}
    (out / "m7_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "summary_rows": len(summary), "distance_bins": len(bins), "nlcd_rows": len(class_rows), "out": str(out)}, indent=2))


if __name__ == "__main__":
    main()
