#!/usr/bin/env python
"""Create the leakage-controlled v2 result package.

This script refuses incomplete model groups, recomputes every metric from saved
probability rasters, and gives every design the same four-seed comparison.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common import load, metrics

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "01_data/interim/results_v2"
OUT = ROOT / "06_manuscript/data_package_v2"

GROUPS = {
    "Random forest": "rf_seed{seed}",
    "U-Net (ResNet-18 encoder)": "unet_seed{seed}",
    "Temporal-attention, four seasons, all bands": "tcmt_attn_s4_all_seed{seed}",
    "Season concatenation, four seasons, all bands": "tcmt_concat_s4_all_seed{seed}",
    "Temporal-attention, summer only, all bands": "tcmt_attn_s1_all_seed{seed}",
    "Temporal-attention, four seasons, thermal only": "tcmt_attn_s4_thermal_seed{seed}",
    "Temporal-attention, four seasons, optical only": "tcmt_attn_s4_optical_seed{seed}",
}
SEEDS = (0, 1, 2, 3)
BANDS = ("ST", "red", "nir08", "swir16", "NDVI", "NDBI", "NDWI", "BSI")
SEASONS = ("spring", "summer", "autumn", "winter")
ZONES = {"North": (1250, 1447), "Central": (1447, 1644), "South": (1644, 1840)}


def dump_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_group(pattern: str):
    records, probs = [], []
    for seed in SEEDS:
        stem = pattern.format(seed=seed)
        rp = RES / f"{stem}_result.json"
        pp = RES / f"{stem}_prob_test.npy"
        if not rp.exists() or not pp.exists():
            raise FileNotFoundError(f"Incomplete four-seed group: {rp} / {pp}")
        records.append(json.loads(rp.read_text(encoding="utf-8")))
        probs.append(np.load(pp).astype(np.float64))
    shapes = {p.shape for p in probs}
    if len(shapes) != 1:
        raise ValueError(f"Probability shapes disagree: {sorted(shapes)}")
    return records, probs


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    y, test = d["y"], d["test"]
    eval_mask = test & (y != 2)
    n_test = int(eval_mask.sum())
    n_pos = int(((y == 1) & eval_mask).sum())

    seed_rows, summary_rows, ensemble_results = [], [], {}
    for design, pattern in GROUPS.items():
        records, probs = load_group(pattern)
        recomputed = []
        for seed, record, prob in zip(SEEDS, records, probs):
            m = metrics(y, prob, test, thresh=0.5)
            stored = record["test"]
            for key in ("auc", "ap", "f1", "precision", "recall"):
                if not np.isclose(m[key], stored[key], atol=1e-10):
                    raise ValueError(f"Stored metric mismatch: {pattern} seed {seed} {key}")
            recomputed.append(m)
            seed_rows.append({
                "design": design, "seed": seed,
                "auc": f"{m['auc']:.6f}", "ap": f"{m['ap']:.6f}",
                "f1_at_0.5": f"{m['f1']:.6f}",
                "precision_at_0.5": f"{m['precision']:.6f}",
                "recall_at_0.5": f"{m['recall']:.6f}",
                "best_epoch": record.get("best_epoch", ""),
                "params_M": f"{record['params_M']:.6f}" if "params_M" in record else "",
                "tta": bool(record.get("tta", False)),
                "source": f"01_data/interim/results_v2/{pattern.format(seed=seed)}_result.json",
            })

        ens_prob = np.mean(probs, axis=0)
        ens = metrics(y, ens_prob, test, thresh=0.5)
        ens_stem = pattern.format(seed="ens4").replace("seedens4", "ens4")
        np.save(RES / f"{ens_stem}_prob_test.npy", ens_prob.astype(np.float32))
        ens_record = {
            "design": design, "seeds": list(SEEDS), "aggregation": "arithmetic mean of probabilities",
            "test": ens,
        }
        dump_json(RES / f"{ens_stem}_result.json", ens_record)
        ensemble_results[design] = {"stem": ens_stem, "prob": ens_prob, "metrics": ens}

        aucs = np.array([m["auc"] for m in recomputed])
        aps = np.array([m["ap"] for m in recomputed])
        f1s = np.array([m["f1"] for m in recomputed])
        summary_rows.append({
            "design": design, "n_seeds": len(SEEDS),
            "auc_mean": f"{aucs.mean():.6f}", "auc_sd": f"{aucs.std(ddof=0):.6f}",
            "ap_mean": f"{aps.mean():.6f}", "ap_sd": f"{aps.std(ddof=0):.6f}",
            "f1_mean_at_0.5": f"{f1s.mean():.6f}", "f1_sd_at_0.5": f"{f1s.std(ddof=0):.6f}",
            "ensemble_auc": f"{ens['auc']:.6f}", "ensemble_ap": f"{ens['ap']:.6f}",
            "ensemble_f1_at_0.5": f"{ens['f1']:.6f}",
            "interpretation": "above chance" if ens["auc"] > 0.5 else "not above chance",
        })

    summary_rows.sort(key=lambda r: float(r["ensemble_auc"]), reverse=True)
    write_csv(OUT / "results_all_seeds.csv", seed_rows)
    write_csv(OUT / "results_summary.csv", summary_rows)

    # Spatial diagnostics are descriptive only; they are not independent tests.
    zonal_rows = []
    for design, result in ensemble_results.items():
        for zone, (r0, r1) in ZONES.items():
            mask = test.copy()
            mask[:r0] = False
            mask[r1:] = False
            m = metrics(y, result["prob"], mask, thresh=0.5)
            zonal_rows.append({
                "design": design, "zone": zone, "rows": f"[{r0},{r1})",
                "n": m["n"], "positive": m["pos"],
                "auc": f"{m['auc']:.6f}", "ap": f"{m['ap']:.6f}",
            })
    write_csv(OUT / "zonal_diagnostics.csv", zonal_rows)

    # RF importance names for the 32 direct and 24 context features.
    rf_records, _ = load_group(GROUPS["Random forest"])
    context_channels = list(range(4)) + list(range(8, 12)) + list(range(16, 20))
    imp_rows = []
    for record in rf_records:
        for rank, (idx, value) in enumerate(record["top_importances"], start=1):
            if idx < 32:
                b, s = divmod(idx, 4)
                feature = f"{BANDS[b]}_{SEASONS[s]}"
            else:
                ctx_i, stat_i = divmod(idx - 32, 2)
                base_idx = context_channels[ctx_i]
                b, s = divmod(base_idx, 4)
                feature = f"{BANDS[b]}_{SEASONS[s]}_ctx_{('mean','sd')[stat_i]}"
            imp_rows.append({"seed": record["seed"], "rank": rank, "feature_index": idx,
                             "feature": feature, "importance": f"{value:.8f}"})
    write_csv(OUT / "rf_top_importances.csv", imp_rows)

    class_counts = {
        "positive": int(((y == 1) & d["valid"]).sum()),
        "ignored_annulus": int(((y == 2) & d["valid"]).sum()),
        "background": int(((y == 0) & d["valid"]).sum()),
    }
    data_stats = {
        "grid": {"rows": int(y.shape[0]), "columns": int(y.shape[1]), "pixel_m": 100,
                 "crs": "EPSG:32613"},
        "source": "Google Earth Engine LANDSAT/LC08/C02/T1_L2 and LANDSAT/LC09/C02/T1_L2",
        "years": "2014-2025", "seasons": list(SEASONS), "scientific_bands": list(BANDS),
        "n_model_channels": 32,
        "count_band_role": "validity mask only; excluded from all model features",
        "valid_pixels": int(d["valid"].sum()), "valid_fraction": float(d["valid"].mean()),
        "label_target": "pixels within 200 m of QFFD segments whose linetype is Inferred",
        "label_semantics": "inferred or approximately located mapped trace; not evidence that a fault is concealed",
        "ignore_annulus_m": [200, 1000], "valid_class_counts": class_counts,
        "split_rows": {"train": [[0, 500], [550, 1050]], "validation": [1050, 1200],
                       "test": [1250, 1840], "buffers": [[500, 550], [1200, 1250]]},
        "normalization": "channelwise z-score fitted on valid training rows only",
        "test_evaluation_pixels": n_test, "test_positive_pixels": n_pos,
        "test_positive_rate": n_pos / n_test,
        "metric_scope": "descriptive held-out spatial test; no pixelwise confidence intervals",
    }
    dump_json(OUT / "data_stats.json", data_stats)

    params_by_design = {}
    for design, pattern in GROUPS.items():
        records, _ = load_group(pattern)
        params_by_design[design] = {
            "seeds": list(SEEDS),
            "parameter_count_M": records[0].get("params_M"),
            "tta": bool(records[0].get("tta", False)),
            "result_stems": [pattern.format(seed=s) for s in SEEDS],
        }
    method_params = {
        "deep_models_common": {
            "patch_pixels": 128,
            "patches_per_epoch": 4096,
            "batch_size": 16,
            "optimizer": "Adam",
            "initial_learning_rate": 0.0001,
            "scheduler": "CosineAnnealingLR, T_max=30",
            "maximum_epochs": 30,
            "early_stopping": "patience 5 on validation-best F1",
            "loss": "BCEWithLogitsLoss",
            "positive_weight": 6.3435448,
            "augmentation": "independent random horizontal and vertical flips",
            "inference": "128-pixel window, 64-pixel stride, overlap averaging",
            "test_time_augmentation": "horizontal-flip probability average",
            "checkpoint_selection": "highest validation-best F1; test never used for selection",
            "determinism": "torch/numpy/cuda seeded; deterministic algorithms requested; DataLoader workers=0",
        },
        "random_forest": {
            "n_estimators": 200,
            "training_pixels": int(rf_records[0]["n_train"]),
            "background_to_positive_sampling_ratio": 3.0,
            "features": "32 direct scientific channels + 7x7 mean/sd on ST, NIR, and NDVI in four seasons (56 total)",
            "count_feature_included": False,
        },
        "designs": params_by_design,
        "evaluation": {
            "primary": "ROC AUC",
            "secondary": ["average precision", "F1 at fixed threshold 0.5"],
            "test_threshold_tuning": False,
            "uncertainty_reporting": "four-seed dispersion; no pixelwise confidence interval",
        },
    }
    dump_json(OUT / "method_params.json", method_params)

    best = summary_rows[0]
    fact_lines = [
        "# 第二版可复核事实表",
        "",
        "> 仅记录修复后管线能够支持的事实；旧版结果与旧稿不得继续作为结论来源。",
        "",
        "## 研究目标边界",
        "",
        "- 监督目标是 QFFD 中 `linetype=Inferred` 线段周围 200 m 像元。`Inferred` 表示断层迹线位置为推断或近似定位，不能等同于隐伏、埋藏或未知断层。",
        "- 模型评估的是这一代理标签的空间判别能力，不构成新断层发现、地质确认或灾害判定。",
        "",
        "## 数据与隔离",
        "",
        f"- 影像为 2014–2025 年 Landsat 8/9 Collection 2 Level 2 的四季年度中位数再聚合，共 32 个科学输入通道；采集次数只用于有效掩膜。",
        f"- 有效像元 {data_stats['valid_pixels']:,}（{data_stats['valid_fraction']:.2%}）。测试评估像元 {n_test:,}，其中正样本 {n_pos:,}（{n_pos/n_test:.2%}）。",
        "- 标准化统计量仅来自训练带；128×128 训练切片完整落在单一训练带内；验证与测试之间保留 5 km 缓冲。",
        "",
        "## 统一四种子结果",
        "",
        "| 设计 | AUC 均值±SD | AP 均值±SD | 四种子集成 AUC | 集成 AP |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        fact_lines.append(
            f"| {row['design']} | {float(row['auc_mean']):.3f}±{float(row['auc_sd']):.3f} | "
            f"{float(row['ap_mean']):.3f}±{float(row['ap_sd']):.3f} | "
            f"{float(row['ensemble_auc']):.3f} | {float(row['ensemble_ap']):.3f} |"
        )
    fact_lines += [
        "",
        f"- 按四种子集成 AUC 排名最高的是“{best['design']}”（AUC={float(best['ensemble_auc']):.3f}）。",
        "- 模型间差异及随机种子波动必须同时报告；这些结果不支持把旧版的高分模型指定为稳定主模型。",
        "- AUC=0.5、AP=测试正样本率是随机排序参照；不得把 F1@0.5 解释为经验证的部署阈值。",
        "",
        "## 文件来源",
        "",
        "- `results_all_seeds.csv`：逐种子、从概率栅格重算的测试指标。",
        "- `results_summary.csv`：四种子均值、总体标准差和概率平均集成指标。",
        "- `zonal_diagnostics.csv`：测试区三个子带的描述性诊断。",
        "- `data_stats.json`：数据、标签、分区和评价口径。",
        "- `method_params.json`：训练、推理、随机森林和评价参数。",
    ]
    (OUT / "fact_sheet.md").write_text("\n".join(fact_lines) + "\n", encoding="utf-8")
    print(f"Wrote verified v2 package to {OUT}")
    for row in summary_rows:
        print(f"{row['design']}: mean AUC={row['auc_mean']}±{row['auc_sd']}; ensemble={row['ensemble_auc']}")


if __name__ == "__main__":
    main()
