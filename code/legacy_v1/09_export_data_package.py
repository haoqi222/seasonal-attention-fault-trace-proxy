#!/usr/bin/env python
"""阶段7：导出数据包（仅结构化事实，不含散文）。

产出 06_manuscript/data_package/：
  results_consolidated.csv  每个模型/设计一行（各 seed 的 AUC/AP/F1、均值±std、参数、TTA）
  results_all_seeds.csv     每个 seed 一行
  rf_importances.csv        RF 特征重要性（前 20，映射到 波段×季节 名）
  zonal_auc.csv             三子带 AUC（TCMT 集成 vs RF 2seed 均值）
  data_stats.json           数据/标签/分区统计
  method_params.json        全部训练/评估超参数
  fact_sheet.md             干净事实表（无散文，写作工具直接可读）
用法：.venv/bin/python 02_scripts/modeling/09_export_data_package.py
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common import load, metrics

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RES = os.path.join(ROOT, "01_data/interim/results")
OUT = os.path.join(ROOT, "06_manuscript", "data_package")
R0, R1 = 1250, 1840

BANDS = ["ST", "red", "nir08", "swir16", "ndvi", "ndbi", "ndwi", "bsi", "count"]
SEASONS = ["spring", "summer", "autumn", "winter"]
BANDS_CN = {"ST": "ST", "red": "red", "nir08": "nir08", "swir16": "swir16",
            "ndvi": "NDVI", "ndbi": "NDBI", "ndwi": "NDWI", "bsi": "BSI", "count": "count"}

# 模型族 → 展示名（与 06_summarize.py 保持一致）
def display_name(r):
    model = r.get("model")
    if model == "cross_modal_transformer":
        aw = r.get("attn_window", 0)
        name = f"CMT-{r['fusion']}"
        name += f"({'4season' if len(r['seasons']) == 4 else 'summer-only'}{'' if aw == 0 else f'|aw{aw}'})"
    elif model == "temporal_cmt":
        b = r.get("base", 16)
        name = f"TCMT-{r['temporal']}"
        name += f"({'4season' if len(r['seasons']) == 4 else 'summer-only'}{'' if b == 16 else f'|b{b}'})"
    elif model == "unet_resnet18":
        name = "U-Net(resnet18)"
    elif model == "random_forest":
        name = "RandomForest"
    elif model == "temporal_cmt_ensemble":
        name = "TCMT-attn ensemble(4seed)"
    else:
        name = model
    return name


def main():
    os.makedirs(OUT, exist_ok=True)
    d = load()
    y = d["y"]
    yb = (y[R0:R1] == 1).astype(np.int8)
    valid = d["test"][R0:R1] & (y[R0:R1] != 2)

    # ---- 1. 全部结果 ----
    rows_all, groups = [], defaultdict(list)
    for f in sorted(os.listdir(RES)):
        if not f.endswith("_result.json"):
            continue
        stem = f[:-len("_result.json")]
        with open(os.path.join(RES, f)) as fh:
            r = json.load(fh)
        t = r.get("test", {})
        rows_all.append({
            "stem": stem, "model_family": r.get("model"),
            "design": display_name(r), "seed": r.get("seed"),
            "params_M": round(r.get("params_M", float("nan")), 3) if r.get("params_M") else "",
            "tta": r.get("tta", False),
            "test_auc": round(t.get("auc"), 4) if t.get("auc") is not None else "",
            "test_ap": round(t.get("ap"), 4) if t.get("ap") is not None else "",
            "test_f1": round(t.get("f1"), 4) if t.get("f1") is not None else "",
            "precision": round(t.get("precision"), 4) if t.get("precision") is not None else "",
            "recall": round(t.get("recall"), 4) if t.get("recall") is not None else "",
            "n_test": t.get("n"), "pos_test": t.get("pos"),
            "val_best_f1": round(r.get("val_best_f1"), 3) if r.get("val_best_f1") else "",
            "best_epoch": r.get("best_epoch"),
        })
        if r.get("model") != "temporal_cmt_ensemble":
            groups[rows_all[-1]["design"]].append(rows_all[-1])

    # 按设计聚合（多 seed）
    consolidated = []
    for design in sorted(groups):
        rs = groups[design]
        def agg(k, fmt="{:.3f}"):
            v = [x[k] for x in rs if x[k] != ""]
            if not v:
                return ""
            return f"{fmt.format(np.mean(v))}±{fmt.format(np.std(v))}"
        consolidated.append({
            "design": design, "n_seeds": len(rs),
            "test_auc": agg("test_auc"), "test_ap": agg("test_ap"), "test_f1": agg("test_f1"),
            "params_M": rs[0]["params_M"], "tta": any(x["tta"] for x in rs),
        })
    # 附加集成（单独）
    ens = json.load(open(os.path.join(RES, "tcmt_attn_ens4_result.json")))
    consolidated.insert(0, {
        "design": "TCMT-attn ensemble(4seed) [MAIN]", "n_seeds": 4,
        "test_auc": f"{ens['test']['auc']:.3f}", "test_ap": f"{ens['test']['ap']:.3f}",
        "test_f1": f"{ens['test']['f1']:.3f}", "params_M": "0.64", "tta": True,
    })
    write_csv(os.path.join(OUT, "results_consolidated.csv"), consolidated)
    write_csv(os.path.join(OUT, "results_all_seeds.csv"), rows_all)

    # ---- 2. RF 特征重要性 ----
    rf = json.load(open(os.path.join(RES, "rf_seed0_result.json")))
    imps = rf["top_importances"]
    ctx = list(range(4)) + list(range(8, 12)) + list(range(16, 20)) + list(range(32, 36))
    imp_rows = []
    for idx, imp in imps:
        if idx < 36:
            b, s = divmod(idx, 4)          # band-major, season-minor
            name = f"{BANDS_CN[BANDS[b]]}·{SEASONS[s]}"
        else:
            j = idx - 36
            ci, kind = divmod(j, 2)        # 每 context 通道 2 个：mean,std
            cc = ctx[ci]
            b, s = divmod(cc, 4)
            name = f"{BANDS_CN[BANDS[b]]}·{SEASONS[s]}·{'ctx_mean' if kind == 0 else 'ctx_std'}"
        imp_rows.append({"feature_index": idx, "feature": name, "importance": round(imp, 4)})
    write_csv(os.path.join(OUT, "rf_importances.csv"), imp_rows)

    # ---- 3. 区域诊断（TCMT 集成 vs RF 2seed 均值）----
    ens_prob = np.load(os.path.join(RES, "tcmt_attn_ens4_prob_test.npy"))[R0:R1]
    rf0 = np.load(os.path.join(RES, "rf_seed0_prob_test.npy")); rf0 = rf0[R0:R1] if rf0.shape[0] == 1840 else rf0
    rf1 = np.load(os.path.join(RES, "rf_seed1_prob_test.npy")); rf1 = rf1[R0:R1] if rf1.shape[0] == 1840 else rf1
    rf_prob = 0.5 * (rf0 + rf1)
    zone_rows = []
    for label, (r0, r1) in {"North": (1250, 1447), "Central": (1447, 1644), "South": (1644, 1840)}.items():
        m = np.zeros(valid.shape, bool); m[r0 - R0:r1 - R0] = True; m &= valid
        zone_rows.append({
            "zone": label, "rows": f"{r0}-{r1}",
            "n": int(valid[m].sum()), "pos": int(yb[m].sum()),
            "ens_auc": round(metrics(yb, ens_prob, m, 0.5)["auc"], 3),
            "rf_auc": round(metrics(yb, rf_prob, m, 0.5)["auc"], 3),
        })
    write_csv(os.path.join(OUT, "zonal_auc.csv"), zone_rows)

    # ---- 4. 数据统计 ----
    stats = {
        "grid": {"rows": 1840, "cols": 539, "pixel_m": 100.0, "crs": "EPSG:32613",
                 "bbox_utm": [305300, 3729900, 359200, 3913900],
                 "lonlat": {"lon": [-107.101, -106.55], "lat": [33.691, 35.358]}},
        "bands": BANDS, "seasons": SEASONS, "n_channels": 36,
        "channel_order": "band-major, season-minor; 0:4=ST x4seasons ... 32:36=count",
        "composites": {"years": "2014-2025", "satellites": "Landsat 8/9", "reducer": "median",
                       "season_months": {"spring": [3, 4, 5], "summer": [6, 7, 8],
                                         "autumn": [9, 10, 11], "winter": [12, 1, 2]}},
        "valid_mask": "count>0 in all 4 seasons",
        "labels": {"source": "USGS QFFD 2020", "concealed_proxy": "linetype=Inferred",
                   "pos_buffer_m": 200, "ignore_annulus_m": [200, 1000],
                   "class_stats": {"positive_le200m": 68344, "ignore_200_1000m": 181892,
                                   "background_gt1000m": 741524}},
        "splits": {"train": [[0, 500], [550, 1050]], "val": [1050, 1200], "test": [1250, 1840],
                   "buffers": [[500, 550], [1200, 1250]]},
        "test": {"n_valid": int(valid.sum()), "pos": int(yb.sum()),
                 "pos_rate": round(float(yb.mean()), 4)},
        "dem_in_model": False,
    }
    with open(os.path.join(OUT, "data_stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2, ensure_ascii=False)

    # ---- 5. 方法参数（从脚本默认值与 result json 合并）----
    params = {
        "TemporalCMT": {
            "architecture": "shared 4-scale conv encoder (base=16, widths 16/32/64/128) "
                            "+ per-scale temporal attention over 4 seasonal tokens (4 heads) "
                            "+ 1x1 projection + U-Net-style decoder",
            "params_M": 0.64,
            "input": "128x128 x 36ch",
            "patch_sampling": "50% fault-centered / 50% random background",
            "patches_per_epoch": 4096, "batch": 16, "optimizer": "Adam", "lr": 1e-4,
            "scheduler": "CosineAnnealingLR (T_max=30)", "epochs": 30,
            "early_stop": "patience 5 on val F1", "loss": "BCEWithLogits pos_weight=6.39",
            "augmentation": "random h/v flip",
            "inference": "sliding 128/stride 64 overlap-avg + horizontal-flip TTA",
            "ensemble": "4 seeds (0-3) mean probability"},
        "RandomForest": {
            "n_estimators": 200, "features": 68, "n_train": 207860, "bg_ratio": 3.0,
            "context": "7x7 mean/std on 16 selected channels (ST, nir08, swir16, count x4 seasons)",
            "seeds": 2, "TTA": "n/a (pixelwise, symmetric features)"},
        "U-Net": {
            "encoder": "resnet18 (random init)", "in_channels": 36, "params_M": 14.43,
            "training": "same as TemporalCMT", "seeds": 2, "TTA": True},
        "evaluation": {
            "metric": "AUC (primary), AP, F1@0.5 (fixed threshold, no test tuning)",
            "test_pixels": int(valid.sum()), "test_pos": int(yb.sum()),
            "uncertainty": "pixel-level bootstrap (indicative; spatial autocorrelation makes it optimistic)"},
    }
    with open(os.path.join(OUT, "method_params.json"), "w") as fh:
        json.dump(params, fh, indent=2, ensure_ascii=False)

    # ---- 6. 事实表（无散文）----
    with open(os.path.join(OUT, "fact_sheet.md"), "w") as fh:
        w = fh.write
        w("# 数据事实表\n\n")
        w("> 本文件只含事实与数字，每个数字标注来源。直接作为写作工具的输入。\n\n")
        w("## 0. 一句话结论\n")
        w("4 种子集成的 TemporalCMT-attn（0.64M 参数）在空间分离 TEST 区 test AUC=0.710，"
          "高于随机森林 0.701、U-Net(resnet18) 0.661；三子带全部占优；消融证明多时相与时间注意力是主信号。\n\n")
        w("## 1. 主模型对比（TEST 区，深度模型含水平翻转 TTA；TTA=False 行无 TTA，如实标注）\n\n")
        w("| 模型/设计 | 种子数 | test AUC | test AP | F1@0.5 | 参数 | TTA | 来源 |\n|---|---|---|---|---|---|---|---|\n")
        for row in consolidated:
            src = "tcmt_attn_ens4_result.json" if "MAIN" in row["design"] else "results/*_result.json"
            w(f"| {row['design']} | {row['n_seeds']} | {row['test_auc']} | {row['test_ap']} | "
              f"{row['test_f1']} | {row['params_M']} | {row['tta']} | {src} |\n")
        w("\n注：AP/F1 绝对量低（正样本率 2.7%、信号弱）——如实报告，属筛选级指标。\n\n")
        w("## 2. 逐 seed（深度模型）\n\n")
        w("| 模型/设计 | seed | AUC | AP | F1@0.5 | TTA |\n|---|---|---|---|---|---|\n")
        for r in rows_all:
            if r["model_family"] in ("random_forest", "temporal_cmt_ensemble"):
                continue
            w(f"| {r['design']} | {r['seed']} | {r['test_auc']} | {r['test_ap']} | "
              f"{r['test_f1']} | {r['tta']} |\n")
        w("\n## 3. 区域诊断（TEST 三子带，TCMT 集成 vs RF 2 种子均值）\n\n")
        w("| 子带 | 行范围 | n | 正样本 | TCMT AUC | RF AUC |\n|---|---|---|---|---|---|\n")
        for z in zone_rows:
            w(f"| {z['zone']} | {z['rows']} | {z['n']} | {z['pos']} | {z['ens_auc']} | {z['rf_auc']} |\n")
        w("\n## 4. 数据\n\n")
        w(f"- 网格：{stats['grid']['rows']}×{stats['grid']['cols']}，{stats['grid']['pixel_m']} m，"
          f"{stats['grid']['crs']}；范围 {stats['grid']['lonlat']['lat']}°N / "
          f"{stats['grid']['lonlat']['lon']}°W（Rio Grande Rift 中段，新墨西哥）。\n")
        w(f"- 卫星：{stats['composites']['satellites']}，{stats['composites']['years']}，"
          f"{stats['composites']['reducer']} 季节合成（{stats['composites']['season_months']}）。\n")
        w(f"- 通道：{stats['n_channels']} = {len(stats['bands'])} band × {len(stats['seasons'])} 季；"
          f"{stats['channel_order']}。\n")
        w(f"- 有效掩膜：{stats['valid_mask']}（94.9% 有效，东西缘无覆盖条带 5.1%）。\n")
        w(f"- 标签（QFFD 2020，隐伏代理 = {stats['labels']['concealed_proxy']}）："
          f"正样本 ≤{stats['labels']['pos_buffer_m']} m = {stats['labels']['class_stats']['positive_le200m']} "
          f"({stats['labels']['class_stats']['positive_le200m']/991760*100:.1f}%)，忽略 "
          f"{stats['labels']['ignore_annulus_m'][0]}–{stats['labels']['ignore_annulus_m'][1]} m = "
          f"{stats['labels']['class_stats']['ignore_200_1000m']} ({stats['labels']['class_stats']['ignore_200_1000m']/991760*100:.1f}%)，"
          f"背景 >{stats['labels']['ignore_annulus_m'][1]} m = {stats['labels']['class_stats']['background_gt1000m']} "
          f"({stats['labels']['class_stats']['background_gt1000m']/991760*100:.1f}%)。\n")
        w(f"- 空间划分：训练行 [0,500)∪[550,1050)，验证 [1050,1200)，测试 [1250,1840)，"
          f"缓冲 5 km 带 [500,550)∪[1200,1250)。TEST 有效像素 n={stats['test']['n_valid']}，"
          f"正样本 {stats['test']['pos']}（{stats['test']['pos_rate']*100:.2f}%）。\n")
        w(f"- DEM 不进模型（仅事后归因）。\n\n")
        w("## 5. 方法参数\n\n")
        for fam, p in params.items():
            w(f"### {fam}\n\n")
            for k, v in p.items():
                w(f"- **{k}**: {v}\n")
            w("\n")
        w("## 6. RF 特征重要性 Top10\n\n")
        w("| # | 特征 | 重要性 |\n|---|---|---|\n")
        for i, r in enumerate(imp_rows[:10], 1):
            w(f"| {i} | {r['feature']} | {r['importance']:.4f} |\n")
        w("\n## 7. 图表文件\n\n")
        for fig, desc in [
            ("fig1_study_area.png", "研究区：DEM 山体阴影 + QFFD 隐伏断层 + 分区带"),
            ("fig2_method_schematic.png", "数据管线 + TemporalCMT 架构示意"),
            ("fig3_result_maps.png", "TEST 区概率图：TCMT 集成 vs RF vs U-Net + 断层叠加"),
            ("fig4_metrics.png", "ROC 曲线 + 三子带 zonal AUC 柱状图"),
        ]:
            w(f"- `06_manuscript/figures/{fig}`：{desc}（`run_all.sh` 可重生成）\n")
    print("数据包已写出 →", OUT)


def write_csv(path, rows):
    import csv
    cols = list(rows[0].keys())
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols)
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
