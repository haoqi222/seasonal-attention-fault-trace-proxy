# Seasonal attention and spatial evaluation design for Landsat-based ranking of mapped fault-trace proxies

**Data and code release v1.0.0 — frozen protocols, aggregated results, publication figures and the public input archive**

This repository is a self-contained data release for a four-fold spatial audit of Landsat-based
models that rank pixels by proximity to mapped fault traces in the central Rio Grande Rift, New
Mexico. It contains the executable code, the frozen experiment protocols, the aggregated metrics
of every analysis, and the public input archive used to build the label proxy. Every number under
`results/` can be re-derived from `code/` under the protocols in `configs/`.

> This release is versioned independently of any publication. Cite it with `CITATION.cff`
> (see *How to cite* below).

---

## 1. What this release is for

The release documents one narrow, falsifiable question: **when pixels are ranked by their proximity
to mapped fault traces, how much of the reported model performance is an artefact of the evaluation
design rather than of the model architecture?**

The answer recorded here is that most of it is an artefact. A random-pixel holdout on the same
data yields average precision above 0.999 for every model, while four geographically blocked folds
put all models at or below the prevalence reference in the southernmost fold. Everything needed to
inspect, re-run and re-derive those statements is included.

**The proxy target is a map artefact by construction.** A positive label means "within 200 m of a
USGS Quaternary Fault and Fold Database (QFFD) segment whose `linetype` field equals `Inferred`".
It is *not* evidence that a fault is concealed, buried, active, or newly discovered, and the code
and documentation here preserve that limitation everywhere it matters.

---

## 2. Repository layout

| Path | Content |
|---|---|
| `code/modeling/` | Frozen V3 training/inference implementation: run matrix, model definitions, dataset loader, random-holdout protocol, scheduler. |
| `code/analysis/` | Validation, aggregation, diagnostic and figure/table generation scripts (V3 and V4.2 entry points). |
| `code/data_prep/` | Data acquisition: Google Earth Engine seasonal composites (`gee/`), USGS M2M scene inventory (`usgs/`), Copernicus DEM GLO-30 via MPC STAC (`mpc/`). |
| `code/legacy_v1/` | Earlier V1 implementations kept for provenance (cross-modal and temporal CMT training, U-Net/RF baselines, dataset packaging). Not used for the reported results. |
| `configs/` | Frozen experiment protocols (JSON): training, inference V4/V4.1/V4.2, random holdout, input masking, coordinate baseline, F4 transfer diagnostic, M7 covariate audit, NLCD acquisition record. |
| `results/` | Aggregated metrics and audit outputs of every analysis (per-run per-seed metrics, fold-level ensembles, model summaries, pairwise differences, split design, normalisation statistics, random-holdout and input-mask diagnostics, coordinate-only control, F4 transfer diagnostics, M7 covariate audit, prediction-seam audit, data package description). |
| `data/` | The public USGS QFFD download used to build the proxy labels, with its SHA-256 checksum. |

---

## 3. Data sources

| Dataset | Role | Access | Reference / checksum |
|---|---|---|---|
| **USGS Quaternary Fault and Fold Database (QFFD), 2020 release** | Source of mapped fault traces; `linetype = Inferred` filtered to build the positive proxy | `data/Qfaults_GIS.zip` (included, 32.4 MB) | U.S. Geological Survey (2020), https://doi.org/10.5066/P9BCVRCK; archive SHA-256 `447eadc5926256710d988c30e5996ba540604caa0154f9746c48bad637926893` |
| **Landsat 8/9 Collection 2 Level 2, 2014–2025** | Four seasonal medians per year: land-surface temperature and optical/indices | Google Earth Engine, `LANDSAT/LC08/C02/T1_L2` and `LANDSAT/LC09/C02/T1_L2`; build script in `code/data_prep/gee/` | Gorelick et al. (2017) |
| **Copernicus DEM GLO-30** | Cartographic hillshade for Figure 1 only; not a model predictor | Copernicus Data Space Ecosystem, MPC STAC `cop-dem-glo-30`; client in `code/data_prep/mpc/` | European Space Agency (n.d.), https://doi.org/10.5270/ESA-c5d3d65 |
| **NHDPlus HR, NetworkNHDFlowline layer 3** | Rio Grande distance covariate (auxiliary M7 audit) | USGS National Map | U.S. Geological Survey (2025) |
| **NLCD 2021 (Esri Living Atlas ImageServer export)** | Land-cover covariate (auxiliary M7 audit) | Public ImageServer export; acquisition record with service URL, bbox and SHA-256 in `configs/m7_nlcd_acquisition_record_v1.json` | U.S. Geological Survey (2026) |

All inputs are public. The derived seasonal composites, the 32-channel model stack, the trained
checkpoints and the out-of-fold probability rasters are **not** redistributed in this repository;
they are rebuilt by the scripts in `code/data_prep/` (see *Reproduction pipeline*).

---

## 4. Design in brief

* **Grid.** 1840 × 539 cells at 100 m, EPSG:32613 (WGS 84 / UTM 13N), 33.70–35.35 °N, 106.55–107.10 °W.
* **Channels.** 8 variables per season (ST, red, NIR, SWIR1, NDVI, NDBI, NDWI, BSI) × 4 seasons = **32 model channels**, in band-major, season-minor order. The per-pixel observation-count band is used for the validity mask only and is never a feature.
* **Validity.** A cell is valid only when the summed ST count exceeds zero in all four seasons: 936,253 cells (94.40 %).
* **Labels.** Within 200 m of a retained QFFD `Inferred` trace → positive (67,280 cells); 200–1,000 m → *ignore* annulus, excluded from loss and from every metric (178,188 cells); beyond 1,000 m → background (690,785 cells).
* **Spatial folds.** Four contiguous north–south row blocks separated by 50-row (5 km) buffers: B1 [0, 400), B2 [450, 800), B3 [850, 1200), B4 [1250, 1840), with buffers [400, 450), [800, 850), [1200, 1250). Any `fault_id` influencing a test block within 1,000 m is purged from the corresponding training and validation sets. Channel-wise z-score statistics are fitted on the training rows of each fold only.
* **Protocols (SHA-256).** V3 training protocol `7e9844b006fc6024cd02b332a23576eb9e070492e113f514561fdebd67a58569`; V4.2 inference protocol `db3c420bc3f8e64ee3c5ff06da7b62e12f858d0a165e74557a3017eb3231b7fe`; M7 NLCD acquisition record `4ed4a60d9f25ba0dffcdeb0bba3941fa5f4f513ba3610a3e20114e48364a6434`.
* **Run matrix.** 8 models × 4 folds × 4 seeds = 128 deep/random-forest cells, plus the random-pixel holdout diagnostic, the input-masking control, the coordinate-only negative control and the M7 covariate audit.

---

## 5. Reproduction pipeline

```bash
# 1. data preparation (GEE composite, QFFD labels, DEM)   code/data_prep/
python gee/02_gee_seasonal_composite.py          # 4 seasonal composites, 8 bands + count
python gee/03_build_labels.py                    # QFFD Inferred traces -> 200 m proxy labels
python mpc/01_dem_prep.py                        # Copernicus DEM -> 100 m, hillshade

# 2. split design, masks, normalisation            code/analysis/
python 11_design_spatial_folds_v3.py             # fold/block selection + audit
python modeling/11_build_fold_masks_v3.py        # per-pixel fold masks
python modeling/12_fit_fold_normalization_v3.py  # per-fold channel statistics

# 3. training and inference (frozen protocol)      code/modeling/
python 14_schedule_experiments_v3.py             # 128-cell resumable scheduler
python 15_rerun_inference_v4.py                  # V4.2 weighted re-inference

# 4. validation, diagnostics, figures and tables   code/analysis/
python 29_validate_and_aggregate_results_v4.py
python 31_generate_formal_figures_tables_v4.py
```

Reference environment (workstation, frozen for the reported runs): Python 3.12.3, `torch 2.11.0+cu128`
on 2 × NVIDIA GeForce RTX 5090 (32 GB), `numpy 2.5.2`, `rasterio 1.5.1`, `geopandas 1.1.4`,
`scikit-learn 1.9.0`, `earthengine-api 1.7.41`, `pandas 3.0.5`, `matplotlib 3.11.1`, `scipy 1.18.1`,
`xarray 2026.7.0`. Exact versions are listed in `requirements.txt`.

The last step writes the figure and table package into the project workspace. Those assets are not
redistributed in this release; the scripts that produce them are included, and every number they
contain is available in the CSV files under `results/`.

---

## 6. Key results

| Comparison | Value | Source |
|---|---|---|
| Equal-weight four-fold mean AP, temporal attention | 0.2301 ± 0.2072 | `results/experiment_v4_aggregate/model_summary_v4.csv` |
| Equal-weight four-fold mean AP, seasonal concatenation | 0.2215 ± 0.1887 | same |
| Equal-weight four-fold mean AP, capacity-matched U-Net | 0.2106 ± 0.2058 | same |
| Attention vs. U-Net, mean paired difference | +0.0195 ± 0.0376 (leads in 3 of 4 folds) | `results/experiment_v4_aggregate/pairwise_summary_v4.csv` |
| Attention vs. seasonal concatenation | +0.0086 ± 0.0213 (leads in 1 of 4 folds) | same |
| Matched random-pixel holdout AP, all three primary models | 0.9994–0.9998 (exceeds the paired spatial result in 12/12 comparisons) | `results/random_holdout_v1_1_aggregate/` |
| Input-masked random-holdout ensemble AP | 0.9284–0.9998 (ceiling persists without direct test-pixel spectra) | `results/random_holdout_input_mask_v1/` |
| Coordinate-only control (row/column logistic regression) | reproduces 37 %, 66 %, 79 % of the attention lift in F1–F3 | `results/coordinate_baseline_v1/` |
| F4 (southernmost fold) | all primary models below the 0.0269 prevalence reference; 22/32 channel associations reverse (11 material) | `results/f4_transfer_diagnostic_v1/` |
| Residual score–label association after river-distance/NLCD control | 0.2337, 0.1965, 0.0492, −0.0146 for F1–F4 | `results/m7_covariate_audit_v1/` |

---

## 7. Not redistributed

The following are intentionally **not** in this repository, because they are either large, derived,
or reproducible from the scripts above:

* seasonal composite GeoTIFFs (4 × 71 MB) and the 32-channel model stack (127 MB) — rebuild with `code/data_prep/gee/02_gee_seasonal_composite.py`;
* trained checkpoints (≈ 7.8 MB per cell, 15 GB including per-cell probability arrays) — regenerate with the frozen protocol; checksums of the reported runs are in `results/experiment_v4_aggregate/aggregate_checksums_v4.sha256`;
* per-pixel out-of-fold probability rasters — the released derivative is the percentile map, which is
  regenerated from the scripts in `code/analysis/`;
* the release figure set (Figures 1–6 and Figure S1, PNG/SVG/Visio) and the reference-table package
  as a standalone table set — regenerate them with `code/analysis/31_generate_formal_figures_tables_v4.py`
  and `code/analysis/40_generate_evaluation_design_figure_v5.py`; every number they contain is
  present in the CSV files under `results/` listed in the sections above.

## 8. How to cite

> haoqi222 (2026). *Seasonal attention and spatial evaluation design for Landsat-based ranking of
> mapped fault-trace proxies* — code, frozen protocols, aggregated results and public data,
> version 1.0.0. <https://github.com/haoqi222/seasonal-attention-fault-trace-proxy>

Version 1.0.0 of this repository is released under the tag `v1.0.0`; cite the tag together with the
commit hash reported by `git log -1` for the copy you used. `CITATION.cff` holds the
machine-readable form.

## 9. Licence

* **Code** (everything under `code/` and `configs/`): MIT — see `LICENSE`.
* **Results** (`results/`): CC BY 4.0.
* **`data/Qfaults_GIS.zip`**: U.S. Government work, public domain (U.S. Geological Survey).

## 10. Issues and contact

Please open an issue for questions about protocol interpretation, label semantics or reproduction
failures. Note that the repository tracks the *frozen* protocols; changes to the protocol files
change their SHA-256 and break the link with the reported run matrix.

---

## 11. Release note: de-identification

The released copies of the code, configuration, results and audit records are de-identified. The
internal project identifier, the workstation project root, the local workspace path and the
internal document file names have been replaced with neutral placeholders such as
`<workstation_project_root>`. Three artefact categories changed as a consequence:

* `schema_version` strings changed from `<project>-…` to `fault-proxy-…`;
* the scheduler launch phrases lost the project prefix (for example `START_V3_128_LOCKED_CELLS`);
* the SHA-256 of every released file changed, so the protocol hashes quoted in this README and
  inside `configs/` refer to the **released** copies, and the checksum manifests under `results/`
  and `results/` were recomputed for them.

The directories under `results/inference_v4_preflight_reports/` and
`results/random_holdout_input_mask_v1/` are named after the SHA-256 of the protocol they belong
to, and were renamed accordingly.

Stage records concerned with document assembly, figure insertion and file delivery are not part of
this release, and the per-stage working records are not distributed either. The protocol freeze,
the acceptance criteria and the verification of the reported runs are documented by the frozen
protocol files in `configs/` and by the acceptance records and checksum manifests under
`results/`.

The release figure set and the reference-table package are not redistributed either. The scripts
that produce them are in `code/analysis/`, and the underlying numbers are the CSV files under
`results/` cited in *Key results* and *Not redistributed*.
