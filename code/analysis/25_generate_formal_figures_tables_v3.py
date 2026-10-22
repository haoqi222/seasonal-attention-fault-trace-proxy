#!/usr/bin/env python3
"""Generate the submission-facing V3 figures and tables.

The script reads only the frozen V3 protocol, accepted stage-9 aggregates, and
the formal V3 probability arrays.  It does not retrain models or modify formal
results.  Multi-panel labels are unique and use lower-case parenthetical form.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Patch
from PIL import Image
from sklearn.metrics import precision_recall_curve


PROTOCOL_SHA = "7e9844b006fc6024cd02b332a23576eb9e070492e113f514561fdebd67a58569"
FOLDS = ["F1_NorthTest", "F2_NorthCentralTest", "F3_SouthCentralTest", "F4_SouthTest"]
FOLD_LABELS = ["F1 North", "F2 North-central", "F3 South-central", "F4 South"]
MODELS = [
    "temporal_attn", "temporal_concat", "unet_matched", "summer_only",
    "optical_only", "rf_context", "rf_direct", "thermal_only",
]
MODEL_LABELS = {
    "temporal_attn": "Temporal attention",
    "temporal_concat": "Seasonal concatenation",
    "unet_matched": "U-Net matched",
    "summer_only": "Summer-only",
    "optical_only": "Optical-only",
    "rf_context": "RF context",
    "rf_direct": "RF direct",
    "thermal_only": "Thermal-only",
}
PRIMARY = ["temporal_attn", "temporal_concat", "unet_matched"]
COLORS = {
    "temporal_attn": "#0072B2",
    "temporal_concat": "#D55E00",
    "unet_matched": "#009E73",
    "summer_only": "#CC79A7",
    "optical_only": "#56B4E9",
    "rf_context": "#E69F00",
    "rf_direct": "#7A7A7A",
    "thermal_only": "#F0E442",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configure_font(font_dir: Path) -> str:
    regular = font_dir / "times.ttf"
    if not regular.exists():
        raise FileNotFoundError(f"Times New Roman regular font not found: {regular}")
    for path in sorted(font_dir.glob("times*.ttf")):
        font_manager.fontManager.addfont(path)
    name = font_manager.FontProperties(fname=regular).get_name()
    plt.rcParams.update({
        "font.family": name,
        "font.size": 14,
        "axes.titlesize": 15,
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "axes.linewidth": 1.0,
    })
    return name


def save_figure(fig, output: Path, name: str) -> None:
    fig.savefig(output / name, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_ensemble(result_root: Path, model: str, fold: str) -> np.ndarray:
    arrays = [
        np.load(result_root / model / fold / f"seed_{seed}" / "test_probability.npy", mmap_mode="r")
        for seed in range(4)
    ]
    return np.mean(np.stack(arrays, axis=0), axis=0, dtype=np.float64)


def figure_1_spatial_cv(
    root: Path,
    output: Path,
    name: str = "fig1_v3_spatial_cross_validation_design.png",
) -> None:
    block = None
    label = None
    for index, fold in enumerate(FOLDS, start=1):
        fold_root = root / "01_data/interim/dataset_v3" / fold
        partition = np.load(fold_root / "partition.npy", mmap_mode="r")
        test_geographic_role = np.asarray(partition == 3)
        if block is None:
            block = np.zeros(test_geographic_role.shape, dtype=np.uint8)
            label = np.load(fold_root / "y.npy", mmap_mode="r")
        block[test_geographic_role] = index
    shown = np.ma.masked_where(block == 0, block)
    cmap = ListedColormap(["#4477AA", "#66CCEE", "#CCBB44", "#EE6677"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    # Keep the legend in a dedicated column so it cannot cover the mapped
    # proxy traces.  The wider canvas also remains legible when Word scales the
    # figure to the release text width.
    fig = plt.figure(figsize=(9.2, 10.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.08)
    ax = fig.add_subplot(grid[0, 0])
    legend_ax = fig.add_subplot(grid[0, 1])
    legend_ax.axis("off")
    ax.imshow(shown, cmap=cmap, norm=norm, origin="upper", interpolation="nearest", aspect="equal")
    proxy = np.ma.masked_where(label != 1, np.ones(label.shape, dtype=np.uint8))
    ax.imshow(proxy, cmap=ListedColormap(["#7F0000"]), origin="upper",
              interpolation="nearest", aspect="equal", alpha=0.95)
    ax.set_xlabel("Column (100 m pixels)")
    ax.set_ylabel("Row (100 m pixels)")
    handles = [Patch(facecolor=cmap(i), label=FOLD_LABELS[i]) for i in range(4)]
    handles.append(Line2D([0], [0], color="#7F0000", lw=2, label="QFFD Inferred proxy trace"))
    legend_ax.legend(handles=handles, loc="upper left", frameon=False,
                     borderaxespad=0.0, handlelength=2.4)
    save_figure(fig, output, name)


def add_box(ax, xy, wh, title, detail, color):
    x, y = xy
    w, h = wh
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.016",
                         facecolor=color, edgecolor="#333333", linewidth=1.3)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h * 0.69, title, ha="center", va="center", weight="bold", fontsize=14)
    ax.text(x + w / 2, y + h * 0.31, detail, ha="center", va="center", fontsize=14, linespacing=1.15)


def figure_2_workflow(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(14.5, 7.2), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    add_box(ax, (0.02, 0.62), (0.20, 0.25), "Remote-sensing inputs",
            "Four seasonal composites\n32 channels; 100 m grid", "#D9EDF7")
    add_box(ax, (0.02, 0.15), (0.20, 0.25), "Proxy target",
            "QFFD Inferred traces\npositive / ignore / background", "#F8D7DA")
    add_box(ax, (0.28, 0.39), (0.20, 0.25), "Spatial controls",
            "Four blocked folds\n5 km gaps; train-only scaling", "#FFF3CD")
    add_box(ax, (0.55, 0.68), (0.19, 0.20), "Primary models",
            "Temporal attention\nconcat; matched U-Net", "#D4EDDA")
    add_box(ax, (0.55, 0.40), (0.19, 0.20), "Input ablations",
            "summer-only; optical-only\nthermal-only", "#D4EDDA")
    add_box(ax, (0.55, 0.12), (0.19, 0.20), "RF baselines",
            "direct channels\ncontext features", "#D4EDDA")
    add_box(ax, (0.81, 0.39), (0.17, 0.25), "Frozen evaluation",
            "4 seeds per fold\nseed ensemble; AP\nequal-weight fold summary", "#E2D9F3")
    arrows = [
        ((0.22, 0.74), (0.28, 0.56)), ((0.22, 0.28), (0.28, 0.47)),
        ((0.48, 0.52), (0.55, 0.78)), ((0.48, 0.52), (0.55, 0.50)),
        ((0.48, 0.52), (0.55, 0.22)), ((0.74, 0.78), (0.81, 0.57)),
        ((0.74, 0.50), (0.81, 0.52)), ((0.74, 0.22), (0.81, 0.46)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=18,
                                     color="#333333", linewidth=1.4))
    save_figure(fig, output, "fig2_v3_frozen_evaluation_workflow.png")


def figure_3_fold_performance(fold_rows: list[dict], summary_rows: list[dict], output: Path) -> None:
    by_key = {(r["model_id"], r["fold_id"]): r for r in fold_rows}
    summaries = {r["model_id"]: r for r in summary_rows}
    fig, axes = plt.subplots(1, 2, figsize=(15.8, 6.8), constrained_layout=True)
    x = np.arange(4)
    prevalence = [float(by_key[(PRIMARY[0], fold)]["test_positive_rate"]) for fold in FOLDS]
    for model in PRIMARY:
        values = [float(by_key[(model, fold)]["test_average_precision"]) for fold in FOLDS]
        axes[0].plot(x, values, marker="o", ms=8, lw=2.1, color=COLORS[model], label=MODEL_LABELS[model])
    axes[0].plot(x, prevalence, marker="s", ms=7, lw=1.7, ls="--", color="#222222",
                 label="Random-ranking reference")
    axes[0].set_xticks(x, FOLD_LABELS, rotation=20, ha="right")
    axes[0].set_ylabel("Average precision")
    axes[0].set_ylim(0, 0.58)
    axes[0].set_title("(a) Primary architecture comparison", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(loc="upper right")

    y = np.arange(len(MODELS))[::-1]
    means = [float(summaries[m]["fold_ensemble_average_precision_mean"]) for m in MODELS]
    sds = [float(summaries[m]["fold_ensemble_average_precision_sd"]) for m in MODELS]
    axes[1].errorbar(means, y, xerr=sds, fmt="none", ecolor="#555555", capsize=5, lw=1.5)
    axes[1].scatter(means, y, s=85, c=[COLORS[m] for m in MODELS], edgecolor="black", linewidth=0.5, zorder=3)
    axes[1].set_yticks(y, [MODEL_LABELS[m] for m in MODELS])
    axes[1].set_xlabel("Equal-weight mean AP across four folds")
    axes[1].set_xlim(0, 0.47)
    axes[1].set_title("(b) Fold mean and sample SD", loc="left", weight="bold")
    axes[1].grid(axis="x", alpha=0.25)
    save_figure(fig, output, "fig3_v3_spatial_fold_performance.png")


def figure_4_pr_curves(root: Path, result_root: Path, output: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 11.4), constrained_layout=True)
    for panel_index, (ax, fold, fold_label) in enumerate(zip(axes.flat, FOLDS, FOLD_LABELS)):
        fold_root = root / "01_data/interim/dataset_v3" / fold
        y = np.load(fold_root / "y.npy", mmap_mode="r")
        test = np.load(fold_root / "test.npy", mmap_mode="r").astype(bool)
        mask = test & (y != 2)
        truth = (np.asarray(y[mask]) == 1).astype(np.uint8)
        for model in [*PRIMARY, "rf_direct"]:
            ensemble = load_ensemble(result_root, model, fold)
            precision, recall, _ = precision_recall_curve(truth, ensemble[mask])
            ax.plot(recall, precision, lw=2, color=COLORS[model], label=MODEL_LABELS[model])
        prevalence = float(truth.mean())
        ax.axhline(prevalence, color="#222222", ls="--", lw=1.4,
                   label=f"Random reference ({prevalence:.3f})")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"({chr(97 + panel_index)}) {fold_label}", loc="left", weight="bold")
        ax.grid(alpha=0.20)
        ax.legend(loc="upper right")
    save_figure(fig, output, "fig4_v3_seed_ensemble_pr_curves.png")


def percentile_map(ensemble: np.ndarray, eligible: np.ndarray, geographic: np.ndarray) -> np.ndarray:
    reference = np.sort(np.asarray(ensemble[eligible], dtype=np.float64))
    result = np.full(ensemble.shape, np.nan, dtype=np.float32)
    values = np.asarray(ensemble[geographic], dtype=np.float64)
    result[geographic] = 100.0 * np.searchsorted(reference, values, side="right") / len(reference)
    return result


def figure_5_maps(root: Path, result_root: Path, output: Path) -> None:
    model_maps = {model: None for model in PRIMARY}
    label = None
    for fold in FOLDS:
        fold_root = root / "01_data/interim/dataset_v3" / fold
        y = np.load(fold_root / "y.npy", mmap_mode="r")
        partition = np.load(fold_root / "partition.npy", mmap_mode="r")
        test = np.load(fold_root / "test.npy", mmap_mode="r").astype(bool)
        geographic = np.asarray(partition == 3)
        eligible = test & (y != 2)
        if label is None:
            label = np.asarray(y)
        for model in PRIMARY:
            if model_maps[model] is None:
                model_maps[model] = np.full(y.shape, np.nan, dtype=np.float32)
            ranked = percentile_map(load_ensemble(result_root, model, fold), eligible, geographic)
            model_maps[model][geographic] = ranked[geographic]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 11.5), sharex=True, sharey=True, constrained_layout=True)
    shown = None
    for panel_index, (ax, model) in enumerate(zip(axes, PRIMARY)):
        shown = ax.imshow(model_maps[model], cmap="magma", vmin=0, vmax=100, origin="upper",
                          interpolation="nearest", aspect="equal")
        proxy = np.ma.masked_where(label != 1, label)
        ax.contour(proxy, levels=[0.5], colors="#00FFFF", linewidths=0.65, origin="upper")
        ax.set_title(f"({chr(97 + panel_index)}) {MODEL_LABELS[model]}", loc="left", weight="bold")
        ax.set_xlabel("Column (100 m pixels)")
    axes[0].set_ylabel("Row (100 m pixels)")
    colorbar = fig.colorbar(shown, ax=axes, orientation="horizontal", fraction=0.035, pad=0.045, aspect=55)
    colorbar.set_label("Within-fold ensemble score percentile")
    save_figure(fig, output, "fig5_v3_out_of_fold_percentile_maps.png")


def figure_6_training(training_rows: list[dict], output: Path) -> None:
    deep = [r for r in training_rows if r["model_kind"] == "deep"]
    by_model = {model: [r for r in deep if r["model_id"] == model] for model in MODELS if model not in {"rf_context", "rf_direct"}}
    order = list(by_model)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 6.8), constrained_layout=True)
    data = [[float(r["best_epoch"]) for r in by_model[m]] for m in order]
    box = axes[0].boxplot(data, tick_labels=[MODEL_LABELS[m] for m in order], patch_artist=True,
                          medianprops={"color": "black", "linewidth": 1.5})
    for patch, model in zip(box["boxes"], order):
        patch.set_facecolor(COLORS[model])
        patch.set_alpha(0.75)
    axes[0].tick_params(axis="x", rotation=28)
    for label in axes[0].get_xticklabels():
        label.set_ha("right")
    axes[0].set_ylabel("Best checkpoint epoch")
    axes[0].set_title("(a) Checkpoint timing", loc="left", weight="bold")
    axes[0].grid(axis="y", alpha=0.25)

    bins = np.arange(0.5, 61.5, 1)
    axes[1].hist([int(r["epochs_completed"]) for r in deep], bins=bins, color="#4477AA", edgecolor="white")
    axes[1].axvline(15, color="#D55E00", ls="--", lw=2, label="Minimum training length")
    axes[1].set_xlabel("Completed epochs")
    axes[1].set_ylabel("Run count")
    axes[1].set_title("(b) Training duration across 96 deep runs", loc="left", weight="bold")
    axes[1].legend(loc="upper right")
    axes[1].grid(axis="y", alpha=0.25)
    save_figure(fig, output, "fig6_v3_training_diagnostics.png")


def make_tables(root: Path, output: Path, protocol: dict, fold_rows: list[dict],
                summary_rows: list[dict], pairwise_rows: list[dict]) -> None:
    by_key = {(r["model_id"], r["fold_id"]): r for r in fold_rows}
    table1 = []
    for fold, label in zip(FOLDS, FOLD_LABELS):
        row = by_key[("temporal_attn", fold)]
        table1.append({
            "fold": label,
            "test_pixels": int(row["test_n"]),
            "positive_pixels": int(row["test_positive"]),
            "positive_rate": f"{float(row['test_positive_rate']):.4f}",
            "random_ranking_ap": f"{float(row['test_positive_rate']):.4f}",
        })
    write_csv(output / "table1_v3_spatial_fold_test_sets.csv", table1)

    table2 = []
    for model in protocol["deep_models"]:
        table2.append({
            "model": MODEL_LABELS[model["model_id"]],
            "analysis_tier": model["analysis_tier"],
            "input_or_features": model["input"],
            "trainable_parameters": model["expected_trainable_parameters"],
        })
    for model in protocol["random_forest_models"]:
        table2.append({
            "model": MODEL_LABELS[model["model_id"]],
            "analysis_tier": model["analysis_tier"],
            "input_or_features": model["features"],
            "trainable_parameters": "not applicable",
        })
    write_csv(output / "table2_v3_model_definitions.csv", table2)

    by_model = {r["model_id"]: r for r in summary_rows}
    table3 = []
    for model in MODELS:
        row = by_model[model]
        table3.append({
            "descriptive_rank": int(row["overall_descriptive_rank"]),
            "model": MODEL_LABELS[model],
            "analysis_tier": row["analysis_tier"],
            "average_precision_mean": f"{float(row['fold_ensemble_average_precision_mean']):.4f}",
            "average_precision_sd": f"{float(row['fold_ensemble_average_precision_sd']):.4f}",
            "roc_auc_mean": f"{float(row['fold_ensemble_roc_auc_mean']):.4f}",
            "roc_auc_sd": f"{float(row['fold_ensemble_roc_auc_sd']):.4f}",
            "precision_top_5pct_mean": f"{float(row['fold_ensemble_precision_top_5pct_mean']):.4f}",
            "recall_top_5pct_mean": f"{float(row['fold_ensemble_recall_top_5pct_mean']):.4f}",
        })
    write_csv(output / "table3_v3_model_performance.csv", table3)

    table4 = []
    for row in pairwise_rows:
        table4.append({
            "comparison": f"{MODEL_LABELS[row['left_model']]} minus {MODEL_LABELS[row['right_model']]}",
            "mean_ap_difference": f"{float(row['mean_ap_difference_left_minus_right']):+.4f}",
            "sd_ap_difference": f"{float(row['sd_ap_difference']):.4f}",
            "left_wins": int(row["left_wins"]),
            "ties": int(row["ties"]),
            "right_wins": int(row["right_wins"]),
            "interpretation_boundary": row["interpretation_boundary"],
        })
    write_csv(output / "table4_v3_pairwise_fold_ap_differences.csv", table4)


def write_readme(output: Path, font_name: str) -> None:
    text = "# V3 candidate formal figures and tables\n\n"
    text += "Status: provisional; not cleared for release insertion because Figure 5 contains verified stride-aligned seams.\n\n"
    text += "Source boundary: frozen V3 protocol, accepted stage-9 aggregates, and formal V3 run probabilities only.\n\n"
    text += f"- Font registered for figures: {font_name}; all explicitly configured visible text is at least 14 pt.\n"
    text += "- Raster export: PNG at 300 ppi.\n"
    text += "- Figure files have unique descriptive names; overall titles are left to the caption layer.\n"
    text += "- Multi-panel labels use (a), (b), and so on, once per panel.\n"
    text += "- Figure 5 uses within-fold score percentiles because raw outputs are not calibrated across folds.\n"
    text += "- The cyan overlay in Figure 5 is the QFFD Inferred proxy trace, not independently verified concealed-fault ground truth.\n"
    text += "- Figure 5 is retained as diagnostic evidence and is rejected from the submission set until inference blending is corrected and every deep-model metric is recomputed under one controlled protocol.\n"
    text += "- Tables are flat CSV files intended for conversion to journal-style three-line Word tables.\n"
    (output / "README.md").write_text(text, encoding="utf-8", newline="\n")


def image_qa(output: Path) -> dict:
    figures = sorted(output.glob("fig*.png"))
    records = []
    for path in figures:
        with Image.open(path) as image:
            dpi = image.info.get("dpi", (0, 0))
            records.append({
                "file": path.name,
                "width_pixels": image.width,
                "height_pixels": image.height,
                "dpi_x": float(dpi[0]),
                "dpi_y": float(dpi[1]),
                "dpi_approximately_300": abs(float(dpi[0]) - 300) < 1 and abs(float(dpi[1]) - 300) < 1,
            })
    return {
        "status": "provisional_not_submission_ready",
        "unique_figure_filenames": len({r["file"] for r in records}) == len(records),
        "all_png_approximately_300_dpi": all(r["dpi_approximately_300"] for r in records),
        "configured_minimum_visible_font_pt": 14,
        "panel_label_policy": "lowercase parenthetical labels, unique within each multi-panel figure",
        "accepted_for_release": [
            "fig1_v3_spatial_cross_validation_design.png",
            "fig2_v3_frozen_evaluation_workflow.png",
            "fig3_v3_spatial_fold_performance.png",
            "fig4_v3_seed_ensemble_pr_curves.png",
        ],
        "supplementary_candidate": ["fig6_v3_training_diagnostics.png"],
        "rejected_pending_inference_correction": ["fig5_v3_out_of_fold_percentile_maps.png"],
        "blocking_reason": "Deep-model maps show visually apparent and quantitatively verified 64-pixel stride-aligned seams.",
        "figures": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--font-dir", type=Path, default=Path("/tmp/times"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (args.output or root / "05_figures_tables/formal_v3").resolve()
    output.mkdir(parents=True, exist_ok=True)
    aggregate = root / "04_results/experiment_v3_aggregate"
    result_root = root / "01_data/interim/results_v3" / PROTOCOL_SHA
    protocol = json.loads((root / "03_configs/experiment_v3_protocol.json").read_text(encoding="utf-8"))
    fold_rows = read_csv(aggregate / "fold_seed_ensemble_metrics_v3.csv")
    summary_rows = read_csv(aggregate / "model_summary_v3.csv")
    pairwise_rows = read_csv(aggregate / "pairwise_summary_v3.csv")
    training_rows = read_csv(aggregate / "training_diagnostics_v3.csv")
    font_name = configure_font(args.font_dir)

    figure_1_spatial_cv(root, output)
    figure_2_workflow(output)
    figure_3_fold_performance(fold_rows, summary_rows, output)
    figure_4_pr_curves(root, result_root, output)
    figure_5_maps(root, result_root, output)
    figure_6_training(training_rows, output)
    make_tables(root, output, protocol, fold_rows, summary_rows, pairwise_rows)
    write_readme(output, font_name)
    qa = image_qa(output)
    (output / "figure_qa_v3.json").write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "status": "provisional_not_submission_ready",
        "protocol_sha256": PROTOCOL_SHA,
        "source_aggregate_acceptance": "04_results/experiment_v3_aggregate/stage9_acceptance_v3.json",
        "figure_count": len(list(output.glob("fig*.png"))),
        "table_count": len(list(output.glob("table*.csv"))),
        "submission_blocker": qa["blocking_reason"],
        "files": [p.name for p in sorted(output.iterdir()) if p.is_file() and p.name not in {"formal_v3_checksums.sha256", "formal_v3_manifest.json"}],
    }
    (output / "formal_v3_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    produced = sorted(p for p in output.iterdir() if p.is_file() and p.name != "formal_v3_checksums.sha256")
    checksum_lines = [f"{sha256(path)}  {path.name}" for path in produced]
    (output / "formal_v3_checksums.sha256").write_text("\n".join(checksum_lines) + "\n", encoding="ascii", newline="\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
