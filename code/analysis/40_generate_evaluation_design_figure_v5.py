#!/usr/bin/env python3
"""Generate the matched evaluation-design and F4 shift figure for this study."""

from __future__ import annotations

import csv, hashlib, json, os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUTPUT=ROOT/"05_figures_tables/formal_0907-07"
MODELS=["unet_matched","temporal_concat","temporal_attn"]
LABELS={"unet_matched":"U-Net","temporal_concat":"Seasonal\nconcatenation","temporal_attn":"Seasonal\nattention"}

def rows(path):
    with Path(path).open("r",encoding="utf-8-sig",newline="") as h:return list(csv.DictReader(h))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def configure(font_dir):
    regular=font_dir/"times.ttf"
    if not regular.is_file():raise FileNotFoundError(regular)
    for p in font_dir.glob("times*.ttf"):font_manager.fontManager.addfont(p)
    name=font_manager.FontProperties(fname=regular).get_name()
    plt.rcParams.update({"font.family":name,"font.size":14,"axes.titlesize":15,"axes.labelsize":14,
                         "xtick.labelsize":14,"ytick.labelsize":14,"legend.fontsize":14,
                         "axes.linewidth":1.0,"savefig.dpi":300})
    return name
def main():
    if OUTPUT.exists():raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True)
    font_dir=Path("C:/Windows/Fonts") if os.name=="nt" else Path("/tmp/times")
    font=configure(font_dir)
    random_summary={r["model_id"]:r for r in rows(ROOT/"04_results/random_holdout_v1_1_aggregate/model_summary_random_v1_1.csv")}
    spatial_summary={r["model_id"]:r for r in rows(ROOT/"04_results/experiment_v4_aggregate/model_summary_v4.csv")}
    random_fold=rows(ROOT/"04_results/random_holdout_v1_1_aggregate/spatial_random_paired_comparison_v1_1.csv")
    reversal=rows(ROOT/"04_results/f4_transfer_diagnostic_v1/feature_association_reversals_f4.csv")
    fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12.0,5.8),gridspec_kw={"width_ratios":[1.08,1]})
    x=np.arange(len(MODELS));width=.32
    spatial_mean=np.array([float(spatial_summary[m]["fold_ensemble_average_precision_mean"]) for m in MODELS])
    spatial_sd=np.array([float(spatial_summary[m]["fold_ensemble_average_precision_sd"]) for m in MODELS])
    random_mean=np.array([float(random_summary[m]["test_ap_mean"]) for m in MODELS])
    random_sd=np.array([float(random_summary[m]["test_ap_sd"]) for m in MODELS])
    ax1.bar(x-width/2,spatial_mean,width,yerr=spatial_sd,color="#4477AA",edgecolor="black",linewidth=.7,
            capsize=4,label="Spatial blocks",zorder=2)
    ax1.bar(x+width/2,random_mean,width,yerr=random_sd,color="#EEAA55",edgecolor="black",linewidth=.7,
            capsize=4,label="Random pixels",zorder=2)
    offsets=np.array([-.045,-.015,.015,.045])
    for index,model in enumerate(MODELS):
        matched=sorted([r for r in random_fold if r["model_id"]==model],key=lambda r:r["source_fold"])
        ax1.scatter(np.full(4,x[index]-width/2)+offsets,[float(r["spatial_v4_2_test_ap"]) for r in matched],
                    s=29,c="white",edgecolors="black",linewidths=.7,zorder=3)
        ax1.scatter(np.full(4,x[index]+width/2)+offsets,[float(r["random_v1_1_test_ap"]) for r in matched],
                    s=29,c="white",edgecolors="black",linewidths=.7,zorder=3)
    ax1.set_xticks(x,labels=[LABELS[m] for m in MODELS]);ax1.set_ylabel("Average precision")
    ax1.set_ylim(0,1.075);ax1.set_title("Matched evaluation designs",pad=10)
    ax1.grid(axis="y",color="#D9D9D9",linewidth=.7,zorder=0);ax1.spines[["top","right"]].set_visible(False)
    ax1.legend(loc="lower center",bbox_to_anchor=(.5,1.095),ncol=2,frameon=False,
               handletextpad=.55,columnspacing=1.15)
    ax1.text(-.13,1.17,"(a)",transform=ax1.transAxes,fontweight="bold",fontsize=15,va="top")

    train=np.array([float(r["train_feature_auc"]) for r in reversal]);test=np.array([float(r["test_feature_auc"]) for r in reversal])
    material=np.array([r["material_reversal"]=="True" for r in reversal])
    ax2.axvspan(.5,.72,ymin=0,ymax=(.5-.3)/(.72-.3),color="#FCE8E6",zorder=0)
    ax2.axvspan(.3,.5,ymin=(.5-.3)/(.72-.3),ymax=1,color="#FCE8E6",zorder=0)
    ax2.scatter(train[~material],test[~material],s=43,c="#8A8A8A",alpha=.75,edgecolors="white",linewidths=.5,
                label="Other channels",zorder=2)
    ax2.scatter(train[material],test[material],s=57,c="#B2182B",edgecolors="white",linewidths=.6,
                label="Material reversal",zorder=3)
    ax2.axvline(.5,color="black",linestyle="--",linewidth=1);ax2.axhline(.5,color="black",linestyle="--",linewidth=1)
    ax2.set_xlim(.3,.72);ax2.set_ylim(.3,.72);ax2.set_aspect("equal",adjustable="box")
    ax2.set_xlabel("Training feature AUC");ax2.set_ylabel("F4 test feature AUC")
    ax2.set_title("F4 feature–proxy association shift",pad=10)
    ax2.grid(color="#E3E3E3",linewidth=.6);ax2.spines[["top","right"]].set_visible(False)
    ax2.legend(loc="lower center",bbox_to_anchor=(.5,1.095),ncol=2,frameon=False,
               handletextpad=.45,columnspacing=.9)
    ax2.text(.04,.97,"Reversed: 22/32\nMaterial: 11/32",transform=ax2.transAxes,
             ha="left",va="top",fontsize=14)
    ax2.text(-.13,1.17,"(b)",transform=ax2.transAxes,fontweight="bold",fontsize=15,va="top")
    fig.subplots_adjust(left=.08,right=.98,bottom=.15,top=.79,wspace=.32)
    png=OUTPUT/"fig5_evaluation_design_and_f4_shift_0907-07.png"
    svg=OUTPUT/"fig5_evaluation_design_and_f4_shift_0907-07.svg"
    fig.savefig(png,dpi=300,facecolor="white");fig.savefig(svg,facecolor="white");plt.close(fig)
    table=[]
    for model in MODELS:
        pairs=[r for r in random_fold if r["model_id"]==model];delta=np.array([float(r["random_minus_spatial_ap"]) for r in pairs])
        table.append({"Model":LABELS[model].replace("\n"," "),
                      "Spatial AP mean ± SD":f"{float(spatial_summary[model]['fold_ensemble_average_precision_mean']):.4f} ± {float(spatial_summary[model]['fold_ensemble_average_precision_sd']):.4f}",
                      "Random AP mean ± SD":f"{float(random_summary[model]['test_ap_mean']):.4f} ± {float(random_summary[model]['test_ap_sd']):.4f}",
                      "Mean paired difference":f"{delta.mean():+.4f}","Random higher pairs":f"{int((delta>0).sum())}/4"})
    with (OUTPUT/"table5_v5_spatial_random_comparison.csv").open("w",encoding="utf-8-sig",newline="") as h:
        w=csv.DictWriter(h,fieldnames=list(table[0]),lineterminator="\n");w.writeheader();w.writerows(table)
    with Image.open(png) as im:
        dpi=im.info.get("dpi",(0,0));qa={"status":"accepted","font":font,"minimum_configured_font_pt":14,
            "png":{"file":png.name,"pixels":[im.width,im.height],"dpi":[float(dpi[0]),float(dpi[1])],"sha256":sha(png)},
            "svg":{"file":svg.name,"sha256":sha(svg)},"panels":["(a)","(b)"],
            "duplicate_internal_figure_title":False,"caption_required_in_word":True}
    (OUTPUT/"figure_qa_v5.json").write_text(json.dumps(qa,indent=2)+"\n",encoding="utf-8")
    (OUTPUT/"README.md").write_text("# formal Figure 5\n\nFigure 5 and Table 5 compare the matched random-pixel and spatially blocked evaluations. Internal experiment-version labels are retained only in the underlying provenance files.\n",encoding="utf-8")
    names=sorted(p.name for p in OUTPUT.iterdir() if p.is_file() and p.name!="formal_v5_checksums.sha256")
    (OUTPUT/"formal_v5_checksums.sha256").write_text("".join(f"{sha(OUTPUT/n)}  {n}\n" for n in names),encoding="ascii")
    print(json.dumps(qa,indent=2))
if __name__=="__main__":main()
