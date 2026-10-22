#!/usr/bin/env python
"""阶段6：对已保存的深度模型 checkpoint 做 TTA 重推演（TEST 区），更新结果文件。

不重新训练，仅推理：加载 .pt → sliding_predict(tta=True) → 更新 _prob_test.npy 与 _result.json。
RF 逐像元特征对称，TTA 不变，不处理。
用法：
  .venv/bin/python 02_scripts/modeling/05c_tta_reeval.py [--device cuda:0] [--only tcmt_attn]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common import load, sliding_predict, metrics
from cross_modal import CrossModalTransformer
from temporal_cmt import TemporalCMT

OUT_DIR = "01_data/interim/results"


def build_model(r):
    model = r["model"]
    seasons = len(r.get("seasons", []))
    season_sel = list(range(4)) if seasons == 4 else [1]
    if model == "cross_modal_transformer":
        m = CrossModalTransformer(season_sel=season_sel, fusion=r["fusion"],
                                  base=r.get("base", 32),
                                  attn_window=r.get("attn_window", 0))
    elif model == "temporal_cmt":
        m = TemporalCMT(season_sel=season_sel, temporal=r["temporal"],
                        base=r.get("base", 16))
    elif model == "unet_resnet18":
        import segmentation_models_pytorch as smp
        m = smp.Unet("resnet18", encoder_weights=None, in_channels=36, classes=1)
    else:
        raise ValueError(f"unknown model {model}")
    return m


def main(device, only):
    d = load()
    for f in sorted(glob.glob(os.path.join(OUT_DIR, "*_result.json"))):
        with open(f) as fh:
            r = json.load(fh)
        if r.get("model") in ("random_forest",):
            continue
        stem = os.path.basename(f).replace("_result.json", "")
        if only and only not in stem:
            continue
        m = build_model(r).to(device)
        m.load_state_dict(torch.load(os.path.join(OUT_DIR, f"{stem}.pt"), map_location=device))
        tp = sliding_predict(m, d, device, tta=True)
        tm = metrics(d["y"], tp, d["test"], thresh=0.5)
        np.save(os.path.join(OUT_DIR, f"{stem}_prob_test.npy"), tp)
        r["test"] = tm
        r["tta"] = True
        with open(f, "w") as fh:
            json.dump(r, fh, indent=2, ensure_ascii=False)
        print(f"{stem:32s} → TTA AUC={tm['auc']:.3f} AP={tm['ap']:.3f} F1={tm['f1']:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--only", default="", help="仅处理名字含该子串的模型")
    args = ap.parse_args()
    main(args.device, args.only)
