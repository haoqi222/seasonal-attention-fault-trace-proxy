#!/usr/bin/env python
"""阶段6：汇总所有模型结果 → 对比表（TEST 区 AUC/AP/F1@0.5）。

读取 01_data/interim/results/*_result.json，按模型/种子/消融打印对比。
用法：
  .venv/bin/python 02_scripts/modeling/06_summarize.py
"""
import glob
import json
import os
from collections import defaultdict

import numpy as np

RES = "01_data/interim/results"


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(RES, "*_result.json"))):
        with open(f) as fh:
            r = json.load(fh)
        t = r.get("test", {})
        model = r.get("model", os.path.basename(f))
        if model == "cross_modal_transformer":
            aw = r.get("attn_window", 0)
            name = f"CMT-{r['fusion']}"
            name += f"({'四季' if len(r['seasons'])==4 else '单季summer'}{'' if aw == 0 else ',aw'+str(aw)})"
        elif model == "temporal_cmt":
            b = r.get("base", 16)
            name = f"TCMT-{r['temporal']}"
            name += f"({'四季' if len(r['seasons'])==4 else '单季summer'}{'' if b == 16 else ',b'+str(b)})"
        elif model == "unet_resnet18":
            name = "U-Net(resnet18)"
        elif model == "random_forest":
            name = "RandomForest"
        else:
            name = model
        rows.append({
            "name": name, "seed": r.get("seed", -1),
            "auc": t.get("auc"), "ap": t.get("ap"), "f1": t.get("f1"),
            "p": t.get("precision"), "r": t.get("recall"), "n": t.get("n"),
            "params_M": r.get("params_M"),
        })
    rows.sort(key=lambda r: (r["name"], r["seed"]))
    print(f"{'模型':<20} {'seed':<4} {'AUC':<7} {'AP':<7} {'F1@0.5':<7} {'P':<7} {'R':<7}")
    print("-" * 62)
    for r in rows:
        print(f"{r['name']:<20} {r['seed']:<4} {r['auc']:.3f}  {r['ap']:.3f}  "
              f"{r['f1']:.3f}  {r['p']:.3f}  {r['r']:.3f}")
    # 按名称聚合均值
    print("\n=== 均值（多 seed） ===")
    agg = defaultdict(list)
    for r in rows:
        agg[r["name"]].append(r)
    print(f"{'模型':<20} {'n':<4} {'AUC±':<12} {'AP±':<12} {'F1±':<12}")
    for name in sorted(agg):
        rs = agg[name]
        auc = np.mean([r["auc"] for r in rs]); auc_s = np.std([r["auc"] for r in rs])
        ap = np.mean([r["ap"] for r in rs]); ap_s = np.std([r["ap"] for r in rs])
        f1 = np.mean([r["f1"] for r in rs]); f1_s = np.std([r["f1"] for r in rs])
        print(f"{name:<20} {len(rs):<4} {auc:.3f}±{auc_s:.3f}  {ap:.3f}±{ap_s:.3f}  {f1:.3f}±{f1_s:.3f}")


if __name__ == "__main__":
    main()
