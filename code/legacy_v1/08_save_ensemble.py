#!/usr/bin/env python
"""阶段6：主模型 = TCMT-attn(base16) 4 种子概率集成，保存正式产物。

产出（01_data/interim/results/）：
  tcmt_attn_ens4_prob_test.npy   —— 4 种子平均概率图（TEST 区）
  tcmt_attn_ens4_result.json     —— 集成评估（test + 区域诊断 + 种子明细）
用法：
  .venv/bin/python 02_scripts/modeling/08_save_ensemble.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common import load, metrics

RES = "01_data/interim/results"
SEEDS = ["tcmt_attn_s4_seed0", "tcmt_attn_s4_seed1", "tcmt_attn_s4_seed2", "tcmt_attn_s4_seed3"]
BANDS = {"北三(1250-1446)": (1250, 1447), "中三(1447-1643)": (1447, 1644), "南三(1644-1840)": (1644, 1840)}


def main():
    d = load()
    y = d["y"]
    ps = np.mean([np.load(os.path.join(RES, f"{s}_prob_test.npy")).astype(np.float64)
                  for s in SEEDS], axis=0)
    np.save(os.path.join(RES, "tcmt_attn_ens4_prob_test.npy"), ps.astype(np.float32))
    tm = metrics(y, ps, d["test"], thresh=0.5)
    per_seed = []
    for s in SEEDS:
        with open(os.path.join(RES, f"{s}_result.json")) as fh:
            r = json.load(fh)
        per_seed.append({"seed": r["seed"], "test_auc": r["test"]["auc"]})
    diag = {}
    for label, (r0, r1) in BANDS.items():
        m = d["test"].copy(); m[:r0] = False; m[r1:] = False
        r = metrics(y, ps, m, thresh=0.5)
        diag[label] = {"n": r["n"], "pos": r["pos"], "auc": r["auc"], "ap": r["ap"]}
    res = {"model": "temporal_cmt_ensemble", "seed": -1, "seeds": SEEDS, "tta": True,
           "per_seed_auc": per_seed, "test": tm, "region_diag": diag}
    with open(os.path.join(RES, "tcmt_attn_ens4_result.json"), "w") as fh:
        json.dump(res, fh, indent=2, ensure_ascii=False)
    print(f"集成：AUC={tm['auc']:.3f} AP={tm['ap']:.3f} F1@0.5={tm['f1']:.3f}")
    print("种子明细：", ", ".join(f"s{r['seed']}={r['test_auc']:.3f}" for r in per_seed))
    for k, v in diag.items():
        print(f"  {k}: AUC={v['auc']:.3f} AP={v['ap']:.3f} (n={v['n']}, pos={v['pos']})")


if __name__ == "__main__":
    main()
