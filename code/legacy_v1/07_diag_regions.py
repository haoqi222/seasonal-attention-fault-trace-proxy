#!/usr/bin/env python
"""阶段6：诊断——TEST 区按纬度带细分每个模型的 AUC/AP。

TEST 区行 1250-1840 分 3 带（各 ~197 行），看模型在南段哪里退化。
用法：
  .venv/bin/python 02_scripts/modeling/07_diag_regions.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common import load, metrics

RES = "01_data/interim/results"
TEST_RANGE = (1250, 1840)
BANDS = {"北三(1250-1446)": (1250, 1447), "中三(1447-1643)": (1447, 1644), "南三(1644-1840)": (1644, 1840)}


def main():
    d = load()
    y = d["y"]
    files = [f for f in os.listdir(RES) if f.endswith("_prob_test.npy")]
    for f in sorted(files):
        p = np.load(os.path.join(RES, f))
        name = f.replace("_prob_test.npy", "")
        print(f"\n== {name} ==")
        for label, (r0, r1) in BANDS.items():
            m = d["test"].copy()
            m[:r0] = False; m[r1:] = False
            if m.sum() == 0:
                continue
            r = metrics(y, p, m)
            print(f"  {label}: n={r['n']:6d} pos={r['pos']:4d} AUC={r['auc']:.3f} AP={r['ap']:.3f}")


if __name__ == "__main__":
    main()
