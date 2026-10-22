#!/usr/bin/env python
"""阶段5：随机森林基线（逐像元 32 基特征 + 7×7 邻域上下文 mean/std）。

训练：TRAIN 区（行 0-500 ∪ 550-1050）有效且非忽略像元 = 全部正样本 + 按 3:1 下采样背景。
评估：TEST 区（行 1250-1840）有效且非忽略像元。
同一空间分块、同一缓冲带，与 U-Net 公平对比。

用法：
  .venv/bin/python 02_scripts/modeling/03_rf_baseline.py [--seed 0] [--trees 200] [--bg-ratio 3]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common import load, rf_features, metrics, best_threshold

OUT_DIR = "01_data/interim/results_v2"


def run(seed, trees, bg_ratio):
    np.random.seed(seed)
    d = load()
    print("构建逐像元特征（32 基 + 邻域上下文；count已排除）...")
    F = rf_features(d)
    print(f"特征矩阵: {F.shape}")

    tr = d["train"] & (d["y"] != 2)
    pos = np.argwhere((d["y"] == 1) & tr)
    bg = np.argwhere((d["y"] == 0) & tr)
    n_bg = min(len(bg), int(len(pos) * bg_ratio))
    bg = bg[np.random.choice(len(bg), n_bg, replace=False)]
    idx = np.concatenate([pos, bg])
    ytr = np.concatenate([np.ones(len(pos), dtype=np.int8),
                          np.zeros(len(bg), dtype=np.int8)])
    Xtr = np.vstack([F[pos[:, 0], pos[:, 1]], F[bg[:, 0], bg[:, 1]]])
    print(f"训练样本: {len(ytr)}（正 {len(pos)}，背景 {n_bg}）")

    clf = RandomForestClassifier(n_estimators=trees, min_samples_leaf=5,
                                 class_weight="balanced_subsample",
                                 n_jobs=-1, random_state=seed)
    t0 = time.time()
    clf.fit(Xtr, ytr)
    print(f"训练完成（{time.time()-t0:.0f}s）")

    te = d["test"] & (d["y"] != 2)
    yt = (d["y"][te] == 1).astype(np.int8)
    Xte = F[te]
    p = clf.predict_proba(Xte)[:, 1]
    prob_map = np.zeros_like(d["y"], dtype=np.float32)
    prob_map[te] = p.astype(np.float32)

    # 阈值在 VAL 区选（仅参考）；test 报固定阈值 0.5，避免空间漂移下阈值不可迁移
    va = d["val"] & (d["y"] != 2)
    pv = clf.predict_proba(F[va])[:, 1]
    prob_val = np.zeros_like(d["y"], dtype=np.float32)
    prob_val[va] = pv.astype(np.float32)
    thr, _ = best_threshold(d["y"], prob_val, d["val"])
    print(f"VAL 上最佳阈值(参考) = {thr:.3f}")

    tm = metrics(d["y"], prob_map, d["test"], thresh=0.5)
    print(f"\n== TEST 区评估（固定阈值 0.5）==")
    print(f"  AUC={tm['auc']:.3f}  AP={tm['ap']:.3f}  F1@0.5={tm['f1']:.3f}  "
          f"P={tm['precision']:.3f}  R={tm['recall']:.3f}  (n={tm['n']}, pos={tm['pos']})")

    imp = sorted(zip(clf.feature_importances_, range(F.shape[2])), reverse=True)
    print("\n特征重要度 Top10（通道号 0-35 基特征，36+ 邻域上下文）:")
    for v, c in imp[:10]:
        print(f"  ch{c:2d}: {v:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f"rf_seed{seed}"
    np.save(os.path.join(OUT_DIR, f"{base}_prob_test.npy"), prob_map)
    res = {"model": "random_forest", "seed": seed, "n_estimators": trees, "tta": False,
           "bg_ratio": bg_ratio, "n_train": int(len(ytr)), "n_features": int(F.shape[2]),
           "test": tm, "top_importances": [(int(c), float(v)) for v, c in imp[:10]]}
    with open(os.path.join(OUT_DIR, f"{base}_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"已保存 → {OUT_DIR}/{base}_prob_test.npy / _result.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trees", type=int, default=200)
    ap.add_argument("--bg-ratio", type=float, default=3.0)
    run(**vars(ap.parse_args()))
