#!/usr/bin/env python
"""阶段5：数据集加载 + 切片采样（128×128）+ 评估工具。

空间划分（北=0，南=1839）：
  TRAIN = 行 [0,500) ∪ [550,1050)     （Z1 + Z2 北段）
  VAL   = 行 [1050,1200)               （Z2 南端连续 15km 校验带）
  TEST  = 行 [1250,1840)               （Z3 南段，含最南负样本富集区）
  缓冲   = [500,550) ∪ [1200,1250)     （各 5km，丢弃，切断空间自相关泄漏）

用法（被 02_unet_baseline.py / 03_rf_baseline.py import）：
  from dataset_common import load, TrainPatchDataset, sliding_predict, rf_features, metrics
"""
import json
import os
import sys

import numpy as np
import torch
from scipy.ndimage import uniform_filter
from torch.utils.data import Dataset

sys.path.insert(0, os.path.dirname(__file__))

PKG = "01_data/interim/dataset_v2"
TRAIN_RANGES = [(0, 500), (550, 1050)]
VAL_RANGE = (1050, 1200)
TEST_RANGE = (1250, 1840)
PATCH = 128


def load():
    """加载数据包 → z-score 归一化 X、标签 y、有效掩膜、分区掩膜。"""
    X = np.load(os.path.join(PKG, "X.npy")).astype(np.float32)
    y = np.load(os.path.join(PKG, "y.npy"))
    valid = np.load(os.path.join(PKG, "valid.npy"))
    with open(os.path.join(PKG, "norm_stats.json")) as f:
        st = json.load(f)
    mean = np.array(st["mean"], dtype=np.float32)
    std = np.array(st["std"], dtype=np.float32)
    Xn = (X - mean) / std
    Xn[~valid] = 0.0
    ny, nx = y.shape

    train = np.zeros((ny, nx), dtype=bool)
    for r0, r1 in TRAIN_RANGES:
        train[r0:r1] = True
    train &= valid
    val = np.zeros((ny, nx), dtype=bool); val[VAL_RANGE[0]:VAL_RANGE[1]] = True; val &= valid
    test = np.zeros((ny, nx), dtype=bool); test[TEST_RANGE[0]:TEST_RANGE[1]] = True; test &= valid
    return {"X": Xn, "y": y, "valid": valid, "train": train, "val": val, "test": test}


def pos_weight_from(d):
    """BCEWithLogits 的正类权重 = 负/正（训练区，排除忽略）。"""
    m = d["train"] & (d["y"] != 2)
    pos = int(((d["y"] == 1) & m).sum())
    neg = int(((d["y"] == 0) & m).sum())
    return neg / max(pos, 1)


class TrainPatchDataset(Dataset):
    """训练切片：50% 断层中心（正样本像元）+ 50% 随机背景，128×128。"""

    def __init__(self, d, fault_frac=0.5, n=4096, seed=0):
        self.X, self.y, self.train = d["X"], d["y"], d["train"]
        self.rng = np.random.default_rng(seed)
        H, W = self.X.shape[:2]
        p = PATCH // 2
        # 仅允许完整落在单一训练带内的切片中心，避免读取验证带或缓冲带标签。
        eligible = np.zeros_like(self.train)
        for r0, r1 in TRAIN_RANGES:
            eligible[r0 + p:r1 - p, p:W - p] = True
        eligible &= self.train
        pos = np.argwhere((self.y == 1) & eligible)
        bg = np.argwhere((self.y == 0) & eligible)
        self.pos, self.bg = pos, bg
        self.fault_frac = fault_frac
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, _):
        r = self.rng.random()
        pool = self.pos if r < self.fault_frac else self.bg
        cy, cx = pool[self.rng.integers(len(pool))]
        x = self.X[cy - 64:cy + 64, cx - 64:cx + 64]   # (P,P,C)
        y = self.y[cy - 64:cy + 64, cx - 64:cx + 64]
        if self.rng.random() < 0.5:
            x, y = x[:, ::-1], y[:, ::-1]              # 水平翻转（断层 N-S，翻转保持语义）
        if self.rng.random() < 0.5:
            x, y = x[::-1], y[::-1]                    # 垂直翻转
        train = self.train[cy - 64:cy + 64, cx - 64:cx + 64]
        # 双重保护：损失只使用训练区内、有效且非忽略的像元。
        m = train & (y != 2) & (np.abs(x).sum(axis=-1) > 0)
        return (torch.from_numpy(x.transpose(2, 0, 1).copy()),
                torch.from_numpy(y.copy()).long(),
                torch.from_numpy(m.copy()).float())


def sliding_predict(model, d, device, step=64, rows=None, tta=False):
    """指定行区滑窗（128，stride 64，含边缘最后一片）预测，重叠概率平均。rows=None→TEST。

    tta=True 时对每个 patch 额外评估水平翻转并平均（4 季热/结构对翻转不变）。"""
    X = d["X"]
    H, W = X.shape[:2]
    r0, r1 = rows if rows is not None else TEST_RANGE
    acc = np.zeros((H, W), dtype=np.float64)
    cnt = np.zeros((H, W), dtype=np.int32)
    model.eval()
    with torch.no_grad():
        ys = set(range(r0, r1 - PATCH + 1, step)) | {r1 - PATCH}
        xs = set(range(0, W - PATCH + 1, step)) | {W - PATCH}
        for y0 in sorted(ys):
            for x0 in sorted(xs):
                patch = X[y0:y0 + PATCH, x0:x0 + PATCH]
                t = torch.from_numpy(patch.transpose(2, 0, 1))[None].to(device)
                p = torch.sigmoid(model(t))[0, 0].cpu().numpy()
                if tta:
                    tf = torch.from_numpy(patch[:, ::-1].transpose(2, 0, 1).copy())[None].to(device)
                    p = 0.5 * (p + torch.sigmoid(model(tf))[0, 0].cpu().numpy()[:, ::-1])
                acc[y0:y0 + PATCH, x0:x0 + PATCH] += p
                cnt[y0:y0 + PATCH, x0:x0 + PATCH] += 1
    prob = np.divide(acc, np.maximum(cnt, 1), out=np.zeros_like(acc), where=cnt > 0)
    return prob


def best_threshold(y, p, mask, n=51):
    """在指定区(mask)上网格搜索使 F1 最大的阈值（用 val，不在 test 上调）。"""
    sel = mask & (y != 2)
    yb = (y[sel] == 1).astype(np.int8)
    pb = np.clip(p[sel], 0, 1)
    best_t, best_f1 = 0.5, -1.0
    for t in np.linspace(0.0, 1.0, n):
        pred = pb >= t
        tp = int((pred & (yb == 1)).sum()); fp = int((pred & (yb == 0)).sum()); fn = int((~pred & (yb == 1)).sum())
        prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return float(best_t), float(best_f1)


def metrics(y, p, mask, thresh=0.5):
    """有效且非忽略像元上的 AUC / AP / F1@thresh / P / R。"""
    from sklearn.metrics import average_precision_score, roc_auc_score
    sel = mask & (y != 2)
    yb = (y[sel] == 1).astype(np.int8)
    pb = np.clip(p[sel], 0, 1)
    auc = roc_auc_score(yb, pb)
    ap = average_precision_score(yb, pb)
    pred = pb >= thresh
    tp = int((pred & (yb == 1)).sum()); fp = int((pred & (yb == 0)).sum()); fn = int((~pred & (yb == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    return {"n": int(len(yb)), "pos": int(yb.sum()), "auc": float(auc), "ap": float(ap),
            "thresh": float(thresh), "f1": float(f1), "precision": float(prec), "recall": float(rec),
            "tp": tp, "fp": fp, "fn": fn}


def rf_features(d, ctx_channels=None):
    """逐像元特征：32 基 + 邻域上下文（mean/std，7×7≈700m）。"""
    X = d["X"]
    if ctx_channels is None:
        # ST、NIR、NDVI；不含 count（第二版已完全从特征张量移除）。
        ctx_channels = list(range(4)) + list(range(8, 12)) + list(range(16, 20))
    feats = [X]
    for c in ctx_channels:
        mean = uniform_filter(X[..., c], size=7, mode="constant")
        m2 = uniform_filter(X[..., c] ** 2, size=7, mode="constant")
        std = np.sqrt(np.maximum(m2 - mean ** 2, 0))
        feats.append(mean[..., None]); feats.append(std[..., None])
    return np.concatenate(feats, axis=-1)
