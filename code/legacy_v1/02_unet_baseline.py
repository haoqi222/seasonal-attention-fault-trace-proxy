#!/usr/bin/env python
"""阶段5：U-Net 基线（resnet18 编码器，32 通道输入，二分类分割）。

训练：128×128 切片（50% 断层中心 + 50% 随机背景），BCEWithLogits(pos_weight)，
      忽略区(y==2)与无效像元不计损失。早停依据：VAL 区（行 1050-1200）F1。
评估：TEST 区（行 1250-1840）滑窗预测，AUC/AP/F1@0.5/P/R。

用法：
  .venv/bin/python 02_scripts/modeling/02_unet_baseline.py [--seed 0] [--epochs 30]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common import (load, TrainPatchDataset, sliding_predict, metrics,
                        best_threshold, pos_weight_from, VAL_RANGE, TEST_RANGE)

OUT_DIR = "01_data/interim/results_v2"


def run(seed, epochs, batch, lr, device, tta):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    d = load()
    pw = pos_weight_from(d)
    print(f"训练正负权重(pos_weight) = {pw:.2f}")

    model = smp.Unet("resnet18", encoder_weights=None, in_channels=d["X"].shape[-1], classes=1).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ds = TrainPatchDataset(d, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=0,
                                     shuffle=True, drop_last=True)

    best_f1, best_thr, best_epoch, best_state, wait = -1.0, 0.5, 0, None, 0
    hist = []
    for ep in range(1, epochs + 1):
        model.train()
        t0, tot, nb = time.time(), 0.0, 0
        for x, y, m in dl:
            x, y, m = x.to(device), y.to(device), m.to(device)
            out = model(x).squeeze(1)
            loss = crit(out[m > 0], y[m > 0].float())
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        vp = sliding_predict(model, d, device, rows=VAL_RANGE)
        thr, vf1 = best_threshold(d["y"], vp, d["val"])
        vm = metrics(d["y"], vp, d["val"], thresh=thr)
        hist.append({"epoch": ep, "train_loss": tot / nb,
                     "val_auc": vm["auc"], "val_f1": vf1, "val_thresh": thr})
        tag = ""
        if vf1 > best_f1:
            best_f1, best_thr, best_epoch, best_state, wait = (
                vf1, thr, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
            tag = " ★"
        else:
            wait += 1
        print(f"ep{ep:2d} loss={tot/nb:.4f} val_AUC={vm['auc']:.3f} val_F1={vf1:.3f} "
              f"(thr={thr:.2f}, {time.time()-t0:.0f}s){tag}")
        if wait >= 5:
            print("早停"); break

    model.load_state_dict(best_state)
    print("\n== TEST 区评估（最佳 val 模型）==")
    tp = sliding_predict(model, d, device, tta=tta)          # TEST 行区
    tm = metrics(d["y"], tp, d["test"], thresh=0.5)
    print(f"  AUC={tm['auc']:.3f}  AP={tm['ap']:.3f}  F1@0.5={tm['f1']:.3f}  "
          f"P={tm['precision']:.3f}  R={tm['recall']:.3f}  (n={tm['n']}, pos={tm['pos']})")

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f"unet_seed{seed}"
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{base}.pt"))
    np.save(os.path.join(OUT_DIR, f"{base}_prob_test.npy"), tp)
    res = {"model": "unet_resnet18", "seed": seed, "epochs": epochs, "batch": batch, "lr": lr,
           "tta": tta, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
           "pos_weight": pw, "val": vm, "test": tm, "best_epoch": best_epoch, "history": hist}
    with open(os.path.join(OUT_DIR, f"{base}_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"已保存 → {OUT_DIR}/{base}.pt / _prob_test.npy / _result.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--tta", action="store_true", help="TEST 评估启用水平翻转 TTA")
    run(**vars(ap.parse_args()))
