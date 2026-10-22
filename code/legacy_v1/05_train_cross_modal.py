#!/usr/bin/env python
"""阶段6：跨模态 Transformer 主模型训练 + 消融。

用法：
  主模型（四季 + cross-attention）：
    .venv/bin/python 02_scripts/modeling/05_train_cross_modal.py --seed 0
  消融1（无注意力，直接拼接）：
    ... --fusion concat
  消融2（单季，无时序维）：
    ... --seasons 1        # 只用 summer（band*4+1）

评估与 02/03 一致：早停用 VAL(行 1050-1200)，TEST(行 1250-1840) 滑窗上报 AUC/AP/F1@0.5。
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))
from dataset_common import (load, TrainPatchDataset, sliding_predict, metrics,
                            best_threshold, pos_weight_from, VAL_RANGE)
from cross_modal import CrossModalTransformer

OUT_DIR = "01_data/interim/results"
SEASON_NAMES = ["spring", "summer", "autumn", "winter"]


def run(seed, epochs, batch, lr, device, fusion, seasons, wd, dropout, base, cosine, aw):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = load()
    pw = pos_weight_from(d)
    season_sel = list(range(4)) if seasons == 4 else [1]   # [1] = summer
    model = CrossModalTransformer(season_sel=season_sel, fusion=fusion, base=base,
                                  dropout=dropout, attn_window=aw).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    if wd > 0:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
    if cosine:
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = TrainPatchDataset(d, seed=seed)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=2,
                                     shuffle=True, drop_last=True)
    tag = f"cmt_{fusion}_s{len(season_sel)}" + (f"_aw{aw}" if aw > 0 else "")
    print(f"[{tag}] seed={seed} 输入: 热{len(model.t_ch)} + 结构{len(model.o_ch)}ch, pos_weight={pw:.2f}")

    best_f1, best_state, wait = -1.0, None, 0
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
        _, vf1 = best_threshold(d["y"], vp, d["val"])
        vm = metrics(d["y"], vp, d["val"], thresh=0.5)
        hist.append({"epoch": ep, "train_loss": tot / nb, "val_auc": vm["auc"], "val_f1": vf1})
        tag_ = ""
        if vf1 > best_f1:
            best_f1, best_state, wait = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
            tag_ = " ★"
        else:
            wait += 1
        print(f"ep{ep:2d} loss={tot/nb:.4f} val_AUC={vm['auc']:.3f} val_F1={vf1:.3f} "
              f"({time.time()-t0:.0f}s){tag_}")
        if wait >= 5:
            print("早停"); break
        if cosine:
            sched.step()

    model.load_state_dict(best_state)
    print("\n== TEST 区评估（最佳 val 模型）==")
    tp = sliding_predict(model, d, device)
    tm = metrics(d["y"], tp, d["test"], thresh=0.5)
    print(f"  AUC={tm['auc']:.3f}  AP={tm['ap']:.3f}  F1@0.5={tm['f1']:.3f}  "
          f"P={tm['precision']:.3f}  R={tm['recall']:.3f}  (n={tm['n']}, pos={tm['pos']})")

    os.makedirs(OUT_DIR, exist_ok=True)
    base = f"{tag}_seed{seed}"
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{base}.pt"))
    np.save(os.path.join(OUT_DIR, f"{base}_prob_test.npy"), tp)
    res = {"model": "cross_modal_transformer", "fusion": fusion, "seasons": SEASON_NAMES if seasons == 4 else ["summer"],
           "seed": seed, "epochs": epochs, "batch": batch, "lr": lr, "attn_window": aw,
           "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
           "test": tm, "val_best_f1": best_f1, "history": hist}
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
    ap.add_argument("--fusion", choices=["cross_attn", "concat"], default="cross_attn")
    ap.add_argument("--seasons", type=int, choices=[4, 1], default=4)
    ap.add_argument("--wd", type=float, default=0.0, help="weight decay（>0 用 AdamW）")
    ap.add_argument("--dropout", type=float, default=0.0, help="conv_block 内 Dropout2d 概率")
    ap.add_argument("--base", type=int, default=32, help="编码器基础通道数")
    ap.add_argument("--cosine", action="store_true", help="余弦学习率退火")
    ap.add_argument("--aw", type=int, default=0, help="cross-attention 窗口（0=全局）")
    run(**vars(ap.parse_args()))
