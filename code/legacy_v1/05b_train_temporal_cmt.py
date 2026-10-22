#!/usr/bin/env python
"""阶段6：时序跨模态 Transformer（v3）训练 + 消融。

用法：
  .venv/bin/python 02_scripts/modeling/05b_train_temporal_cmt.py --seed 0
  消融1（无时间注意力，仅拼接）：... --temporal concat
  消融2（单季）：... --seasons 1
评估同 05：早停 VAL，TEST 滑窗报 AUC/AP/F1@0.5。
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
from temporal_cmt import TemporalCMT

OUT_DIR = "01_data/interim/results_v2"


def run(seed, epochs, batch, lr, device, temporal, seasons, npatches, base, tta, modality):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    d = load()
    pw = pos_weight_from(d)
    season_sel = list(range(4)) if seasons == 4 else [1]
    band_sel = {"all": list(range(8)), "thermal": [0], "optical": list(range(1, 8))}[modality]
    model = TemporalCMT(season_sel=season_sel, temporal=temporal, base=base,
                        band_sel=band_sel, n_bands=8).to(device)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = TrainPatchDataset(d, seed=seed, n=npatches)
    # Dataset 内部按 seed 在线抽样；单进程避免 fork 后多个 worker 复制同一 RNG 状态。
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=0,
                                     shuffle=True, drop_last=True)
    tag = f"tcmt_{temporal}_s{len(season_sel)}_{modality}" + (f"_b{base}" if base != 16 else "")
    print(f"[{tag}] seed={seed} 季节{len(season_sel)} modality={modality} base={base} pos_weight={pw:.2f}")

    best_f1, best_epoch, best_state, wait = -1.0, 0, None, 0
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
        sched.step()
        vp = sliding_predict(model, d, device, rows=VAL_RANGE)
        _, vf1 = best_threshold(d["y"], vp, d["val"])
        vm = metrics(d["y"], vp, d["val"], thresh=0.5)
        hist.append({"epoch": ep, "train_loss": tot / nb, "val_auc": vm["auc"], "val_f1": vf1})
        tag_ = ""
        if vf1 > best_f1:
            best_f1, best_epoch, best_state, wait = (
                vf1, ep, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, 0)
            tag_ = " ★"
        else:
            wait += 1
        print(f"ep{ep:2d} loss={tot/nb:.4f} val_AUC={vm['auc']:.3f} val_F1={vf1:.3f} "
              f"({time.time()-t0:.0f}s){tag_}")
        if wait >= 5:
            print("早停"); break

    model.load_state_dict(best_state)
    print("\n== TEST 区评估（最佳 val 模型%s）==" % (" + TTA" if tta else ""))
    tp = sliding_predict(model, d, device, tta=tta)
    tm = metrics(d["y"], tp, d["test"], thresh=0.5)
    print(f"  AUC={tm['auc']:.3f}  AP={tm['ap']:.3f}  F1@0.5={tm['f1']:.3f}  "
          f"P={tm['precision']:.3f}  R={tm['recall']:.3f}  (n={tm['n']}, pos={tm['pos']})")

    os.makedirs(OUT_DIR, exist_ok=True)
    base_fname = f"{tag}_seed{seed}"
    torch.save(model.state_dict(), os.path.join(OUT_DIR, f"{base_fname}.pt"))
    np.save(os.path.join(OUT_DIR, f"{base_fname}_prob_test.npy"), tp)
    res = {"model": "temporal_cmt", "temporal": temporal, "base": base, "tta": tta,
           "modality": modality, "band_sel": band_sel,
           "seasons": ["spring", "summer", "autumn", "winter"] if seasons == 4 else ["summer"],
           "seed": seed, "params_M": sum(p.numel() for p in model.parameters()) / 1e6,
           "best_epoch": best_epoch,
           "test": tm, "val_best_f1": best_f1, "history": hist}
    with open(os.path.join(OUT_DIR, f"{base_fname}_result.json"), "w") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)
    print(f"已保存 → {OUT_DIR}/{base_fname}.pt / _prob_test.npy / _result.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--temporal", choices=["attn", "concat"], default="attn")
    ap.add_argument("--seasons", type=int, choices=[4, 1], default=4)
    ap.add_argument("--npatches", type=int, default=4096)
    ap.add_argument("--base", type=int, default=16)
    ap.add_argument("--tta", action="store_true", help="TEST 评估启用水平翻转 TTA")
    ap.add_argument("--modality", choices=["all", "thermal", "optical"], default="all")
    run(**vars(ap.parse_args()))
