#!/usr/bin/env python
"""Official ChangeOS R101 on the xBD test split, scored per inventory building.

For every evaluation condition of make_conditions.py the condition's pre-event
image (displaced with the inventory, or registered to the post-event image) and
the post-event image go through the model, and each test building, at its
condition footprint, is scored in two ways:
  inventory  mean damage probability over the footprint's pixels, ignoring the
             model's own localization (the pixel model given the inventory)
  end2end    the official weighted vote over footprint pixels that the model
             localizes as building; a footprint with no such pixel is a miss
"""
import argparse
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch

from bda_data import IMG, ImageSource, load_conditions, load_tables, polygons
from bda_metrics import building_metrics, pred_metrics

MEAN = torch.tensor([123.675, 116.28, 103.53] * 2).view(1, 6, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375] * 2).view(1, 6, 1, 1)
VOTE_WEIGHTS = np.array([8.0, 38.0, 25.0, 11.0])  # official ChangeOS object vote


def footprint_ids(polys, offsets):
    """1024 x 1024 map of local building index + 1 (0 = no footprint), footprints moved by offsets."""
    canvas = np.zeros((IMG, IMG), np.int32)
    for k, (parts, o) in enumerate(zip(polys, offsets)):
        for p in parts:
            cv2.fillPoly(canvas, [np.round(p + o).astype(np.int32)], k + 1)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--weights", default="external/checkpoints/changeos_r101_official.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond-dir", default=None, help="evaluation conditions (default: <data>/conditions_v2)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scenes, buildings = load_tables(args.data)
    te = buildings[(buildings.official == "test") & (buildings.label >= 0)].reset_index(drop=True)
    polys = polygons(te.wkt_pre)
    groups = {sid: np.flatnonzero(te.scene_id.to_numpy() == sid) for sid in te.scene_id.unique()}
    img_dir = args.image_dir or args.data
    pre_orig = ImageSource(os.path.join(img_dir, "pre.u8"), len(scenes))
    post_orig = ImageSource(os.path.join(img_dir, "post.u8"), len(scenes))
    model = torch.jit.load(args.weights, map_location="cuda").eval()
    mean, std = MEAN.cuda(), STD.cuda()
    cond_dir = args.cond_dir or os.path.join(args.data, "conditions_v2")
    conds, scene_ids = load_conditions(cond_dir, te.building_id.to_numpy())
    rows, saved, t0 = [], {}, time.time()
    for c in conds:
        cond = c["name"]
        src = lambda path, orig: orig if path is None else ImageSource(path, len(scene_ids), scene_ids)
        pre_src, post_src = src(c["pre"], pre_orig), src(c["post"], post_orig)
        p_inv = np.zeros((len(te), 4))
        vote = np.full(len(te), -1)
        for sid, idx in groups.items():
            x = np.concatenate([pre_src[int(sid)], post_src[int(sid)]], axis=2)
            x = torch.from_numpy(np.ascontiguousarray(x)).cuda().permute(2, 0, 1)[None].float()
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                loc_logits, dmg_logits = model((x - mean) / std)
            loc = (loc_logits[0, 0] > 0).cpu().numpy().ravel()
            prob = torch.softmax(dmg_logits[0].float(), 0)[1:5].cpu().numpy().reshape(4, -1)
            dmg = prob.argmax(0)
            ids = footprint_ids([polys[i] for i in idx], c["offsets"][idx]).ravel()
            n = len(idx) + 1
            cnt = np.bincount(ids, minlength=n)[1:].clip(min=1)
            sums = np.stack([np.bincount(ids, weights=prob[k], minlength=n)[1:] for k in range(4)], 1)
            p_inv[idx] = sums / sums.sum(1, keepdims=True).clip(min=1e-9)
            on = loc & (ids > 0)
            score = np.stack([np.bincount(ids[on & (dmg == k)], minlength=n)[1:] * VOTE_WEIGHTS[k] for k in range(4)], 1)
            has = score.sum(1) > 0
            vote[idx[has]] = score[has].argmax(1)
            del cnt
        y = te.label.to_numpy()
        m_inv = building_metrics(y, p_inv, te.disaster.to_numpy())["overall"]
        m_e2e = pred_metrics(y, vote)
        rows.append({"condition": cond, "scoring": "inventory", **m_inv})
        rows.append({"condition": cond, "scoring": "end2end", **m_e2e})
        saved[f"{cond}_inventory"], saved[f"{cond}_vote"] = p_inv, vote
        print(f"{cond}: inventory macro-F1 {m_inv['macro_f1']:.3f} QWK {m_inv['qwk']:.3f} | end2end macro-F1 "
              f"{m_e2e['macro_f1']:.3f} (coverage {m_e2e['coverage']:.3f}) ({time.time() - t0:.0f}s)", flush=True)
    tag = os.path.basename(os.path.normpath(cond_dir))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"test_{tag}.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, f"test_{tag}_preds.npz"), building_id=te.building_id.to_numpy(),
                        label=te.label.to_numpy(), **saved)


if __name__ == "__main__":
    main()
