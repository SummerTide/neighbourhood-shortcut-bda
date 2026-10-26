#!/usr/bin/env python
"""Where does misregistration hurt? Test-set scores of saved predictions by stratum.

Strata (computed from the ground truth of the labelled test buildings):
  neighbourhood  share of the other buildings within RADIUS px of the centroid
                 that carry the same damage label: concordant >= 0.75,
                 mixed 0.25-0.75, discordant < 0.25, isolated = no neighbour
  size           footprint area terciles
  disaster       xBD disaster type
Usage: python analysis/stratify.py [--tag conditions_v2] [--results results]
(--tag conditions reads the protocol-v1 predictions)
"""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

RADIUS = 64.0
MODELS = {  # name -> (results dir, key pattern for a condition's class probabilities)
    "rn50_post": ("rn50_post_none", "{}"),
    "rn50_post_jit8": ("rn50_post_none_jit8", "{}"),
    "rn50_post_jit16": ("rn50_post_none_jit16", "{}"),
    "rn50_pre_post": ("rn50_pre_post_none", "{}"),
    "rn50_pre_post_jit8": ("rn50_pre_post_none_jit8", "{}"),
    "rn50_pre_post_jit16": ("rn50_pre_post_none_jit16", "{}"),
    "rn50_post_tight": ("rn50_post_tight", "{}"),
    "rn50_pre_post_tight": ("rn50_pre_post_tight", "{}"),
    "changeos_inventory": ("changeos_r101", "{}_inventory"),
    "dinov3sat_mlp_post": ("dinov3sat_mlp_post_none", "{}"),
    "dinov3sat_mlp_post_pre": ("dinov3sat_mlp_post_pre_none", "{}"),
}
CONDITIONS = ["L0_none", "L0_dense", "L1_none", "L2_none", "L3_none", "L4_none", "L5_none", "L3_global", "L3_dense",
              "L5_global", "L5_dense", "blur_linear", "blur_cubic"]


def neighbourhood(te):
    agree = np.full(len(te), np.nan)
    for _, idx in te.groupby("scene_id").indices.items():
        xy = te[["cx", "cy"]].to_numpy()[idx]
        lab = te.label.to_numpy()[idx]
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        near = (d < RADIUS) & ~np.eye(len(idx), dtype=bool)
        n = near.sum(1)
        same = (near & (lab[:, None] == lab[None, :])).sum(1)
        agree[idx] = np.where(n > 0, same / np.maximum(n, 1), np.nan)
    return np.select([np.isnan(agree), agree < 0.25, agree < 0.75], ["isolated", "discordant", "mixed"], "concordant")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--results", default="results")
    ap.add_argument("--tag", default="conditions_v2", help="condition set the predictions were made on")
    args = ap.parse_args()
    b = pd.read_csv(os.path.join(args.data, "buildings.csv"), usecols=["building_id", "scene_id", "label", "area", "cx", "cy"])
    scenes = pd.read_csv(os.path.join(args.data, "scenes.csv"), usecols=["scene_id", "official", "disaster_type"])
    te = b.merge(scenes, on="scene_id")
    te = te[(te.official == "test") & (te.label >= 0)].reset_index(drop=True)
    strata = {
        "neighbourhood": neighbourhood(te),
        "size": pd.qcut(te.area, 3, labels=["small", "medium", "large"]).astype(str).to_numpy(),
        "disaster": te.disaster_type.to_numpy(),
    }
    y = te.label.to_numpy()
    rows = []
    for model, (rdir, key) in MODELS.items():
        path = os.path.join(args.results, rdir, "test_preds.npz" if args.tag == "conditions" else f"test_{args.tag}_preds.npz")
        if not os.path.exists(path):
            continue
        z = np.load(path)
        assert (z["building_id"] == te.building_id.to_numpy()).all()
        for cond in CONDITIONS:
            if key.format(cond) not in z:
                continue
            pred = z[key.format(cond)].argmax(1)
            for kind, values in [("all", np.full(len(te), "all"))] + list(strata.items()):
                for s in pd.unique(values):
                    m = values == s
                    f1 = f1_score(y[m], pred[m], labels=range(4), average=None, zero_division=0)
                    present = np.bincount(y[m], minlength=4) > 0
                    rows.append({"model": model, "condition": cond, "stratum": kind, "value": s, "n": int(m.sum()),
                                 "acc": float((pred[m] == y[m]).mean()), "macro_f1": float(f1[present].mean())})
    df = pd.DataFrame(rows)
    df.to_csv(f"analysis/stratify_{args.tag}.csv", index=False)
    pd.set_option("display.width", 200)
    for kind, metric in [("all", "macro_f1"), ("neighbourhood", "macro_f1"), ("all", "acc"), ("neighbourhood", "acc"),
                         ("size", "acc"), ("disaster", "acc")]:
        t = df[df.stratum == kind].pivot_table(index=["model", "value"], columns="condition", values=metric)
        t = t[[c for c in CONDITIONS if c in t.columns]]
        t.insert(0, "n", df[df.stratum == kind].groupby(["model", "value"]).n.first())
        print(f"\n== {metric} by {kind}")
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()
