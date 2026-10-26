#!/usr/bin/env python
"""Islahiye (Turkiye 2023 earthquake) results of turkey_eval.py: each existing model alone and with
the building-evidence plug-in, on the BRIGHT recommended instances (3 classes: intact / damaged /
destroyed; xBD levels mapped as no -> intact, minor + major -> damaged, destroyed -> destroyed).

Variants
  alone (sum3)     the model's 4 level probabilities (normalised to sum 1) summed into the 3 classes
  alone (argmax4)  argmax over the 4 levels, then mapped
  + plug-in (4L)   log p_existing + log p_tight(k) + log p_swap(k) over the 4 xBD levels (log(p + 1e-6),
                   as analysis/paper_tables.py), argmax, then mapped; seeds k = 0, 1, 2
  + plug-in (3C)   each model's probabilities first summed into 3 classes, then the same product
  + control (4L)   log p_existing + log p_context(k) + log p_context(k+1): equal-size product without
                   the building-evidence experts
No hazard gating (earthquake). Neighbourhood strata: analysis/discordant.categories(te, 64) on the
scored buildings as one scene (damaged = label >= 1, 'concordant' = >= 75 % of neighbours share
the exact 3-class label); --neighbours all also counts the non-recommended labelled records as
neighbours (sensitivity), the scored set staying the same.
Metrics: macro-F1 over the 3 classes, accuracy on concordant buildings, isolated damage detected
(pred >= damaged), isolated intact kept intact, their mean (discordant balanced), level-discordant
accuracy, per-class recall.
Usage: python analysis/turkey_summary.py [--root results/turkey] [--neighbours recommended|all]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from discordant import categories  # noqa: E402

PROJ = os.path.dirname(HERE)
EXISTING = [("changeos_r101", "ChangeOS (R101)"), ("changeos_swint", "ChangeOS (Swin-T)"),
            ("changestar_changen2", "ChangeStar (Changen2)"), ("mambabda_small", "MambaBDA-S"),
            ("strong_baseline", "Strong Baseline"), ("disasteradaptivenet", "DisasterAdaptiveNet"),
            ("xview2_first", "xView2 1st place"), ("xview2_first_light", "xView2 1st place (light)"),
            ("crop_rn50_pre_post", "ResNet-50 crop (context)")]
SEEDS = {"context": ["crop_rn50_pre_post", "crop_rn50_pre_post_s1", "crop_rn50_pre_post_s2"],
         "tight": ["crop_rn50_pre_post_tight", "crop_rn50_pre_post_tight_s1", "crop_rn50_pre_post_tight_s2"],
         "swap": ["crop_rn50_pre_post_pswap_dw5", "crop_rn50_pre_post_pswap_dw5_s1", "crop_rn50_pre_post_pswap_dw5_s2"]}
MAP = np.array([0, 1, 1, 2])
M = np.zeros((4, 3))
M[np.arange(4), MAP] = 1
KEYS = ["f1", "conc", "iso_det", "iso_kept", "disc", "level", "rec_intact", "rec_damaged", "rec_destroyed"]


def load(root, run, bid):
    f = os.path.join(root, run, "turkey_preds.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    assert (z["building_id"] == bid).all(), run
    return z["probs"].astype(np.float64)


def lp(p):
    return np.log(p + 1e-6)


def lp3(p):
    return np.log((p / p.sum(1, keepdims=True)) @ M + 1e-6)


def scores(pred, y, cat):
    d, i, ld, c = (cat == "isolated damage"), (cat == "isolated intact"), (cat == "level discordant"), (cat == "concordant")
    r = {"f1": f1_score(y, pred, average="macro", labels=[0, 1, 2]), "conc": (pred[c] == y[c]).mean(),
         "iso_det": (pred[d] >= 1).mean(), "iso_kept": (pred[i] == 0).mean(), "level": (pred[ld] == y[ld]).mean()}
    r["disc"] = (r["iso_det"] + r["iso_kept"]) / 2
    for k, n in enumerate(["intact", "damaged", "destroyed"]):
        r[f"rec_{n}"] = (pred[y == k] == k).mean()
    return {k: float(v) for k, v in r.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(PROJ, "results", "turkey"))
    ap.add_argument("--data", default=os.path.join(PROJ, "data_turkey"))
    ap.add_argument("--neighbours", choices=["recommended", "all"], default="recommended")
    a = ap.parse_args()
    te = pd.read_csv(os.path.join(a.data, "buildings.csv"))
    y, bid = te.label.to_numpy(), te.building_id.to_numpy()
    if a.neighbours == "all":
        allb = pd.read_csv(os.path.join(a.data, "buildings_all.csv"))
        cat = pd.Series(categories(allb, 64), index=allb.building_id).loc[bid].to_numpy()
    else:
        cat = categories(te, 64)
    strata = pd.crosstab(pd.Series(cat, name="stratum"), pd.Series(y, name="label").map({0: "intact", 1: "damaged", 2: "destroyed"}),
                         margins=True)
    print(f"== {len(y)} scored buildings; strata at R = 64 px (neighbours: {a.neighbours})\n{strata}\n")
    P = {r: load(a.root, r, bid) for r in {r for v in SEEDS.values() for r in v} | {r for r, _ in EXISTING}}
    missing = sorted(r for r, p in P.items() if p is None)
    if missing:
        print("missing runs:", missing)
    rows = []

    def add(model, variant, preds):
        rs = [scores(p, y, cat) for p in preds]
        row = {"model": model, "variant": variant, "seeds": len(rs)}
        for k in KEYS:
            v = [r[k] for r in rs]
            row[k], row[k + "_sd"] = float(np.mean(v)), (float(np.std(v)) if len(v) > 1 else np.nan)
        rows.append(row)

    have = {k: all(P[r] is not None for r in v) for k, v in SEEDS.items()}
    for k in ("context", "tight", "swap"):
        if have[k]:
            add(f"expert: {k} crop", "alone (argmax4)", [MAP[P[r].argmax(1)] for r in SEEDS[k]])
    if have["tight"] and have["swap"]:
        add("experts: tight x swap", "product (4L)",
            [MAP[(lp(P[t]) + lp(P[s])).argmax(1)] for t, s in zip(SEEDS["tight"], SEEDS["swap"])])
    for run, name in EXISTING:
        p0 = P[run]
        if p0 is None:
            continue
        add(name, "alone (sum3)", [lp3(p0).argmax(1)])
        add(name, "alone (argmax4)", [MAP[p0.argmax(1)]])
        if have["tight"] and have["swap"]:
            add(name, "+ plug-in (4L)", [MAP[(lp(p0) + lp(P[t]) + lp(P[s])).argmax(1)]
                                         for t, s in zip(SEEDS["tight"], SEEDS["swap"])])
            add(name, "+ plug-in (3C)", [(lp3(p0) + lp3(P[t]) + lp3(P[s])).argmax(1)
                                         for t, s in zip(SEEDS["tight"], SEEDS["swap"])])
        if have["context"]:
            c = SEEDS["context"]
            add(name, "+ control (4L)", [MAP[(lp(p0) + lp(P[c[k]]) + lp(P[c[(k + 1) % 3]])).argmax(1)] for k in range(3)])
    t = pd.DataFrame(rows)
    suffix = "" if a.neighbours == "recommended" else "_nbrs_all"
    t.to_csv(os.path.join(a.root, f"summary{suffix}.csv"), index=False)
    strata.to_csv(os.path.join(a.root, f"strata{suffix}.csv"))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 200)
    show = t[["model", "variant"]].copy()
    for k in KEYS:
        show[k] = [f"{m:.3f}" if np.isnan(s) else f"{m:.3f}±{s:.3f}" for m, s in zip(t[k], t[k + "_sd"])]
    print(show.to_string(index=False))
    with open(os.path.join(a.root, f"summary{suffix}.md"), "w") as fh:
        fh.write(f"Strata (neighbours: {a.neighbours}):\n\n" + strata.to_markdown() + "\n\n" + show.to_markdown(index=False) + "\n")
    json.dump({"n": int(len(y)), "strata": {k: int(v) for k, v in pd.Series(cat).value_counts().items()},
               "missing": missing}, open(os.path.join(a.root, f"summary{suffix}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
