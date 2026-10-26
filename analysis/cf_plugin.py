#!/usr/bin/env python
"""Counterfactual flip rates of the plug-in combination, from the per-target logs.

All runs use the same targets and donors (seeded), and each log keeps p(damaged). The 4-class
product of experts is not recoverable from these logs, so this script uses its binary version:
a building is called damaged when the experts' damaged-vs-intact log-odds sum to more than 0.
Non-flood events only. Control: SOTA x context crop (seeds 0 and 2). Plug-in: SOTA x tight x v1
(seeds 1 and 2, the seeds whose counterfactual logs exist).
Usage: python analysis/cf_plugin.py [--blend hard|poisson|poissonkeep] [--out results/harness/cf_plugin.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from harness import test_buildings  # noqa: E402

ROOT = "results/harness"
SOTA = ["strong_baseline", "mambabda_small", "changeos_r101"]
COMBOS = {"alone": [[]],
          "control (+ 2 context)": [["crop_rn50_pre_post", "crop_rn50_pre_post_s2"]],
          "plug-in (+ tight + v1)": [["crop_rn50_pre_post_tight_s1", "crop_rn50_pre_post_pswap_dw5_s1"],
                                     ["crop_rn50_pre_post_tight_s2", "crop_rn50_pre_post_pswap_dw5_s2"]]}
CASES = {"damaged kept in intact context": ("concordant damage", "intact context", 1),
         "intact called damaged in damaged context": ("concordant intact", "damaged context", 1)}


def load(run, blend, placement="building"):
    suffix = ("" if blend == "hard" else f"_{blend}") + ("" if placement == "building" else "_vacant")
    f = os.path.join(ROOT, run, f"harness_counterfactual{suffix}_targets.csv")
    t = pd.read_csv(f)
    p = t.p_damaged.clip(1e-6, 1 - 1e-6)
    t["logit"] = np.log(p) - np.log(1 - p)
    return t.set_index(["building_id", "group", "variant"]).logit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blend", default="hard")
    ap.add_argument("--out", default="results/harness/cf_plugin.csv")
    ap.add_argument("--tex", help="also write the table body as LaTeX")
    ap.add_argument("--placement", default="building", choices=["building", "vacant"])
    ap.add_argument("--sota", nargs="+", default=SOTA)
    a = ap.parse_args()
    _, te = test_buildings("data", "test")
    flood = set(te.building_id[te.disaster_type == "flooding"])
    rows = []
    for s in a.sota:
        base = load(s, a.blend, a.placement)
        for combo, sets in COMBOS.items():
            for runs in sets:
                z = base.copy()
                for r in runs:
                    z = z + load(r, a.blend, a.placement)
                z = z.dropna()
                z = z[~z.index.get_level_values(0).isin(flood)]
                r = {"sota": s, "combo": combo, "experts": "+".join(runs)}
                for name, (g, v, want) in CASES.items():
                    sel = z.xs((g, v), level=("group", "variant"))
                    r[name] = ((sel > 0) == bool(want)).mean()
                r["n per case"] = len(sel)
                rows.append(r)
    t = pd.DataFrame(rows)
    t.to_csv(a.out, index=False)
    pd.set_option("display.width", 250)
    m = t.groupby(["sota", "combo"], sort=False).mean(numeric_only=True)
    print(m.round(3).to_string())
    if a.tex:  # the paper table: each model alone and with the plug-in (the control stays in the CSV)
        names = {"strong_baseline": "Strong Baseline", "mambabda_small": "MambaBDA-S", "changeos_r101": "ChangeOS R101",
                 "changeos_swint": "ChangeOS Swin-T", "disasteradaptivenet": "DisasterAdaptiveNet",
                 "xview2_first_light": "xView2 winning solution", "changestar_changen2": "ChangeStar"}
        rows = []  # one row per model: kept alone, kept with the plug-in, flip alone, flip with the plug-in
        for s in a.sota:
            al = m.loc[(s, "alone")]
            pl = m.loc[(s, next(c for c in m.loc[s].index if c.startswith("plug-in")))]
            rows.append(f"{names[s]} & {al.iloc[0]:.2f} & {pl.iloc[0]:.2f} & {al.iloc[1]:.2f} & {pl.iloc[1]:.2f} \\\\")
        open(a.tex, "w").write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
