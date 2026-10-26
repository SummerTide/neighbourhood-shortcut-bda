#!/usr/bin/env python
"""Counterfactual results split by disaster type, from harness.py's per-target logs
(results/harness/<run>/harness_counterfactual[_poisson][_vacant]_targets.csv).

Flood labels in xBD are partly defined by the surroundings (Joint Damage Scale: minor =
water surrounding the structure, major = encroaching water), so moving a flooded building
into a dry neighbourhood should remove its damage; the shortcut test is the non-flood part.
Usage: python analysis/cf_by_type.py [--runs changeos_r101 crop_rn50_post ...] [--blend hard|poisson|poissonkeep] [--placement building|vacant]
"""
import argparse
import glob
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from harness import test_buildings  # noqa: E402

COLS = {("concordant damage", "intact context"): "conc. damage -> intact ctx",
        ("concordant intact", "damaged context"): "conc. intact -> damaged ctx",
        ("isolated damage", "damaged context"): "iso. damage -> damaged ctx",
        ("isolated intact", "intact context"): "iso. intact -> intact ctx",
        ("concordant damage", "building gone"): "conc. damage, gone"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="*", default=None)
    ap.add_argument("--blend", default="hard", choices=["hard", "poisson", "poissonkeep"])
    ap.add_argument("--placement", default="building", choices=["building", "vacant"])
    ap.add_argument("--root", default="results/harness")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    suffix = ("" if args.blend == "hard" else f"_{args.blend}") + ("" if args.placement == "building" else "_vacant")
    files = (glob.glob(os.path.join(args.root, "*", f"harness_counterfactual{suffix}_targets.csv")) if not args.runs
             else [os.path.join(args.root, r, f"harness_counterfactual{suffix}_targets.csv") for r in args.runs])
    _, te = test_buildings("data")
    dtype = te.set_index("building_id").disaster_type
    rows = []
    for f in sorted(files):
        if not os.path.exists(f):
            continue
        t = pd.read_csv(f)
        t["events"] = (dtype.loc[t.building_id].to_numpy() == "flooding")
        t["events"] = t.events.map({True: "flood", False: "non-flood"})
        # called damaged: probability of any damage level above 0.5, the rule of cf_plugin.py, so that the
        # counterfactual rates of the diagnosis and of the plug-in table agree
        t["damaged"] = t.p_damaged > 0.5
        g = t.groupby(["events", "group", "variant"]).damaged.mean()
        n = t[t.variant == "building gone"].groupby(["events", "group"]).size()
        for ev in ("non-flood", "flood"):
            rows.append({"run": os.path.basename(os.path.dirname(f)), "events": ev,
                         "n conc. damage": int(n.get((ev, "concordant damage"), 0)),
                         **{name: g.get((ev,) + key, float("nan")) for key, name in COLS.items()}})
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    if args.out:
        t.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
