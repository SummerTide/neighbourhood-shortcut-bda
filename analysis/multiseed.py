#!/usr/bin/env python
"""Mean and spread over seeds for the ResNet-50 crop configurations, all buildings and
non-flood events (where the surroundings are not part of the label definition).

Configurations (seed 0, 1, 2 -> results/harness/<run>):
  context   the usual crop;  tight  building + 4 px rim, zoomed;  v1  Poisson event swap +
  discordance x5;  context x tight / v1 x tight  geometric mean of the same-seed pair;
  context x context / v1 x v1  the same two-model combination over pairs of seeds (control).
Metrics: macro-F1, concordant accuracy, isolated damage detected (any level), isolated intact
kept intact, their mean (discordant balanced), level-discordant accuracy.
Usage: python analysis/multiseed.py [--inputs post pre+post] [--out results/harness/multiseed.csv]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discordant import categories  # noqa: E402
from harness import test_buildings  # noqa: E402
from sklearn.metrics import f1_score  # noqa: E402

RUNS = {  # config -> run per seed, with {inp} = post | pre_post
    "context": ["crop_rn50_{inp}", "crop_rn50_{inp}_s1", "crop_rn50_{inp}_s2"],
    "tight": ["rn50_{inp}_tight", "crop_rn50_{inp}_tight_s1", "crop_rn50_{inp}_tight_s2"],
    "v1": ["crop_rn50_{inp}_pswap_dw5", "crop_rn50_{inp}_pswap_dw5_s1", "crop_rn50_{inp}_pswap_dw5_s2"],
}
FUSED = {"context x tight": ("context", "tight"), "v1 x tight": ("v1", "tight")}


def probs(run, te, root):
    f = os.path.join(root, run, "harness_preds.npz")
    if not os.path.exists(f):
        return None
    z = np.load(f)
    assert (z["building_id"] == te.building_id.to_numpy()).all(), run
    return z["probs"].astype(np.float64)


def geo(a, b):
    g = np.sqrt((a + 1e-6) * (b + 1e-6))
    return g / g.sum(1, keepdims=True)


def scores(p, y, cat, keep):
    pred = p.argmax(1)
    r = {"macro-F1": f1_score(y[keep], pred[keep], average="macro", labels=[0, 1, 2, 3])}
    for name, c in [("concordant", "concordant"), ("level discordant", "level discordant")]:
        s = keep & (cat == c)
        r[name] = (pred[s] == y[s]).mean()
    d, i = keep & (cat == "isolated damage"), keep & (cat == "isolated intact")
    r["iso damage detected"], r["iso intact kept"] = (pred[d] >= 1).mean(), (pred[i] == 0).mean()
    r["discordant balanced"] = (r["iso damage detected"] + r["iso intact kept"]) / 2
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=["post", "pre_post"])
    ap.add_argument("--root", default="results/harness")
    ap.add_argument("--out", default="results/harness/multiseed.csv")
    args = ap.parse_args()
    _, te = test_buildings("data")
    cat, y = categories(te, 64), te.label.to_numpy()
    nonflood = (te.disaster_type != "flooding").to_numpy()
    rows = []
    for inp in args.inputs:
        loaded = {c: [probs(r.format(inp=inp), te, args.root) for r in runs] for c, runs in RUNS.items()}
        for c, (a, b) in FUSED.items():
            loaded[c] = [geo(x, z) if x is not None and z is not None else None
                         for x, z in zip(loaded[a], loaded[b])]
        for c in ("context", "v1"):  # two-model ensemble controls: pairs of different seeds
            ps = loaded[c]
            loaded[f"{c} x {c} (2 seeds)"] = [geo(ps[i], ps[j]) for i, j in ((0, 1), (1, 2), (2, 0))
                                              if ps[i] is not None and ps[j] is not None]
        for c, ps in loaded.items():
            for events, keep in [("all", np.ones(len(y), bool)), ("non-flood", nonflood)]:
                per_seed = [scores(p, y, cat, keep) for p in ps if p is not None]
                if not per_seed:
                    continue
                t = pd.DataFrame(per_seed)
                row = {"inputs": inp, "config": c, "events": events, "seeds": len(per_seed)}
                for m in t.columns:
                    row[m] = t[m].mean()
                    row[m + " sd"] = t[m].std(ddof=1) if len(t) > 1 else np.nan
                rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    show = ["inputs", "config", "events", "seeds", "macro-F1", "macro-F1 sd", "concordant", "iso damage detected",
            "iso intact kept", "discordant balanced", "discordant balanced sd", "level discordant"]
    pd.set_option("display.width", 250)
    print(out[show].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
