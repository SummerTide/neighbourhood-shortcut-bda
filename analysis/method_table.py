#!/usr/bin/env python
"""Compare methods on the xBD test buildings from harness_preds.npz, including post-hoc
combinations, with tile-level bootstrap intervals.

A spec is a harness run name (results/harness/<name>), or a combination
'geo:<a>+<b>[+...]' (normalised geometric mean of probabilities, i.e. a product of experts)
or 'mean:<a>+<b>[+...]' (arithmetic mean, an ensemble).
Columns: macro-F1, accuracy on concordant buildings, isolated damage, isolated intact, their
mean ("discordant balanced": both directions of a context conflict, 0.5 for a model that calls
everything damaged), level discordant, mixed, and isolated destroyed predicted intact.
--bootstrap N resamples test tiles (scenes) N times and reports 95% intervals for macro-F1 and
discordant balanced accuracy, and for the difference of each method from the first one.
Usage: python analysis/method_table.py crop_rn50_post geo:crop_rn50_post+rn50_post_tight [--bootstrap 1000]
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


def load(name, te, root):
    z = np.load(os.path.join(root, name, "harness_preds.npz"))
    assert (z["building_id"] == te.building_id.to_numpy()).all(), name
    return z["probs"].astype(np.float64)


def probs_of(spec, te, root):
    if ":" not in spec:
        return load(spec, te, root)
    how, names = spec.split(":", 1)
    ps = [load(n, te, root) for n in names.split("+")]
    if how == "mean":
        return np.mean(ps, 0)
    g = np.exp(np.mean([np.log(p + 1e-6) for p in ps], 0))
    return g / g.sum(1, keepdims=True)


def scores(y, pred, cat, idx=None):
    if idx is not None:
        y, pred, cat = y[idx], pred[idx], cat[idx]
    acc = {c: (pred[cat == c] == y[cat == c]).mean() for c in ("concordant", "isolated damage", "isolated intact",
                                                                "level discordant", "mixed")}
    iso_d = (cat == "isolated damage") & (y == 3)
    return {"macro-F1": f1_score(y, pred, average="macro", labels=[0, 1, 2, 3]),
            "concordant": acc["concordant"], "iso damage": acc["isolated damage"], "iso intact": acc["isolated intact"],
            "discordant balanced": (acc["isolated damage"] + acc["isolated intact"]) / 2,
            "level discordant": acc["level discordant"], "mixed": acc["mixed"],
            "iso destroyed -> intact": (pred[iso_d] == 0).mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("specs", nargs="+")
    ap.add_argument("--root", default="results/harness")
    ap.add_argument("--bootstrap", type=int, default=0)
    ap.add_argument("--out", default=None, help="write the table as CSV")
    args = ap.parse_args()
    _, te = test_buildings("data")
    cat, y = categories(te, 64), te.label.to_numpy()
    preds = {s: probs_of(s, te, args.root).argmax(1) for s in args.specs}
    rows = [{"method": s, **scores(y, p, cat)} for s, p in preds.items()]
    t = pd.DataFrame(rows)
    if args.bootstrap:
        rng = np.random.RandomState(0)
        tiles = te.groupby("scene_id").indices
        keys = list(tiles)
        boot = {s: {"macro-F1": [], "discordant balanced": []} for s in args.specs}
        for _ in range(args.bootstrap):
            idx = np.concatenate([tiles[keys[k]] for k in rng.choice(len(keys), len(keys))])
            for s, p in preds.items():
                r = scores(y, p, cat, idx)
                for m in boot[s]:
                    boot[s][m].append(r[m])
        base = args.specs[0]
        for m in ("macro-F1", "discordant balanced"):
            t[f"{m} 95% CI"] = [f"[{np.percentile(boot[s][m], 2.5):.3f}, {np.percentile(boot[s][m], 97.5):.3f}]"
                                for s in args.specs]
            d = {s: np.array(boot[s][m]) - np.array(boot[base][m]) for s in args.specs}
            t[f"{m} diff vs first, 95% CI"] = [f"[{np.percentile(d[s], 2.5):+.3f}, {np.percentile(d[s], 97.5):+.3f}]"
                                               for s in args.specs]
    pd.set_option("display.width", 250)
    print(t.round(3).to_string(index=False))
    if args.out:
        t.to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
