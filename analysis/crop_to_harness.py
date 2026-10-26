#!/usr/bin/env python
"""Crop models (train_bda.py) in harness.py's format, for analysis/summarize_harness.py.

Standard table from the L0 (as-provided) test predictions; counterfactual table from
analysis/context_swap.py, read from its CSV or, for older runs, parsed from its log. The
crop-level test keeps the same 4 px rim, donors and target groups as the tile-level one,
but swaps the surroundings inside the crop rather than pasting into another tile.
Usage: python analysis/crop_to_harness.py --run results/rn50_post_none --inputs post \
           [--cf-log logs/context_swap_9952440.log] [--name rn50_post]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discordant import categories  # noqa: E402
from harness import standard_table, test_buildings  # noqa: E402

GROUPS = ["isolated damage", "concordant damage", "isolated intact", "concordant intact"]
COLS = ["n", "original", "intact context", "damaged context", "building gone"]


def parse_log(path, run, inputs):
    """The table context_swap.py prints after 'share predicted damaged (<run>, <inputs>)'."""
    lines = open(path).read().splitlines()
    head = f"share predicted damaged ({run}, {inputs})"
    start = [i for i, line in enumerate(lines) if line.strip() == head]
    if not start:
        return None
    rows = []
    for line in lines[start[-1] + 1:start[-1] + 8]:
        for g in GROUPS:
            if line.startswith(g):
                vals = line[len(g):].split()
                rows.append({"targets": g, **dict(zip(COLS, map(float, vals)))})
    return pd.DataFrame(rows) if len(rows) == len(GROUPS) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="train_bda.py results dir")
    ap.add_argument("--inputs", choices=["post", "pre+post"], required=True)
    ap.add_argument("--cond", default="L0_none")
    ap.add_argument("--preds", default="test_conditions_v2_preds.npz")
    ap.add_argument("--cf-log", default=None)
    ap.add_argument("--name", default=None, help="results/harness/<name> (default: the run dir name)")
    ap.add_argument("--data", default="data")
    args = ap.parse_args()
    run = os.path.basename(os.path.normpath(args.run))
    out = os.path.join("results/harness", args.name or run)
    os.makedirs(out, exist_ok=True)
    _, te = test_buildings(args.data)
    z = np.load(os.path.join(args.run, args.preds))
    pos = pd.Series(np.arange(len(z["building_id"])), index=z["building_id"])
    probs = z[args.cond][pos.loc[te.building_id].to_numpy()]
    np.savez_compressed(os.path.join(out, "harness_preds.npz"), building_id=te.building_id.to_numpy(),
                        label=te.label.to_numpy(), probs=probs)
    std, m = standard_table(te, probs, categories(te, 64))
    std.to_csv(os.path.join(out, "harness_standard.csv"), index=False)
    print(f"{run} ({args.cond}): macro-F1 {m['macro_f1']:.3f} QWK {m['qwk']:.3f}")
    csv = os.path.join(args.run, f"context_swap_{args.inputs.replace('+', '_')}.csv")
    cf = pd.read_csv(csv) if os.path.exists(csv) else (parse_log(args.cf_log, run, args.inputs) if args.cf_log else None)
    if cf is not None:
        cf.to_csv(os.path.join(out, "harness_counterfactual.csv"), index=False)
        print(cf.to_string(index=False))
    else:
        print("no counterfactual table found")


if __name__ == "__main__":
    main()
