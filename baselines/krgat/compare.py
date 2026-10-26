#!/usr/bin/env python
"""Side-by-side harness numbers: the KR-GAT rows next to the other harness models and the
footprint_bda crop baselines (scored from their L0_none test predictions with the same strata).
Writes results/comparison.csv and results/counterfactual_comparison.csv."""
import os

import numpy as np
import pandas as pd

from common import PROJ, ROOT

STRATA = ["concordant", "mixed", "isolated", "level discordant", "isolated intact", "isolated damage"]


def summarise(name, std):
    a = std.set_index("stratum")
    r = {"model": name, "macro_f1": a.loc["all", "macro_f1"], "qwk": a.loc["all", "qwk"],
         **{f"F1_{k}": a.loc["all", f"f1_{k}"] for k in range(4)}}
    for s in STRATA:
        r[f"acc {s}"] = a.loc[s, "acc"]
    for l in ("minor", "major", "destroyed"):
        r[f"isolated damage missed {l}"] = a.loc["isolated damage", f"missed_{l}"]
    return r


def main():
    import sys
    sys.path.insert(0, f"{PROJ}/analysis")
    from bda_data import load_tables
    from krgat import harness_tables
    from discordant import categories
    rows, cfs = [], []
    hdir = f"{PROJ}/results/harness"
    for name in sorted(os.listdir(hdir)):
        p = f"{hdir}/{name}/harness_standard.csv"
        if os.path.exists(p):
            rows.append(summarise(name, pd.read_csv(p)))
        c = f"{hdir}/{name}/harness_counterfactual.csv"
        if os.path.exists(c):
            cfs.append(pd.read_csv(c).assign(model=name))
    scenes, b = load_tables(f"{PROJ}/data")
    te = b[(b.official == "test") & (b.label >= 0)].reset_index(drop=True)
    cat = categories(te, 64)
    for rdir in ("rn50_post_none", "rn50_pre_post_none"):
        z = np.load(f"{PROJ}/results/{rdir}/test_conditions_v2_preds.npz")
        assert (z["building_id"] == te.building_id.to_numpy()).all()
        std, _ = harness_tables(te, z["L0_none"], cat)
        rows.append(summarise(f"crop {rdir}", std))
    df = pd.DataFrame(rows)
    df.to_csv(f"{ROOT}/results/comparison.csv", index=False)
    pd.set_option("display.width", 250)
    print(df.set_index("model").T.round(3).to_string())
    if cfs:
        cf = pd.concat(cfs)
        cf.to_csv(f"{ROOT}/results/counterfactual_comparison.csv", index=False)
        print(cf.set_index(["model", "targets"]).round(3).to_string())


if __name__ == "__main__":
    main()
