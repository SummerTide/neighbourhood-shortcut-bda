#!/usr/bin/env python
"""Paper table of the Islahiye case: three representative existing models alone and with the plug-in, on the town
whole labelled scene (the top row of Fig. turkey), on the town centre (region R4) and on the snow-covered outskirts
(the box of Fig. turkey_snow). Whole scene and town centre: macro-F1, level-discordant accuracy and recall of destroyed buildings. Outskirts: false alarms, i.e.
intact buildings predicted damaged or destroyed. Plug-in values are means over the three expert seeds.
Writes ../paper/tables/tab_turkey.tex and results/turkey_regint/paper_numbers.json.
Usage: python analysis/turkey_paper.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from discordant import categories  # noqa: E402
from turkey_summary import MAP, SEEDS, load, lp, scores  # noqa: E402

PROJ = os.path.dirname(HERE)
ROOT, DATA = os.path.join(PROJ, "results/turkey_regint"), os.path.join(PROJ, "data_turkey_regint")
TOWN, SNOW = (8960, 8960, 1536, 1536), (6400, 8704, 1280, 1280)
MODELS = [("changeos_r101", "ChangeOS (R101)"), ("mambabda_small", "MambaBDA-S"), ("xview2_first", "xView2 1st place")]


def inside(te, box):
    x0, y0, w, h = box
    return (te.cx.between(x0, x0 + w) & te.cy.between(y0, y0 + h)).to_numpy()


def main():
    te = pd.read_csv(os.path.join(DATA, "buildings.csv"))
    y, bid = te.label.to_numpy(), te.building_id.to_numpy()
    cat = categories(te, 64)  # strata over the whole scene, then restricted to the box
    town, snow = inside(te, TOWN), inside(te, SNOW)
    scene = np.ones(len(te), bool)
    out = {"scene": {"n": int(scene.sum()), "labels": np.bincount(y, minlength=3).tolist(),
                     "level_discordant": int((cat == "level discordant").sum())},
           "town": {"n": int(town.sum()), "labels": np.bincount(y[town], minlength=3).tolist(),
                    "level_discordant": int((town & (cat == "level discordant")).sum())},
           "snow": {"n": int(snow.sum()), "labels": np.bincount(y[snow], minlength=3).tolist()}, "models": {}}
    lines = []
    for run, name in MODELS:
        p0 = load(ROOT, run, bid)
        preds = {"alone": [MAP[p0.argmax(1)]],
                 "plug": [MAP[(lp(p0) + lp(load(ROOT, t, bid)) + lp(load(ROOT, s, bid))).argmax(1)] for t, s in zip(SEEDS["tight"], SEEDS["swap"])]}
        r = {}
        for v, ps in preds.items():
            r[v] = {}
            for reg, m in (("scene", scene), ("town", town)):
                sc = [scores(p[m], y[m], cat[m]) for p in ps]
                r[v].update({f"{reg}_f1": np.mean([s["f1"] for s in sc]), f"{reg}_level": np.mean([s["level"] for s in sc]),
                             f"{reg}_destroyed": np.mean([s["rec_destroyed"] for s in sc])})
            r[v].update({"false_alarms": np.mean([((p[snow] >= 1) & (y[snow] == 0)).sum() for p in ps])})
        out["models"][name] = r
        for v, lab in (("alone", name), ("plug", "")):
            o = "plug" if v == "alone" else "alone"
            cells = []
            for k, fmt in [(f"{reg}_{m}", "{:.3f}") for reg in ("scene", "town") for m in ("f1", "level", "destroyed")] + [("false_alarms", "{:.1f}")]:
                better = r[v][k] < r[o][k] if k == "false_alarms" else r[v][k] > r[o][k]
                s = fmt.format(r[v][k]).replace(".0", "") if k == "false_alarms" and r[v][k] == int(r[v][k]) else fmt.format(r[v][k])
                cells.append(f"\\textbf{{{s}}}" if better else s)
            lines.append(f"{lab} & {'alone' if v == 'alone' else '+ plug-in'} & " + " & ".join(cells) + " \\\\")
    open(os.path.join(PROJ, "..", "paper", "tables", "tab_turkey.tex"), "w").write("\n".join(lines) + "\n")
    json.dump(out, open(os.path.join(ROOT, "paper_numbers.json"), "w"), indent=1, default=float)
    print("\n".join(lines)); print(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
