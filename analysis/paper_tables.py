#!/usr/bin/env python
"""Numbers and LaTeX tables for the paper, all recomputed from the saved test predictions
(results/harness/<run>/harness_preds.npz) so that every value in the manuscript is reproducible.

Writes <out>/tab_baselines.tex, tab_resnet.tex, tab_plugin.tex and <out>/numbers.json.
  baselines  macro-F1, concordant accuracy, isolated damage detected (any level), isolated intact
             kept, discordant balanced, level-discordant accuracy; non-flood counterfactuals from
             results/harness/cf_by_type.csv; xFBD pasted detection from results/harness_xfbd.
  resnet     ResNet-50 crop configurations, mean (sd) over 3 seeds, pre+post.
  plugin     SOTA x tight x swap-trained against the equal-size control SOTA x context x context
             (product of experts, same seed pairing as analysis/gated_experts.py), 3 seeds, pre+post
             experts; also the relative error reduction on discordant buildings.
Usage: python analysis/paper_tables.py --out ../paper/tables [--cf results/harness/cf_by_type_poisson_vacant.csv]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from discordant import categories  # noqa: E402
from harness import test_buildings  # noqa: E402
from summarize_harness import xfbd_columns  # noqa: E402

ROOT = "results/harness"
BASELINES = [  # run, display name, paradigm, xFBD run (xView2: the 4-network variant without TTA)
    ("changeos_r101", "ChangeOS (R101)", "Object-level change detection", "changeos_r101"),
    ("changeos_swint", "ChangeOS (Swin-T)", "Object-level change detection", "changeos_swint"),
    ("xview2_first", "xView2 1st place", "Pixel segmentation ensemble", "xview2_first_light"),
    ("changestar_changen2", "ChangeStar (Changen2)", "Synthetic-change pre-training", "changestar_changen2"),
    ("strong_baseline", "Strong Baseline", "Pixel segmentation", "strong_baseline"),
    ("disasteradaptivenet", "DisasterAdaptiveNet", "Disaster-conditioned segmentation", "disasteradaptivenet"),
    ("mambabda_small", "MambaBDA-S", "State-space change detection", "mambabda_small"),
    ("damage_triageformer", "Damage-TriageFormer", "Footprint-conditioned crop (DINOv3)", None),
    ("krgat", "KR-GAT", "Building graph network", None),
]
SEEDS = {"context": ["crop_rn50_{i}", "crop_rn50_{i}_s1", "crop_rn50_{i}_s2"],
         "tight": ["rn50_{i}_tight", "crop_rn50_{i}_tight_s1", "crop_rn50_{i}_tight_s2"],
         "v1": ["crop_rn50_{i}_pswap_dw5", "crop_rn50_{i}_pswap_dw5_s1", "crop_rn50_{i}_pswap_dw5_s2"]}
SOTA = [("strong_baseline", "Strong Baseline"), ("mambabda_small", "MambaBDA-S"),
        ("xview2_first", "xView2 1st place"), ("changeos_r101", "ChangeOS (R101)")]
XFBD_SOTA = {"xview2_first": "xview2_first_light"}  # xFBD was run with the 4-network variant without TTA
XFBD_SEEDS = {"context": ["crop_rn50_pre_post", "crop_rn50_pre_post_s1", "crop_rn50_pre_post_s2"],
              "tight": ["crop_rn50_pre_post_tight", "crop_rn50_pre_post_tight_s1", "crop_rn50_pre_post_tight_s2"],
              "v1": ["crop_rn50_pre_post_pswap_dw5", "crop_rn50_pre_post_pswap_dw5_s1", "crop_rn50_pre_post_pswap_dw5_s2"]}


def logp(run, te):
    z = np.load(os.path.join(ROOT, run, "harness_preds.npz"))
    assert (z["building_id"] == te.building_id.to_numpy()).all(), run
    return np.log(z["probs"].astype(np.float64) + 1e-6)


def xfbd_logp(run):
    z = np.load(os.path.join("results/harness_xfbd", run, "harness_preds.npz"))
    return z["building_id"], np.log(z["probs"].astype(np.float64) + 1e-6)


def xfbd_scores(pred, bid, flags):
    """Pasted damaged buildings called damaged, and intact buildings within 64 px of them kept intact."""
    f = flags.loc[bid]
    pasted = f.pasted.to_numpy(bool)
    near = ~pasted & (f.dist_pasted.to_numpy() <= 64)
    return {"xfbd_det": float((pred[pasted] >= 1).mean()), "xfbd_int": float((pred[near] == 0).mean())}


def scores(pred, y, cat):
    d, i, ld, c = (cat == "isolated damage"), (cat == "isolated intact"), (cat == "level discordant"), (cat == "concordant")
    r = {"f1": f1_score(y, pred, average="macro", labels=[0, 1, 2, 3]), "conc": (pred[c] == y[c]).mean(),
         "iso_det": (pred[d] >= 1).mean(), "iso_kept": (pred[i] == 0).mean(), "level": (pred[ld] == y[ld]).mean()}
    r["disc"] = (r["iso_det"] + r["iso_kept"]) / 2
    return r


def f3(x):
    return f"{x:.3f}"


def ms(v):  # mean over the expert seeds; the spread is small and not shown
    return f"{np.mean(v):.3f}"


def bold_best(cols):
    """cols: per column, a list of values (None for missing); returns the formatted cells, the best of
    each column in bold (all measures here are higher-is-better)."""
    out = []
    for vals in cols:
        ok = [v for v in vals if v is not None and not pd.isna(v)]
        best = max(ok) if ok else None
        out.append(["--" if v is None or pd.isna(v) else (f"\\textbf{{{v:.3f}}}" if f"{v:.3f}" == f"{best:.3f}" else f"{v:.3f}") for v in vals])
    return [list(r) for r in zip(*out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cf", default=os.path.join(ROOT, "cf_by_type_vacant.csv"),
                    help="open-ground counterfactuals from cf_by_type.py (Poisson: results/harness/cf_by_type_poisson_vacant.csv)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    _, te = test_buildings("data", "test")
    y, cat = te.label.to_numpy(), categories(te, 64)
    nums = {"n_test": int(len(y)), "strata": {k: int(v) for k, v in pd.Series(cat).value_counts().items()}}
    cf = pd.read_csv(a.cf)  # open-ground counterfactuals
    cf = cf[cf.events == "non-flood"].set_index("run")
    flags = pd.read_csv("data_xfbd_test/buildings.csv", usecols=["building_id", "pasted", "dist_pasted"]).set_index("building_id")

    # Table: baselines
    rows, nb = [], {}
    groups = {"changeos_r101": "Footprints used for scoring only (pixel outputs averaged over each footprint)",
              "damage_triageformer": "Footprints also used as input"}
    for run, name, paradigm, sname in BASELINES:
        if run in groups:
            rows.append(("\\midrule\n" if rows else "") + f"\\multicolumn{{9}}{{l}}{{\\emph{{{groups[run]}}}}} \\\\")
        r = scores(logp(run, te).argmax(1), y, cat)
        ck = run if run in cf.index else sname  # xView2: the counterfactuals ran with the 4-network variant
        r["cf_dmg_kept"] = cf.loc[ck, "conc. damage -> intact ctx"] if ck in cf.index else np.nan
        r["cf_intact_flip"] = cf.loc[ck, "conc. intact -> damaged ctx"] if ck in cf.index else np.nan
        r["xfbd"] = xfbd_columns(sname, "results/harness_xfbd", flags).get("xFBD pasted -> damaged", np.nan) if sname else np.nan
        nb[run] = {k: (None if pd.isna(v) else float(v)) for k, v in r.items()}
        cell = lambda v, fmt="{:.3f}": "--" if pd.isna(v) else fmt.format(v)  # noqa: E731
        rows.append(f"{name} & {f3(r['f1'])} & {f3(r['conc'])} & {f3(r['iso_det'])} & {f3(r['iso_kept'])} & "
                    f"{f3(r['disc'])} & {cell(r['cf_dmg_kept'], '{:.2f}')} & {cell(r['cf_intact_flip'], '{:.2f}')} & "
                    f"{cell(r['xfbd'], '{:.2f}')} \\\\")
    nums["baselines"] = nb
    open(os.path.join(a.out, "tab_baselines.tex"), "w").write("\n".join(rows) + "\n")

    # Table: ResNet configurations (pre+post), 3 seeds
    inp = "pre_post"
    L = {k: [logp(r.format(i=inp), te) for r in v] for k, v in SEEDS.items()}
    configs = {
        "Context crop": [L["context"][s] for s in range(3)],
        "Tight crop": [L["tight"][s] for s in range(3)],
        "Swap-trained": [L["v1"][s] for s in range(3)],
        "Context $\\times$ context (control)": [L["context"][s] + L["context"][(s + 1) % 3] for s in range(3)],
        "Context $\\times$ context $\\times$ context (control)": [sum(L["context"]) for _ in range(1)],
        "Context $\\times$ tight": [L["context"][s] + L["tight"][s] for s in range(3)],
        "Swap-trained $\\times$ tight": [L["v1"][s] + L["tight"][s] for s in range(3)],
        "Context $\\times$ tight $\\times$ swap-trained (ours)": [L["context"][s] + L["tight"][s] + L["v1"][s] for s in range(3)],
    }
    rows, nr, K = [], {}, ["f1", "conc", "iso_det", "iso_kept", "disc", "level"]
    for name, lps in configs.items():
        rs = [scores(lp.argmax(1), y, cat) for lp in lps]
        nr[name] = {k: [float(r[k]) for r in rs] for k in rs[0]}
    # grouped: single models, ensemble controls, combinations with the experts; our plug-in in bold
    first_of = {"Context $\\times$ context (control)", "Context $\\times$ tight"}
    for name in configs:
        c = [f3(np.mean(nr[name][k])) for k in K]
        if "(ours)" in name:
            name, c = f"\\textbf{{{name}}}", [f"\\textbf{{{v}}}" for v in c]
        rows.append(("\\midrule\n" if name in first_of else "") + f"{name} & " + " & ".join(c) + " \\\\")
    nums["resnet_pre_post"] = nr
    open(os.path.join(a.out, "tab_resnet.tex"), "w").write("\n".join(rows) + "\n")

    # Table: the plug-in on every diagnosed model, used per hazard (flood events keep the existing model
    # alone, since the Joint Damage Scale defines flood damage through the surroundings); the same
    # combinations on xFBD (Palu earthquake, no flood). The equal-size control is kept in numbers.json.
    X = {k: [xfbd_logp(r) for r in v] for k, v in XFBD_SEEDS.items()}
    xb = X["context"][0][0]
    assert all((b == xb).all() for v in X.values() for b, _ in v)
    XL = {k: [l for _, l in v] for k, v in X.items()}
    flood = (te.disaster_type == "flooding").to_numpy()[:, None]
    groups = {"changeos_r101": "Footprints used for scoring only", "damage_triageformer": "Footprints also used as input"}
    rows, npl = [], {}
    for run, name, _, xrun in BASELINES:
        if run in groups:
            rows.append(("\\midrule\n" if rows else "") + f"\\multicolumn{{10}}{{l}}{{\\emph{{{groups[run]}}}}} \\\\")
        s0 = logp(run, te)
        x0 = None
        if xrun and os.path.exists(os.path.join("results/harness_xfbd", xrun, "harness_preds.npz")):
            b, x0 = xfbd_logp(xrun)
            assert (b == xb).all(), run
        xs = (lambda lp: xfbd_scores(lp.argmax(1), xb, flags)) if x0 is not None else (lambda lp: {})
        gate = lambda lp: np.where(flood, s0, lp)  # noqa: E731
        ctrl = [{**scores(gate(s0 + L["context"][s] + L["context"][(s + 1) % 3]).argmax(1), y, cat),
                 **xs(x0 + XL["context"][s] + XL["context"][(s + 1) % 3] if x0 is not None else None)} for s in range(3)]
        plug = [{**scores(gate(s0 + L["tight"][s] + L["v1"][s]).argmax(1), y, cat),
                 **xs(x0 + XL["tight"][s] + XL["v1"][s] if x0 is not None else None)} for s in range(3)]
        alone = {**scores(s0.argmax(1), y, cat), **xs(x0)}
        e_a, e_c, e_p = 1 - alone["disc"], 1 - np.mean([r["disc"] for r in ctrl]), 1 - np.mean([r["disc"] for r in plug])
        npl[run] = {"alone": {k: float(v) for k, v in alone.items()},
                    "control": {k: [float(r[k]) for r in ctrl] for k in ctrl[0]},
                    "plugin": {k: [float(r[k]) for r in plug] for k in plug[0]},
                    "rel_disc_error_reduction_vs_control": float((e_c - e_p) / e_c),
                    "rel_disc_error_reduction_vs_alone": float((e_a - e_p) / e_a)}
        KP = ["f1", "conc", "iso_det", "iso_kept", "disc", "level", "xfbd_det", "xfbd_int"]
        val = lambda rs, k: None if k not in rs[0] else float(np.mean([r[k] for r in rs]))  # noqa: E731
        ca, cp = bold_best([[val([alone], k), val(plug, k)] for k in KP])
        rows.append(f"{name} & alone & " + " & ".join(ca) + " \\\\")
        rows.append(" & + plug-in & " + " & ".join(cp) + " \\\\")
    nums["plugin"] = npl
    open(os.path.join(a.out, "tab_plugin.tex"), "w").write("\n".join(rows) + "\n")

    # Table: comparison with the xFBD augmentation (pre+post ResNet-50 crop): both directions on real discordant
    # buildings, xFBD detection and the open-ground counterfactual flip (non-flood). The existing model and the plug-in
    # are means over the three seeds, as in the ablation table; the retrained model has one seed.
    cfv = pd.read_csv(a.cf)
    cfv = cfv[cfv.events == "non-flood"].set_index("run")
    flip = lambda r: cfv.loc[r, "conc. intact -> damaged ctx"] if r in cfv.index and pd.notna(cfv.loc[r, "conc. intact -> damaged ctx"]) else np.nan  # noqa: E731
    xl = lambda r: xfbd_logp(r)[1]  # noqa: E731
    mit = [("Existing model", [(L["context"][s], XL["context"][s]) for s in range(3)], flip("crop_rn50_pre_post")),
           ("Retrained with xFBD augmentation", [(logp("crop_rn50_pre_post_xfbd", te), xl("crop_rn50_pre_post_xfbd"))], flip("crop_rn50_pre_post_xfbd")),
           ("With the plug-in (ours)", [(L["context"][s] + L["tight"][s] + L["v1"][s], XL["context"][s] + XL["tight"][s] + XL["v1"][s]) for s in range(3)], np.nan)]
    rows, nm = [], {}
    for name, runs, fl in mit:
        rs = [{**scores(lp.argmax(1), y, cat), **xfbd_scores(xlp.argmax(1), xb, flags)} for lp, xlp in runs]
        r = {k: float(np.mean([q[k] for q in rs])) for k in rs[0]}
        r["cf_flip"] = fl
        nm[name] = {k: (None if pd.isna(v) else float(v)) for k, v in r.items()}
    cells = bold_best([[nm[n][k] for n, *_ in mit] for k in ["f1", "iso_det", "iso_kept", "disc"]])  # xFBD and CF in nums
    for (name, *_), c in zip(mit, cells):
        rows.append(f"{name} & " + " & ".join(c) + " \\\\")
    nums["mitigation"] = nm
    open(os.path.join(a.out, "tab_mitigation.tex"), "w").write("\n".join(rows) + "\n")
    # Table: BRIGHT (pre-event optical, post-event SAR, three levels): the authors' five benchmark models
    # alone and with the plug-in (experts trained on BRIGHT, three seeds); flood events keep the existing model
    # alone, as on xBD
    _, tb = test_buildings("data_bright", "test")
    yb, cb = tb.label.to_numpy(), categories(tb, 64)
    floodb = (tb.disaster_type == "flooding").to_numpy()[:, None]
    def lpb(run):
        z = np.load(os.path.join("results/harness_bright", run, "harness_preds.npz"))
        assert (z["building_id"] == tb.building_id.to_numpy()).all(), run
        return np.log(z["probs"].astype(np.float64) + 1e-6)
    def sb(pred):
        d, i, ld, c = cb == "isolated damage", cb == "isolated intact", cb == "level discordant", cb == "concordant"
        r = {"f1": f1_score(yb, pred, average="macro", labels=[0, 1, 2]), "conc": (pred[c] == yb[c]).mean(),
             "iso_det": (pred[d] >= 1).mean(), "iso_kept": (pred[i] == 0).mean(), "level": (pred[ld] == yb[ld]).mean()}
        r["disc"] = (r["iso_det"] + r["iso_kept"]) / 2
        return r
    TB = [lpb("crop_brt2_rn50_pre_post_tight" + k) for k in ("", "_s1", "_s2")]
    SB = [lpb("crop_brt2_rn50_pre_post_pswap_dw5" + k) for k in ("", "_s1", "_s2")]
    rows, nbr = [], {}
    for run, name in [("bright_unet", "UNet"), ("bright_deeplabv3plus", "DeepLabV3+"), ("bright_siamattnunet", "SiamAttnUNet"),
                      ("bright_siamcrnn", "SiamCRNN"), ("bright_damageformer", "DamageFormer")]:
        e = lpb(run)
        alone, plug = sb(e.argmax(1)), [sb(np.where(floodb, e, e + TB[k] + SB[k]).argmax(1)) for k in range(3)]
        nbr[run] = {"alone": {k: float(v) for k, v in alone.items()}, "plugin": {k: [float(r[k]) for r in plug] for k in plug[0]}}
        ca, cp = bold_best([[float(alone[k]), float(np.mean([r[k] for r in plug]))] for k in ["f1", "conc", "iso_det", "iso_kept", "disc", "level"]])
        rows.append(f"{name} & alone & " + " & ".join(ca) + " \\\\")
        rows.append(" & + plug-in & " + " & ".join(cp) + " \\\\")
    nums["bright"] = nbr
    open(os.path.join(a.out, "tab_bright.tex"), "w").write("\n".join(rows) + "\n")
    json.dump(nums, open(os.path.join(a.out, "numbers.json"), "w"), indent=1)
    for run, v in npl.items():
        print(f"{run:22} F1 {v['alone']['f1']:.3f} -> {np.mean(v['plugin']['f1']):.3f} (ctrl {np.mean(v['control']['f1']):.3f})"
              f"  disc {v['alone']['disc']:.3f} -> {np.mean(v['plugin']['disc']):.3f} (ctrl {np.mean(v['control']['disc']):.3f})"
              f"  rel. err. red. vs alone {v['rel_disc_error_reduction_vs_alone']:.3f}")


if __name__ == "__main__":
    main()
