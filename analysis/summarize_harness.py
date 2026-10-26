#!/usr/bin/env python
"""One table for every model scored with harness.py (or written in its format).

Reads results/harness/<run>/harness_standard.csv and, when present, harness_counterfactual.csv.
Standard columns: overall macro-F1, QWK and per-class F1; accuracy on concordant, isolated
damage, isolated intact and level-discordant buildings; isolated damage predicted intact.
Counterfactual columns: share predicted damaged for concordant damage moved into an intact
neighbourhood, concordant intact moved into a damaged one, and isolated damage moved into
a damaged one, plus damaged buildings inpainted away ("gone"); PCF = the same with
Poisson-blended pastes (harness.py --blend poisson).
xFBD columns (results/harness_xfbd/<run>, Melamed et al.'s synthetic single-damaged-building
test): share of pasted (major / destroyed) buildings predicted damaged, their exact-level
accuracy, and the share of other buildings within 64 px of the pasted one predicted damaged.
Usage: python analysis/summarize_harness.py [--runs results/harness/*] [--out results/harness/summary_harness]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd

NAMES = {
    "changeos_r101": "ChangeOS R101 (official)",
    "changeos_swint": "ChangeOS Swin-T (TorChange)",
    "changestar_changen2": "ChangeStar 1x256, Changen2 pre-trained",
    "xview2_first": "xView2 1st place (12 nets, TTA)",
    "xview2_first_light": "xView2 1st place (4 nets, no TTA)",
    "xview2_first_res34": "xView2 1st place, ResNet34 folds",
    "xview2_first_res50": "xView2 1st place, SE-ResNeXt50 folds",
    "xview2_first_dpn92": "xView2 1st place, DPN92 folds",
    "xview2_first_se154": "xView2 1st place, SENet154 folds",
    "mambabda_small": "MambaBDA-Small (retrained, sel. hold)",
    "strong_baseline": "Strong Baseline (retrained)",
    "disasteradaptivenet": "DisasterAdaptiveNet (retrained)",
    "damage_triageformer": "Damage-TriageFormer crop (retrained)",
    "krgat": "KR-GAT (re-implemented)",
    "krgat_patch_only": "KR-GAT patch-only ablation",
    "krgat_vanilla_gat": "KR-GAT vanilla-GAT ablation",
    "rn50_post": "ResNet-50 crop, post (ours)",
    "rn50_pre_post": "ResNet-50 crop, pre+post (ours)",
    "dinov3sat_frozen_mlp_post": "DINOv3-SAT ViT-L frozen + MLP, post (ours)",
    "dinov3ft_post": "DINOv3-SAT ViT-L fine-tuned crop, post (ours)",
    "dinov3ft_pre_post": "DINOv3-SAT ViT-L fine-tuned crop, pre+post (ours)",
    "rn50_post_cswap": "ResNet-50 crop, post + donor context swap",
    "rn50_pre_post_cswap": "ResNet-50 crop, pre+post + donor context swap",
    "rn50_post_preswap": "ResNet-50 crop, post + pre-image context swap",
    "rn50_pre_post_preswap": "ResNet-50 crop, pre+post + pre-image context swap",
    "dinov3ft_post_cswap": "DINOv3-SAT fine-tuned crop, post + donor context swap",
    "dinov3ft_pre_post_cswap": "DINOv3-SAT fine-tuned crop, pre+post + donor context swap",
    "rn50_post_tight": "ResNet-50 tight crop (no surroundings), post",
    "crop_rn50_post": "ResNet-50 crop, post",
    "rn50_post_balanced": "ResNet-50 post, class-balanced loss",
    "rn50_pre_post_balanced": "ResNet-50 pre+post, class-balanced loss",
    "crop_rn50_pre_post": "ResNet-50 crop, pre+post",
    "crop_rn50_post_cswap": "ResNet-50 post + hard donor swap (v0)",
    "crop_rn50_pre_post_cswap": "ResNet-50 pre+post + hard donor swap (v0)",
    "crop_rn50_post_preswap": "ResNet-50 post + pre-image swap (v0)",
    "crop_rn50_pre_post_preswap": "ResNet-50 pre+post + pre-image swap (v0)",
    "crop_rn50_post_pswap": "ResNet-50 post + Poisson event swap (v1)",
    "crop_rn50_post_dw5": "ResNet-50 post + discordance x5 (v1)",
    "crop_rn50_post_pswap_dw5": "ResNet-50 post + Poisson swap + discordance x5 (v1)",
    "crop_rn50_pre_post_pswap_dw5": "ResNet-50 pre+post + Poisson swap + discordance x5 (v1)",
    "rn50_pre_post_tight": "ResNet-50 tight crop (no surroundings), pre+post",
}


def xfbd_columns(run, xfbd_root, flags):
    path = os.path.join(xfbd_root, os.path.basename(os.path.normpath(run)), "harness_preds.npz")
    if flags is None or not os.path.exists(path):
        return {}
    z = np.load(path)
    f = flags.loc[z["building_id"]]
    pred, y = z["probs"].argmax(1), z["label"]
    pasted, near = f.pasted.to_numpy(bool), ~f.pasted.to_numpy(bool) & (f.dist_pasted.to_numpy() <= 64)
    return {"xFBD pasted -> damaged": (pred[pasted] >= 1).mean(), "xFBD pasted exact": (pred[pasted] == y[pasted]).mean(),
            "xFBD near others -> damaged": (pred[near] >= 1).mean() if near.any() else np.nan}


def row(run, xfbd_root=None, flags=None):
    std = pd.read_csv(os.path.join(run, "harness_standard.csv")).set_index("stratum")
    name = os.path.basename(os.path.normpath(run))
    a = std.loc["all"]
    r = {"model": NAMES.get(name, name), "macro-F1": a.macro_f1, "QWK": a.qwk,
         **{f"F1 {c}": a[f"f1_{k}"] for k, c in enumerate(["no", "minor", "major", "destroyed"])}}
    for s, col in [("concordant", "acc concordant"), ("isolated damage", "acc isolated damage"),
                   ("isolated intact", "acc isolated intact"), ("level discordant", "acc level discordant")]:
        r[col] = std.loc[s, "acc"] if s in std.index else float("nan")
    if "isolated damage" in std.index and "isolated intact" in std.index:
        # both directions of a context conflict: a model that calls everything damaged scores 0.5
        r["discordant balanced acc"] = (std.loc["isolated damage", "acc"] + std.loc["isolated intact", "acc"]) / 2
    if "isolated damage" in std.index:
        r["iso. major -> intact"] = std.loc["isolated damage", "missed_major"]
        r["iso. destroyed -> intact"] = std.loc["isolated damage", "missed_destroyed"]
    cf_path = os.path.join(run, "harness_counterfactual.csv")
    if os.path.exists(cf_path):
        cf = pd.read_csv(cf_path).set_index("targets")
        r["CF conc. damage, intact ctx"] = cf.loc["concordant damage", "intact context"]
        r["CF conc. intact, damaged ctx"] = cf.loc["concordant intact", "damaged context"]
        r["CF iso. damage, damaged ctx"] = cf.loc["isolated damage", "damaged context"]
        r["CF conc. damage, gone"] = cf.loc["concordant damage", "building gone"]
        r["CF n per group"] = int(cf.n.min())
    pcf_path = os.path.join(run, "harness_counterfactual_poisson.csv")
    if os.path.exists(pcf_path):  # the same test with Poisson-blended pastes (no hard seam)
        cf = pd.read_csv(pcf_path).set_index("targets")
        r["PCF conc. damage, intact ctx"] = cf.loc["concordant damage", "intact context"]
        r["PCF conc. intact, damaged ctx"] = cf.loc["concordant intact", "damaged context"]
        r["PCF iso. damage, damaged ctx"] = cf.loc["isolated damage", "damaged context"]
    r.update(xfbd_columns(run, xfbd_root, flags))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=sorted(glob.glob("results/harness/*/")))
    ap.add_argument("--out", default="results/harness/summary_harness")
    ap.add_argument("--xfbd-root", default="results/harness_xfbd")
    ap.add_argument("--xfbd-data", default="data_xfbd_test")
    args = ap.parse_args()
    runs = [r for r in args.runs if os.path.exists(os.path.join(r, "harness_standard.csv"))]
    flags = None
    if os.path.isdir(args.xfbd_root) and os.path.exists(os.path.join(args.xfbd_data, "buildings.csv")):
        flags = pd.read_csv(os.path.join(args.xfbd_data, "buildings.csv"),
                            usecols=["building_id", "pasted", "dist_pasted"]).set_index("building_id")
    t = pd.DataFrame([row(r, args.xfbd_root, flags) for r in runs])
    t.to_csv(args.out + ".csv", index=False)
    with open(args.out + ".md", "w") as fh:
        fh.write("Share of buildings in each column, xBD official test (53,850 labelled buildings).\n"
                 "acc = 4-level accuracy; 'iso. major / destroyed -> intact' = isolated damaged buildings "
                 "predicted no-damage; CF = share predicted damaged after moving the building (4 px rim) "
                 "into another neighbourhood or inpainting it away.\n\n")
        fh.write(t.to_markdown(index=False, floatfmt=".3f"))
        fh.write("\n")
    print(t.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
