#!/usr/bin/env python
"""How much does a trained crop model judge a building by its surroundings?

Counterfactual test on xBD test buildings (L0, no retraining). For each target
building the crops (post, and pre for pre+post models) are scored as:
  original        the crop as in training
  intact context  the building and a 4 px margin kept, the rest of the crop taken
                  from a building in an all-intact neighbourhood
  damaged context the same with a building in an all-destroyed / major neighbourhood
  building gone   the building and its margin inpainted from the surroundings
Targets: isolated damage, concordant damage, isolated intact and concordant
intact buildings (categories of analysis/discordant.py). Reports the share
predicted as damaged (any level >= minor) per variant.
Usage: python analysis/context_swap.py --model results/rn50_post_none --inputs post
"""
import argparse
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bda_data import BuildingDataset, ImageSource, load_tables  # noqa: E402
from discordant import categories  # noqa: E402
from train_bda import build_model, to_input  # noqa: E402

MARGIN = 4  # px around the footprint that stays with the building


def keep_mask(mask, side_scale):
    k = 2 * int(round(MARGIN * side_scale)) + 1
    return cv2.dilate(mask, np.ones((k, k), np.uint8)) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="results dir holding best.pt")
    ap.add_argument("--inputs", choices=["post", "pre+post"], default="post")
    ap.add_argument("--n-concordant", type=int, default=1500)
    ap.add_argument("--weights", default="external/checkpoints/resnet50-11ad3fa6.pth")
    ap.add_argument("--backbone", choices=["resnet50", "dinov3_vitl16_sat"], default="resnet50")
    ap.add_argument("--hub-dir", default="external/torch_hub")
    ap.add_argument("--dinov3-repo", default="external/dinov3")
    args = ap.parse_args()
    rng = np.random.RandomState(0)
    scenes, b = load_tables("data")
    te = b[(b.official == "test") & (b.label >= 0)].reset_index(drop=True)
    cat = categories(te, 64)
    y = te.label.to_numpy()
    groups = {"isolated damage": np.flatnonzero(cat == "isolated damage"),
              "concordant damage": rng.permutation(np.flatnonzero((cat == "concordant") & (y >= 1)))[:args.n_concordant],
              "isolated intact": np.flatnonzero(cat == "isolated intact"),
              "concordant intact": rng.permutation(np.flatnonzero((cat == "concordant") & (y == 0)))[:args.n_concordant]}
    donors = {"intact context": np.flatnonzero((cat == "concordant") & (y == 0)),
              "damaged context": np.flatnonzero((cat == "concordant") & (y >= 2))}
    use_pre = args.inputs == "pre+post"
    post = ImageSource("data/post.u8", len(scenes))
    pre = ImageSource("data/pre.u8", len(scenes)) if use_pre else None
    ds = BuildingDataset(te, post, pre)
    model = build_model(args).eval()
    model.load_state_dict(torch.load(os.path.join(args.model, "best.pt"), map_location="cuda"))

    def score(items):
        batch = {k: torch.stack([torch.as_tensor(it[k]) for it in items]) for k in items[0] if k in ("post", "mask", "pre")}
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            p = F.softmax(model(*to_input(batch, False)).float(), 1).cpu().numpy()
        return p

    rows = []
    for gname, idx in groups.items():
        out = {v: [] for v in ["original", "intact context", "damaged context", "building gone"]}
        for s in range(0, len(idx), 128):
            chunk = idx[s:s + 128]
            items = {v: [] for v in out}
            for i in chunk:
                it = ds[int(i)]
                box = te.loc[int(i), ["minx", "miny", "maxx", "maxy"]].to_numpy(float)
                side = max(64, np.ceil(max(box[2] - box[0], box[3] - box[1])) + 64)
                keep = keep_mask(it["mask"].numpy(), 224 / side)
                items["original"].append({k: it[k] for k in ("post", "mask", "pre") if k in it})
                for v in ("intact context", "damaged context"):
                    d = ds[int(rng.choice(donors[v]))]
                    mix = {"mask": it["mask"]}
                    for k in ("post", "pre"):
                        if k in it:
                            a, c = it[k].numpy(), d[k].numpy()
                            mix[k] = torch.from_numpy(np.where(keep[..., None], a, c))
                    items[v].append(mix)
                gone = {"mask": it["mask"]}
                for k in ("post", "pre"):
                    if k in it:
                        gone[k] = torch.from_numpy(cv2.inpaint(np.ascontiguousarray(it[k].numpy()),
                                                               keep.astype(np.uint8), 7, cv2.INPAINT_TELEA))
                items["building gone"].append(gone)
            for v in out:
                out[v].append(score(items[v]))
        r = {"targets": gname, "n": len(idx)}
        for v, ps in out.items():
            r[v] = round(float((np.concatenate(ps).argmax(1) >= 1).mean()), 3)
        rows.append(r)
    t = pd.DataFrame(rows).set_index("targets")
    print(f"share predicted damaged ({os.path.basename(os.path.normpath(args.model))}, {args.inputs})")
    print(t.to_string())
    t.to_csv(os.path.join(args.model, f"context_swap_{args.inputs.replace('+', '_')}.csv"))


if __name__ == "__main__":
    main()
