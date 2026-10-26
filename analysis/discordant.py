#!/usr/bin/env python
"""Anatomy of the neighbourhood-discordant buildings of the xBD test split.

Neighbours are the other labelled buildings within R px of a building's centroid.
Categories (L0, as released):
  isolated damage   damaged (label >= 1) with < 25 % damaged neighbours
  isolated intact   intact with >= 75 % damaged neighbours
  level discordant  damaged among damaged neighbours, but < 25 % share its level
  concordant        >= 75 % of the neighbours share its level
  mixed / isolated  the rest / no neighbour within R
Prints, per category and model, accuracy and damage recall (any damage level
predicted for a damaged building), the predictions given to isolated damage,
the radius sensitivity, and where isolated damage occurs; draws a gallery of
isolated damage that every model missed or caught.
Usage: python analysis/discordant.py [--radius 64]
"""
import argparse
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from shapely import wkt as shapely_wkt  # noqa: E402

MODELS = {"ResNet-50 post": ("rn50_post_none", "{}"), "ResNet-50 pre+post": ("rn50_pre_post_none", "{}"),
          "ChangeOS": ("changeos_r101", "{}_inventory"), "DINOv3-SAT MLP": ("dinov3sat_mlp_post_none", "{}")}
COLORS = {0: (60, 200, 60), 1: (255, 230, 0), 2: (255, 140, 0), 3: (230, 30, 30)}  # intact .. destroyed


def categories(te, radius):
    cat = np.full(len(te), "isolated", object)
    xy, lab, sid = te[["cx", "cy"]].to_numpy(), te.label.to_numpy(), te.scene_id.to_numpy()
    for _, idx in pd.Series(np.arange(len(te))).groupby(sid).indices.items():
        d = np.hypot(xy[idx, None, 0] - xy[None, idx, 0], xy[idx, None, 1] - xy[None, idx, 1])
        near = (d < radius) & ~np.eye(len(idx), dtype=bool)
        n = near.sum(1)
        same = (near & (lab[idx, None] == lab[None, idx])).sum(1) / np.maximum(n, 1)
        dmg = (near & (lab[None, idx] >= 1)).sum(1) / np.maximum(n, 1)
        c = np.where(same >= 0.75, "concordant", "mixed")
        c = np.where((lab[idx] >= 1) & (dmg < 0.25), "isolated damage", c)
        c = np.where((lab[idx] == 0) & (dmg >= 0.75), "isolated intact", c)
        c = np.where((lab[idx] >= 1) & (dmg >= 0.25) & (same < 0.25), "level discordant", c)
        cat[idx] = np.where(n > 0, c, "isolated")
    return cat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radius", type=float, default=64)
    ap.add_argument("--gallery", default="figures/discordant_gallery.png")
    ap.add_argument("--gallery-levels", default="1,2,3", help="true levels drawn in the gallery, e.g. 2,3")
    args = ap.parse_args()
    b = pd.read_csv("data/buildings.csv", usecols=["building_id", "scene_id", "label", "wkt_pre", "cx", "cy", "area",
                                                   "minx", "miny", "maxx", "maxy"])
    s = pd.read_csv("data/scenes.csv", usecols=["scene_id", "official", "disaster_type"])
    te = b.merge(s, on="scene_id")
    te = te[(te.official == "test") & (te.label >= 0)].reset_index(drop=True)
    y = te.label.to_numpy()
    cat = categories(te, args.radius)
    preds = {}
    for name, (rdir, key) in MODELS.items():
        z = np.load(f"results/{rdir}/test_conditions_v2_preds.npz")
        assert (z["building_id"] == te.building_id.to_numpy()).all()
        preds[name] = z[key.format("L0_none")].argmax(1)
    order = ["concordant", "mixed", "isolated", "isolated damage", "isolated intact", "level discordant"]
    rows = []
    for c in order:
        m = cat == c
        r = {"category": c, "n": int(m.sum()), "share": round(m.mean(), 3)}
        for name, p in preds.items():
            r[f"{name} acc"] = round(float((p[m] == y[m]).mean()), 3)
            d = m & (y >= 1)
            r[f"{name} dmg recall"] = round(float((p[d] >= 1).mean()), 3) if d.any() else np.nan
        rows.append(r)
    pd.set_option("display.width", 250)
    print(f"== categories at R = {args.radius:.0f} px (xBD test, {len(te)} buildings)")
    print(pd.DataFrame(rows).set_index("category").T.to_string())
    m = cat == "isolated damage"
    print("\n== isolated damage: true levels", np.bincount(y[m], minlength=4)[1:].tolist(), "(minor, major, destroyed)")
    for name, p in preds.items():
        print(f"   {name:20s} predicted as no/minor/major/destroyed: {np.bincount(p[m], minlength=4).tolist()}")
    print("\n== where isolated damage occurs:", pd.Series(te.disaster_type[m]).value_counts().to_dict())
    print("   all damaged buildings by type:", pd.Series(te.disaster_type[y >= 1]).value_counts().to_dict())
    print("\n== radius sensitivity (ResNet-50 post accuracy)")
    for r in (32, 64, 128):
        c = categories(te, r)
        p = preds["ResNet-50 post"]
        print(f"   R = {r:3d} px: " + ", ".join(f"{k} {np.mean(c == k):.3f} (acc {np.mean(p[c == k] == y[c == k]):.2f})"
                                          for k in ["concordant", "isolated damage", "isolated intact", "level discordant"]))
    # Gallery: isolated damage that every model missed, then some that every model caught.
    levels = np.isin(y, [int(v) for v in args.gallery_levels.split(",")])
    missed = np.flatnonzero(m & levels & np.all([p == 0 for p in preds.values()], 0))
    caught = np.flatnonzero(m & levels & np.all([p >= 1 for p in preds.values()], 0))
    print(f"\n== isolated damage missed by every model: {len(missed)} / {m.sum()}, caught by every model: {len(caught)}")
    rng = np.random.RandomState(0)
    pick = list(rng.choice(missed, min(8, len(missed)), replace=False)) + list(rng.choice(caught, min(4, len(caught)), replace=False))
    scenes = pd.read_csv("data/scenes.csv")
    mm = {k: np.memmap(f"data/{k}.u8", np.uint8, "r", shape=(len(scenes), 1024, 1024, 3)) for k in ("pre", "post")}
    fig, axes = plt.subplots(len(pick) // 2, 4, figsize=(12, 3.1 * len(pick) // 2))
    for k, i in enumerate(pick):
        sid, cx, cy = int(te.scene_id[i]), te.cx[i], te.cy[i]
        half = int(max(48, 0.5 * max(te.maxx[i] - te.minx[i], te.maxy[i] - te.miny[i]) + 40))
        x0, y0 = int(np.clip(cx - half, 0, 1024 - 2 * half)), int(np.clip(cy - half, 0, 1024 - 2 * half))
        near = te[(te.scene_id == sid) & (np.hypot(te.cx - cx, te.cy - cy) < 2 * half)]
        for j, kind in enumerate(("pre", "post")):
            img = np.ascontiguousarray(mm[kind][sid][y0:y0 + 2 * half, x0:x0 + 2 * half])
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
            sc = 256 / (2 * half)
            for _, nb in near.iterrows():
                g = shapely_wkt.loads(nb.wkt_pre)
                for poly in getattr(g, "geoms", [g]):
                    pts = np.round((np.asarray(poly.exterior.coords) - [x0, y0]) * sc).astype(np.int32)
                    cv2.polylines(img, [pts], True, COLORS[int(nb.label)], 3 if nb.building_id == te.building_id[i] else 1,
                                  cv2.LINE_AA)
            ax = axes[k // 2, (k % 2) * 2 + j]
            ax.imshow(img)
            ax.set_xticks([]), ax.set_yticks([])
            if j == 0:
                tag = "missed by all" if i in missed else "caught by all"
                ax.set_title(f"{te.disaster_type[i]}, label {['no', 'minor', 'major', 'destroyed'][y[i]]} ({tag})",
                             fontsize=8.5, loc="left")
    fig.suptitle("Isolated damage in xBD test: pre (left) and post (right); thick outline = the building, "
                 "colour = label (green intact, yellow minor, orange major, red destroyed)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.gallery, dpi=110)
    print("gallery:", args.gallery)


if __name__ == "__main__":
    main()
