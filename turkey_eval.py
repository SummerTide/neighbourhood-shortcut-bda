#!/usr/bin/env python
"""Per-building damage probabilities of xBD-trained models on the Islahiye (2023 Turkiye earthquake)
scene prepared by prep_turkey.py (one 10,625 x 10,625 px scene at 0.5 m, BRIGHT reference footprints).

Tile models (tile_models.py / tile_models_ext): 1024 x 1024 windows with stride 768 (the last one
flush with the scene edge); windows that hold no pixel of a scored footprint are skipped (every
footprint pixel is then covered by every window that contains it); per-pixel probabilities are
averaged over the overlapping windows and then over each footprint (harness.footprint_probs).
Models with needs_meta (DisasterAdaptiveNet) get disaster 'turkey-earthquake' (not in the repo's
event table) and type 'earthquake', i.e. the earthquake hazard id, through harness.run.

Building models (needs_footprints, the crop classifiers): each building is scored in a 1024 tile
that holds its whole crop window (bda_data.BuildingDataset: bounding box + 2 x context_px, at least
MIN_SIDE; the largest context of the crop runs, 32 px, is assumed), preferably one of the same
window grid, else a tile centred on the building; tiles are cut from the scene and the footprints
shifted into them (harness.score). Crops that reach past the scene edge are zero-padded, as at xBD
tile edges.

Writes <out>/<model>/turkey_preds.npz (building_id, label, probs (n, 4) over the xBD levels).
Usage: python turkey_eval.py --model changeos_r101 [more models] [--region x0,y0,w,h] [--out results/turkey]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tile_models  # noqa: E402
from bda_data import CONTEXT_PX, IMG, MIN_SIDE, polygons  # noqa: E402
from harness import footprint_probs, run, score  # noqa: E402

STRIDE = 768
CROP_CONTEXT = max(CONTEXT_PX, 32)


def grid(n):
    pos = list(range(0, n - IMG + 1, STRIDE))
    return pos if pos[-1] == n - IMG else pos + [n - IMG]


def load(data, region):
    b = pd.read_csv(os.path.join(data, "buildings.csv"))
    pre = np.load(os.path.join(data, "pre.npy"), mmap_mode="r")
    post = np.load(os.path.join(data, "post.npy"), mmap_mode="r")
    ids = np.load(os.path.join(data, "ids.npy"), mmap_mode="r")
    keep = np.ones(len(b), bool)
    if region is not None:  # smoke test: buildings whose bounding box lies inside the region
        x0, y0, w, h = region
        keep = (b.minx >= x0) & (b.maxx < x0 + w) & (b.miny >= y0) & (b.maxy < y0 + h)
        keep = keep.to_numpy()
    return b, keep, pre, post, ids


def tile_model_probs(model, b, keep, pre, post, ids, region):
    H, W = ids.shape
    meta = pd.DataFrame({"disaster": ["turkey-earthquake"], "disaster_type": ["earthquake"]})
    sel = np.zeros(len(b) + 1, bool)
    sel[1:] = keep
    wins = []
    for y0 in grid(H):
        for x0 in grid(W):
            if region is not None:
                rx, ry, rw, rh = region
                if x0 + IMG <= rx or x0 >= rx + rw or y0 + IMG <= ry or y0 >= ry + rh:
                    continue
            if sel[np.asarray(ids[y0:y0 + IMG, x0:x0 + IMG])].any():
                wins.append((x0, y0))
    print(f"  {len(wins)} windows with scored buildings", flush=True)
    # accumulate only over the bounding box of the windows used
    bx0, by0 = min(x for x, _ in wins), min(y for _, y in wins)
    bx1, by1 = max(x for x, _ in wins) + IMG, max(y for _, y in wins) + IMG
    acc, cnt = None, np.zeros((by1 - by0, bx1 - bx0), np.float32)
    t0 = time.time()
    for k, (x0, y0) in enumerate(wins):
        p = run(model, np.ascontiguousarray(pre[y0:y0 + IMG, x0:x0 + IMG]),
                np.ascontiguousarray(post[y0:y0 + IMG, x0:x0 + IMG]), meta, 0)
        if acc is None:
            acc = np.zeros((p.shape[0],) + cnt.shape, np.float32)
        acc[:, y0 - by0:y0 - by0 + IMG, x0 - bx0:x0 - bx0 + IMG] += p
        cnt[y0 - by0:y0 - by0 + IMG, x0 - bx0:x0 - bx0 + IMG] += 1
        if (k + 1) % 20 == 0:
            print(f"  {k + 1}/{len(wins)} windows ({time.time() - t0:.0f}s)", flush=True)
    sub = np.asarray(ids[by0:by1, bx0:bx1]).copy()
    sub[~sel[sub]] = 0
    assert (cnt[sub > 0] > 0).all()
    acc /= np.maximum(cnt, 1)[None]
    probs = footprint_probs(acc, sub, len(b))
    return probs[keep]


def crop_tiles(b, keep, H, W):
    """{(x0, y0): [building indices]}: a 1024 tile per building that holds its whole crop window."""
    gx, gy = np.array(grid(W)), np.array(grid(H))
    out = {}
    for i in np.flatnonzero(keep):
        minx, miny, maxx, maxy = b.loc[i, ["minx", "miny", "maxx", "maxy"]]
        side = max(MIN_SIDE, np.ceil(max(maxx - minx, maxy - miny)) + 2 * CROP_CONTEXT)
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        # the crop window, clipped to the scene (beyond it the crop is zero-padded either way)
        c0, r0 = max(0, cx - side / 2 - 1), max(0, cy - side / 2 - 1)
        c1, r1 = min(W, cx + side / 2 + 1), min(H, cy + side / 2 + 1)
        okx, oky = gx[(gx <= c0) & (gx + IMG >= c1)], gy[(gy <= r0) & (gy + IMG >= r1)]
        if len(okx) and len(oky):
            x0 = int(okx[np.argmin(np.abs(okx + IMG / 2 - cx))])
            y0 = int(oky[np.argmin(np.abs(oky + IMG / 2 - cy))])
        else:
            x0 = int(np.clip(round(cx - IMG / 2), 0, W - IMG))
            y0 = int(np.clip(round(cy - IMG / 2), 0, H - IMG))
        out.setdefault((x0, y0), []).append(i)
    return out


def building_model_probs(model, b, keep, pre, post, ids):
    H, W = ids.shape
    polys = polygons(b.wkt_pre)
    tiles = crop_tiles(b, keep, H, W)
    print(f"  {len(tiles)} tiles for {keep.sum()} buildings", flush=True)
    probs = {}
    for (x0, y0), idx in tiles.items():
        p = score(model, np.ascontiguousarray(pre[y0:y0 + IMG, x0:x0 + IMG]),
                  np.ascontiguousarray(post[y0:y0 + IMG, x0:x0 + IMG]), [polys[i] for i in idx], None, None,
                  shift=(-x0, -y0))
        probs.update(zip(idx, p))
    return np.stack([probs[i] for i in np.flatnonzero(keep)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True, choices=tile_models.available())
    ap.add_argument("--data", default=os.path.join(HERE, "data_turkey"))
    ap.add_argument("--out", default=os.path.join(HERE, "results", "turkey"))
    ap.add_argument("--region", default=None, help="x0,y0,w,h (resampled px): smoke test on the buildings inside")
    ap.add_argument("--skip-done", action="store_true", help="skip models whose turkey_preds.npz exists")
    args = ap.parse_args()
    region = None if args.region is None else tuple(int(v) for v in args.region.split(","))
    b, keep, pre, post, ids = load(args.data, region)
    print(f"{keep.sum()} scored buildings (labels {np.bincount(b.label[keep], minlength=3).tolist()})", flush=True)
    for name in args.model:
        f = os.path.join(args.out, name, "turkey_preds.npz")
        if args.skip_done and os.path.exists(f):
            print(f"{name}: {f} exists, skipped", flush=True)
            continue
        t0 = time.time()
        try:
            model = tile_models.build(name)
            if getattr(model, "needs_footprints", False):
                probs = building_model_probs(model, b, keep, pre, post, ids)
            else:
                probs = tile_model_probs(model, b, keep, pre, post, ids, region)
        except Exception as e:  # keep going with the other models of the job; the log says what failed
            import traceback
            traceback.print_exc()
            print(f"{name}: FAILED ({type(e).__name__}: {e})", flush=True)
            continue
        os.makedirs(os.path.dirname(f), exist_ok=True)
        np.savez_compressed(f, building_id=b.building_id.to_numpy()[keep], label=b.label.to_numpy()[keep],
                            probs=probs.astype(np.float32))
        y = b.label.to_numpy()[keep]
        pred3 = np.array([0, 1, 1, 2])[probs.argmax(1)]
        print(f"{name}: {len(probs)} buildings, acc (argmax4 -> 3) {np.mean(pred3 == y):.3f}, "
              f"mean probs {probs.mean(0).round(3).tolist()} ({time.time() - t0:.0f}s) -> {f}", flush=True)
        del model
        import torch
        torch.cuda.empty_cache()
    print("done", flush=True)


if __name__ == "__main__":
    main()
