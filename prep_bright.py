#!/usr/bin/env python
"""Index BRIGHT (pre-event optical + post-event SAR) per building, in the table + memmap format of
prep_dm3.py, so that train_bda.py, harness.py and the analysis scripts run on it unchanged
(with BDA_N_CLASSES=3).

Buildings are the instance polygons of the BRIGHT challenge (target_instance_level/*.json,
categories 1 intact, 2 damaged, 3 destroyed -> labels 0, 1, 2), drawn in the 1024 x 1024 tile grid.
Splits follow BRIGHT's standard_ML split (train / test); scenes without instance labels or without a
pre-event image (events whose optical data is not redistributed) are left out. The instance labels
do not cover the standard_ML val scenes, so val is 10% of the train scenes (seed 0), as in
prep_dm3cat.py.
The post-event SAR amplitude (one band) is stored in all three channels of post.u8.
Usage: python prep_bright.py --root ../data_raw/bright --splits ../baselines/BRIGHT/bda_benchmark/dataset/splitname/standard_ML --out data_bright
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
import rasterio

SIZE = 1024
MIN_AREA = 10
TYPES = {"bata-explosion": "explosion", "beirut-explosion": "explosion", "congo-volcano": "volcano",
         "haiti-earthquake": "earthquake", "hawaii-wildfire": "fire", "la_palma-volcano": "volcano",
         "libya-flood": "flooding", "marshall-wildfire": "fire", "mexico-hurricane": "wind",
         "morocco-earthquake": "earthquake", "myanmar-hurricane": "wind", "noto-earthquake": "earthquake",
         "turkey-earthquake": "earthquake", "ukraine-conflict": "conflict"}


def to_u8(a):
    """Bands-first raster -> HxWx3 uint8 (SAR: one band copied to three; stretched if not uint8)."""
    a = a[:3] if a.shape[0] >= 3 else np.repeat(a[:1], 3, 0)
    a = a.transpose(1, 2, 0)
    if a.dtype != np.uint8:
        a = a.astype(np.float32)
        lo, hi = np.percentile(a, 1), np.percentile(a, 99)
        a = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    return a


def process(args):
    scene_id, stem, root, out, n = args
    imgs = {}
    for k, sub, suffix in (("pre", "pre-event", "pre_disaster"), ("post", "post-event", "post_disaster")):
        with rasterio.open(os.path.join(root, sub, f"{stem}_{suffix}.tif")) as r:
            imgs[k] = to_u8(r.read())
    if any(im.shape != (SIZE, SIZE, 3) for im in imgs.values()):
        return scene_id, None
    for k, im in imgs.items():
        mm = np.memmap(os.path.join(out, f"{k}.u8"), np.uint8, "r+", shape=(n, SIZE, SIZE, 3))
        mm[scene_id] = im
        mm.flush()
        del mm
    ann = json.load(open(os.path.join(root, "target_instance_level", f"{stem}_instance_damage.json"), encoding="utf-8-sig"))
    rows = []
    for a in ann["annotations"]:
        seg = a["segmentation"]
        seg = json.loads(seg) if isinstance(seg, str) else seg
        poly = np.array(max(seg, key=len), float).reshape(-1, 2)
        if len(poly) < 3:
            continue
        m = np.zeros((SIZE, SIZE), np.uint8)
        cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
        area = int(m.sum())
        if area < MIN_AREA:
            continue
        ys, xs = np.nonzero(m)
        ring = np.vstack([poly, poly[:1]])
        rows.append({"scene_id": scene_id, "uid": f"{stem}_{a['id']}", "label": int(a["category_id"]) - 1,
                     "wkt_pre": "POLYGON ((" + ", ".join(f"{x:.1f} {y:.1f}" for x, y in ring) + "))",
                     "minx": float(xs.min()), "miny": float(ys.min()), "maxx": float(xs.max()), "maxy": float(ys.max()),
                     "area": area, "cx": round(float(xs.mean()), 2), "cy": round(float(ys.mean()), 2)})
    return scene_id, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    scenes = []
    for split, name in (("train", "train_set.txt"), ("val", "val_set.txt"), ("test", "test_set.txt")):
        for stem in open(os.path.join(a.splits, name)).read().split():
            ok = (os.path.exists(os.path.join(a.root, "target_instance_level", f"{stem}_instance_damage.json"))
                  and os.path.exists(os.path.join(a.root, "pre-event", f"{stem}_pre_disaster.tif")))
            if ok:
                event = stem.rsplit("_", 1)[0]
                scenes.append({"split": f"bright_{split}", "official": split, "disaster": event,
                               "disaster_type": TYPES[event], "stem": stem})
    tr = [i for i, s in enumerate(scenes) if s["official"] == "train"]
    if not any(s["official"] == "val" for s in scenes):
        for i in np.random.RandomState(0).permutation(tr)[:int(round(0.1 * len(tr)))]:
            scenes[i]["official"], scenes[i]["split"] = "val", "bright_val"
    n = len(scenes)
    for k in ("pre", "post"):
        np.memmap(os.path.join(a.out, f"{k}.u8"), np.uint8, "w+", shape=(n, SIZE, SIZE, 3)).flush()
    rows, bad = [], []
    with Pool(a.workers) as pool:
        for sid, r in pool.imap_unordered(process, [(i, s["stem"], a.root, a.out, n) for i, s in enumerate(scenes)], chunksize=4):
            if r is None:
                bad.append(sid)
            else:
                rows.extend(r)
    assert not bad, f"{len(bad)} scenes with unexpected sizes, e.g. {scenes[bad[0]]['stem']}"
    with open(os.path.join(a.out, "scenes.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["scene_id", "split", "official", "disaster", "disaster_type", "stem"])
        wr.writeheader()
        wr.writerows({"scene_id": i, **s} for i, s in enumerate(scenes))
    rows.sort(key=lambda r: (r["scene_id"], r["uid"]))
    with open(os.path.join(a.out, "buildings.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["building_id"] + list(rows[0]))
        wr.writeheader()
        for i, r in enumerate(rows):
            wr.writerow({"building_id": i, **r})
    counts = {s: sum(1 for x in scenes if x["official"] == s) for s in ("train", "val", "test")}
    print(f"done: scenes {counts}, buildings {len(rows)} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
