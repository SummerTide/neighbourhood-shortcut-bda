#!/usr/bin/env python
"""Index the complete xBD release for footprint-conditioned damage assessment.

Writes to <out>:
  scenes.csv     one row per pre/post image pair: scene_id, split (train/tier3/hold/test),
                 official split (train = train+tier3, val = hold, test = test), disaster,
                 stem and acquisition metadata of both images
  buildings.csv  one row per building: building_id, scene_id, uid, damage label
                 (0-3, -1 = un-classified), the pre-event polygon (the inventory
                 geometry), the post-event polygon, bounding box, area, centroid
  pre.u8, post.u8  all images decoded once, uint8 memmaps of shape (n_scenes, 1024, 1024, 3)
"""
import argparse
import csv
import json
import os
from multiprocessing import Pool

import cv2
import numpy as np
from shapely import wkt as shapely_wkt

LABELS = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3, "un-classified": -1}
OFFICIAL = {"train": "train", "tier3": "train", "hold": "val", "test": "test"}
SIZE = 1024
META_KEYS = ["off_nadir_angle", "target_azimuth", "sun_azimuth", "sun_elevation", "gsd", "capture_date", "sensor"]


def list_scenes(root):
    scenes = []
    for split in ("train", "tier3", "hold", "test"):
        for fn in sorted(os.listdir(os.path.join(root, split, "labels"))):
            if fn.endswith("_post_disaster.json"):
                scenes.append((split, fn[: -len("_post_disaster.json")]))
    return scenes


def process(args):
    scene_id, root, split, stem, out, n_scenes = args
    base = os.path.join(root, split)
    post = json.load(open(f"{base}/labels/{stem}_post_disaster.json"))
    pre = json.load(open(f"{base}/labels/{stem}_pre_disaster.json"))
    for name, mm_name in (("pre", "pre.u8"), ("post", "post.u8")):
        img = cv2.imread(f"{base}/images/{stem}_{name}_disaster.png", cv2.IMREAD_COLOR)
        assert img.shape == (SIZE, SIZE, 3), (stem, img.shape)
        mm = np.memmap(os.path.join(out, mm_name), np.uint8, "r+", shape=(n_scenes, SIZE, SIZE, 3))
        mm[scene_id] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mm.flush()
        del mm
    pre_wkt = {f["properties"]["uid"]: f["wkt"] for f in pre["features"]["xy"]}
    buildings = []
    for f in post["features"]["xy"]:
        uid = f["properties"]["uid"]
        w_pre = pre_wkt.get(uid, f["wkt"])
        g = shapely_wkt.loads(w_pre)
        if g.is_empty:
            continue
        minx, miny, maxx, maxy = g.bounds
        c = g.centroid
        buildings.append({
            "scene_id": scene_id, "uid": uid,
            "label": LABELS.get(f["properties"].get("subtype", "un-classified"), -1),
            "wkt_pre": w_pre, "wkt_post": f["wkt"],
            "minx": round(minx, 2), "miny": round(miny, 2), "maxx": round(maxx, 2), "maxy": round(maxy, 2),
            "area": round(g.area, 2), "cx": round(c.x, 2), "cy": round(c.y, 2),
        })
    meta = {"scene_id": scene_id, "split": split, "official": OFFICIAL[split],
            "disaster": post["metadata"]["disaster"], "disaster_type": post["metadata"].get("disaster_type"), "stem": stem}
    for k in META_KEYS:
        meta[f"pre_{k}"] = pre["metadata"].get(k)
        meta[f"post_{k}"] = post["metadata"].get(k)
    return meta, buildings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="complete xBD release with train/tier3/hold/test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scenes = list_scenes(args.root)
    n = len(scenes)
    for name in ("pre.u8", "post.u8"):
        np.memmap(os.path.join(args.out, name), np.uint8, "w+", shape=(n, SIZE, SIZE, 3)).flush()
    jobs = [(i, args.root, sp, st, args.out, n) for i, (sp, st) in enumerate(scenes)]
    metas, rows = [None] * n, []
    with Pool(args.workers) as pool:
        for k, (meta, buildings) in enumerate(pool.imap_unordered(process, jobs, chunksize=8)):
            metas[meta["scene_id"]] = meta
            rows.extend(buildings)
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{n} scenes", flush=True)
    with open(os.path.join(args.out, "scenes.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(metas[0]))
        wr.writeheader()
        wr.writerows(metas)
    rows.sort(key=lambda r: (r["scene_id"], r["uid"]))
    with open(os.path.join(args.out, "buildings.csv"), "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=["building_id"] + list(rows[0]))
        wr.writeheader()
        for i, r in enumerate(rows):
            wr.writerow({"building_id": i, **r})
    print(f"done: {n} scenes, {len(rows)} buildings -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
