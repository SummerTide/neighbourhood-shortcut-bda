#!/usr/bin/env python
"""xFBD synthetic single-building sets (Melamed et al., arXiv 2212.13876) in our table + memmap layout.

Each xFBD sample is one Palu tile whose post image is the original pre image with ONE damaged
building Poisson-blended in from the real post image, paired with a secondary pre-event image
of another date; the pasted building keeps its level and every other building is relabelled
no-damage. Output (load_tables-compatible): scenes.csv (one row per sample, official =
xfbd_<split>), buildings.csv (+ pasted flag and xbd_building_id, the same building in
data/buildings.csv), pre.u8 / post.u8.
Usage: python prep_xfbd.py --xfbd ../baselines/data/xfbd --split test --out data_xfbd_test
"""
import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
import rasterio
from shapely import wkt as shapely_wkt

LEVEL = {"no-damage": 0, "minor-damage": 1, "major-damage": 2, "destroyed": 3}
IMG = 1024


def read(path):
    with rasterio.open(path) as src:
        return src.read([1, 2, 3]).astype(np.uint8).transpose(1, 2, 0)


def add_dist_pasted(b):
    """Centroid distance (px) of every building to its sample's pasted building."""
    p = b[b.pasted].set_index("scene_id")[["cx", "cy"]]
    px, py = p.cx.reindex(b.scene_id).to_numpy(), p.cy.reindex(b.scene_id).to_numpy()
    return b.assign(dist_pasted=np.hypot(b.cx.to_numpy() - px, b.cy.to_numpy() - py))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xfbd", default="../baselines/data/xfbd")
    ap.add_argument("--split", default="test", choices=["tier1", "test", "hold"])
    ap.add_argument("--xbd-data", default="data", help="our xBD tables, to link pasted buildings by uid")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    labels = sorted(glob.glob(os.path.join(args.xfbd, args.split, "labels", "*_post_disaster.json")))
    xbd_s = pd.read_csv(os.path.join(args.xbd_data, "scenes.csv"))
    xbd_b = pd.read_csv(os.path.join(args.xbd_data, "buildings.csv")).merge(xbd_s[["scene_id", "stem"]], on="scene_id")
    xbd_id = {(s, u): i for s, u, i in zip(xbd_b.stem, xbd_b.uid, xbd_b.building_id)}
    shape = (len(labels), IMG, IMG, 3)
    mm = {k: np.memmap(os.path.join(args.out, f"{k}.u8"), np.uint8, "w+", shape=shape) for k in ("pre", "post")}
    scenes, rows = [], []
    for sid, lf in enumerate(labels):
        sample = os.path.basename(lf)[: -len("_post_disaster.json")]
        disaster, tile, pasted_uid = sample.split("_")
        stem = f"{disaster}_{tile}"
        meta = json.load(open(lf))
        for k in ("pre", "post"):
            mm[k][sid] = read(os.path.join(args.xfbd, args.split, "images", f"{sample}_{k}_disaster.tif"))
        scenes.append({"scene_id": sid, "split": f"xfbd_{args.split}", "official": f"xfbd_{args.split}",
                       "disaster": disaster, "disaster_type": meta["metadata"].get("disaster_type", "tsunami"),
                       "stem": stem, "sample": sample, "pasted_uid": pasted_uid})
        for ft in meta["features"]["xy"]:
            g = shapely_wkt.loads(ft["wkt"])
            if g.is_empty:
                continue
            uid = ft["properties"]["uid"]
            minx, miny, maxx, maxy = g.bounds
            rows.append({"scene_id": sid, "uid": uid, "label": LEVEL.get(ft["properties"].get("subtype"), -1),
                         "wkt_pre": ft["wkt"], "wkt_post": ft["wkt"], "minx": minx, "miny": miny, "maxx": maxx,
                         "maxy": maxy, "area": g.area, "cx": g.centroid.x, "cy": g.centroid.y,
                         "pasted": uid == pasted_uid, "xbd_building_id": xbd_id.get((stem, uid), -1)})
        if (sid + 1) % 200 == 0:
            print(f"{sid + 1}/{len(labels)} samples", flush=True)
    for k in ("pre", "post"):
        mm[k].flush()
    b = pd.DataFrame(rows)
    b.insert(0, "building_id", np.arange(len(b)))
    b = add_dist_pasted(b)
    pd.DataFrame(scenes).to_csv(os.path.join(args.out, "scenes.csv"), index=False)
    b.to_csv(os.path.join(args.out, "buildings.csv"), index=False)
    p = b[b.pasted]
    print(f"{len(scenes)} samples, {len(b)} buildings; pasted: {len(p)} "
          f"({np.bincount(p.label.clip(0), minlength=4).tolist()} no/minor/major/destroyed), "
          f"linked to xBD buildings: {(p.xbd_building_id >= 0).mean():.3f}", flush=True)


if __name__ == "__main__":
    main()
