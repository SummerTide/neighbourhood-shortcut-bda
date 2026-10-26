#!/usr/bin/env python
"""Islahiye (2023 Turkiye earthquake) BRIGHT instance labels on the Maxar pre/post pair 'full',
resampled to 0.5 m, for the footprint-conditioned evaluation (turkey_eval.py).

Run ONCE in env bda_tc (rasterio 1.4.4): the JPEG-in-TIFF RGB decodes slightly differently
across rasterio/GDAL versions, so every model reads the decoded, resampled arrays cached here.

Writes <out>/
  pre.npy, post.npy   uint8 (H, W, 3) RGB, cv2.INTER_AREA resampling by 0.30517578125 / 0.5
  ids.npy             int32 (H, W): row index + 1 of buildings.csv per pixel (0 = none),
                      rasterised with cv2.fillPoly as harness.raster does
  buildings.csv       the recommended_for_instance_scoring instances: building_id (= raster_id),
                      scene_id 0, label 0/1/2 (intact / damaged / destroyed), wkt_pre (resampled
                      pixels, x = col, y = row), minx..maxy, area (px), cx, cy (polygon centroid)
  buildings_all.csv   every labelled instance record (also non-recommended), same columns plus
                      'recommended', for the neighbourhood sensitivity analysis only
  meta.json           sizes, scale, counts, environment
Usage: python prep_turkey.py [--out data_turkey]
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd
import rasterio
from shapely.affinity import scale as shp_scale
from shapely import wkt as shapely_wkt
from shapely.geometry import shape

ROOT = "datasets/islahiye"
SRC_PX, DST_PX = 0.30517578125, 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_turkey")
    ap.add_argument("--redo-images", action="store_true", help="decode and resample again even if the .npy exist")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = SRC_PX / DST_PX
    pairs = {p["id"]: p for p in json.load(open(os.path.join(ROOT, "pairs.json")))["pairs"]}
    pair = pairs["full"]
    H = W = None
    for k in ("pre", "post"):
        if os.path.exists(os.path.join(a.out, f"{k}.npy")) and not a.redo_images:
            H, W = np.load(os.path.join(a.out, f"{k}.npy"), mmap_mode="r").shape[:2]
            print(f"{k}: kept {a.out}/{k}.npy ({H} x {W})", flush=True)
            continue
        with rasterio.open(os.path.join(ROOT, pair[k])) as src:
            assert src.count == 3 and src.dtypes[0] == "uint8", (src.count, src.dtypes)
            assert abs(src.transform.a - SRC_PX) < 1e-9 and src.crs.to_epsg() == 32637
            img = np.ascontiguousarray(src.read().transpose(1, 2, 0))  # RGB, (H, W, 3)
        h, w = img.shape[:2]
        H, W = int(round(h * s)), int(round(w * s))
        out = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        np.save(os.path.join(a.out, f"{k}.npy"), out)
        print(f"{k}: {img.shape} -> {out.shape}, mean {out.reshape(-1, 3).mean(0).round(1)}", flush=True)
        del img, out

    cat = {r["id"]: r for r in json.load(open(os.path.join(ROOT, pair["instance_catalog"])))["instances"]}
    feats = json.load(open(os.path.join(ROOT, pair["pixel_vectors"])))["features"]
    rows = []
    for f in feats:
        c = cat[f["id"]]  # instanceId is the source annotation id; join on the tile:annotation key
        assert c["raster_id"] == f["raster_id"] and c["classId"] == f["classId"]
        g = shp_scale(shape(f["geometry"]), s, s, origin=(0, 0))
        minx, miny, maxx, maxy = g.bounds
        rows.append({"building_id": int(c["raster_id"]), "scene_id": 0, "label": int(c["classId"]) - 1,
                     "recommended": bool(c["recommended_for_instance_scoring"]), "source_id": c["id"],
                     "split": c["split"], "wkt_pre": g.wkt, "minx": minx, "miny": miny, "maxx": maxx, "maxy": maxy,
                     "area": g.area, "cx": g.centroid.x, "cy": g.centroid.y})
    allb = pd.DataFrame(rows).sort_values("building_id").reset_index(drop=True)
    allb.to_csv(os.path.join(a.out, "buildings_all.csv"), index=False)
    b = allb[allb.recommended].drop(columns="recommended").reset_index(drop=True)
    b.to_csv(os.path.join(a.out, "buildings.csv"), index=False)

    ids = np.zeros((H, W), np.int32)
    for k, w in enumerate(b.wkt_pre):
        g = shapely_wkt.loads(w)
        for p in getattr(g, "geoms", [g]):
            cv2.fillPoly(ids, [np.round(np.asarray(p.exterior.coords)).astype(np.int32)], k + 1)
    np.save(os.path.join(a.out, "ids.npy"), ids)
    npx = np.bincount(ids.ravel(), minlength=len(b) + 1)[1:]
    meta = {"source": ROOT, "pair": "full", "src_px_m": SRC_PX, "dst_px_m": DST_PX, "scale": s, "size": [H, W],
            "resample": "cv2.INTER_AREA", "rasterio": rasterio.__version__, "gdal": rasterio.__gdal_version__,
            "cv2": cv2.__version__, "n_records": len(allb), "n_scored": len(b),
            "scored_per_label": {int(k): int(v) for k, v in b.label.value_counts().sort_index().items()},
            "buildings_without_pixels": int((npx == 0).sum())}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
