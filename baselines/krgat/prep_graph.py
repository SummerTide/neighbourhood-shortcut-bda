#!/usr/bin/env python
"""Per-event building graphs from the geographic xBD annotations (KR-GAT Sec. 3.1, 4.1).

For every labelled building: the pre-event polygon in lon/lat (features.lng_lat of the label JSON,
matched by uid), projected to UTM (one zone per event: the zone of the event's median longitude);
centroid, area (m^2) and perimeter (m). Graphs: one directed k-NN graph (k = 16, self excluded)
per event *within each official split* (train = train + tier3, val = hold, test), so no split sees
another split's buildings. Writes data/graph.npz.
"""
import json
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd
import shapely
from scipy.spatial import cKDTree

from common import DATA, EVENT_ROLLUP, K_NN, PROJ, ROLLUPS, SPLITS, XBD, lonlat_to_utm, utm_zone


def read_scene(args):
    split, stem = args
    out = {}
    for kind in ("post", "pre"):  # pre overrides post, as prep_official.py prefers the pre polygon
        with open(f"{XBD}/{split}/labels/{stem}_{kind}_disaster.json") as fh:
            d = json.load(fh)
        for f in d["features"]["lng_lat"]:
            out[f["properties"]["uid"]] = f["wkt"]
    return out


def main():
    t0 = time.time()
    scenes = pd.read_csv(f"{PROJ}/data/scenes.csv", usecols=["scene_id", "split", "official", "disaster", "stem"])
    b = pd.read_csv(f"{PROJ}/data/buildings.csv", usecols=["building_id", "scene_id", "uid", "label", "cx", "cy"])
    b = b[b.label >= 0].merge(scenes, on="scene_id").sort_values("building_id").reset_index(drop=True)
    print(f"{len(b)} labelled buildings", flush=True)
    need = scenes[scenes.scene_id.isin(b.scene_id.unique())]
    with Pool(max(1, len(os.sched_getaffinity(0)))) as pool:
        maps = pool.map(read_scene, list(zip(need.split, need.stem)), chunksize=16)
    lut = {}
    for m in maps:
        lut.update(m)
    wkts = b.uid.map(lut)
    assert wkts.notna().all(), f"{wkts.isna().sum()} buildings without lng_lat"
    print(f"read {len(need)} scenes ({time.time() - t0:.0f}s)", flush=True)
    geoms = shapely.from_wkt(wkts.to_numpy())
    c = shapely.centroid(geoms)
    lon, lat = shapely.get_x(c), shapely.get_y(c)
    x, y = np.zeros(len(b)), np.zeros(len(b))
    area, perim = np.zeros(len(b)), np.zeros(len(b))
    events = sorted(b.disaster.unique())
    zones = {}
    for e in events:
        m = (b.disaster == e).to_numpy()
        zone, south = utm_zone(np.median(lon[m])), bool(np.median(lat[m]) < 0)
        zones[e] = (zone, south)
        proj = shapely.transform(geoms[m], lambda xy: np.stack(lonlat_to_utm(xy[:, 0], xy[:, 1], zone, south), 1))
        pc = shapely.centroid(proj)
        x[m], y[m] = shapely.get_x(pc), shapely.get_y(pc)
        area[m], perim[m] = shapely.area(proj), shapely.length(proj)
        print(f"  {e}: UTM zone {zone}{'S' if south else 'N'}, {m.sum()} buildings", flush=True)
    split = b.official.map({s: i for i, s in enumerate(SPLITS)}).to_numpy().astype(np.int8)
    event = b.disaster.map({e: i for i, e in enumerate(events)}).to_numpy().astype(np.int16)
    rollup = b.disaster.map(lambda e: ROLLUPS.index(EVENT_ROLLUP[e])).to_numpy().astype(np.int8)
    nbr = np.full((len(b), K_NN), -1, np.int64)
    nbr_d = np.full((len(b), K_NN), np.inf, np.float32)
    xy = np.stack([x, y], 1)
    for s in range(len(SPLITS)):
        for e in range(len(events)):
            idx = np.flatnonzero((split == s) & (event == e))
            if len(idx) < 2:
                continue
            k = min(K_NN, len(idx) - 1)
            d, j = cKDTree(xy[idx]).query(xy[idx], k=k + 1)
            j, d = np.atleast_2d(j), np.atleast_2d(d)
            own = j == np.arange(len(idx))[:, None]
            # drop the node itself (or the farthest one if duplicates pushed it out of the list)
            drop = np.where(own.any(1), own.argmax(1), k)
            keep = np.ones_like(own)
            keep[np.arange(len(idx)), drop] = False
            nbr[idx, :k] = idx[j[keep].reshape(len(idx), k)]
            nbr_d[idx, :k] = d[keep].reshape(len(idx), k)
    # sanity: metres per pixel between nearest neighbours of the same tile (expect ~0.3-0.6)
    j0 = nbr[:, 0]
    same = (j0 >= 0) & (b.scene_id.to_numpy() == np.where(j0 >= 0, b.scene_id.to_numpy()[np.maximum(j0, 0)], -1))
    pix = np.hypot(b.cx.to_numpy() - b.cx.to_numpy()[np.maximum(j0, 0)], b.cy.to_numpy() - b.cy.to_numpy()[np.maximum(j0, 0)])
    ok = same & (pix > 5)
    ratio = nbr_d[ok, 0] / pix[ok]
    print("m/px of nearest same-tile neighbours: median %.3f, 5-95%% %.3f-%.3f" % (np.median(ratio), *np.percentile(ratio, [5, 95])))
    for s, name in enumerate(SPLITS):
        m = split == s
        print(f"{name}: {m.sum()} nodes, median NN distance {np.median(nbr_d[m, 0]):.1f} m, "
              f"median 16th-NN distance {np.median(nbr_d[m, -1][np.isfinite(nbr_d[m, -1])]):.1f} m, "
              f"nodes with < 16 neighbours {(nbr[m] < 0).any(1).sum()}", flush=True)
    os.makedirs(DATA, exist_ok=True)
    np.savez(f"{DATA}/graph.npz", building_id=b.building_id.to_numpy(), scene_id=b.scene_id.to_numpy(),
             label=b.label.to_numpy().astype(np.int8), split=split, event=event, rollup=rollup,
             x=x, y=y, lon=lon, lat=lat, area_m2=area, perim_m=perim, nbr=nbr, nbr_d=nbr_d,
             events=np.array(events), zones=np.array([zones[e][0] for e in events]),
             south=np.array([zones[e][1] for e in events]))
    print(f"wrote {DATA}/graph.npz ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
