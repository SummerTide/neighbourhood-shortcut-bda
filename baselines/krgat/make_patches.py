#!/usr/bin/env python
"""Pre/post combined (PPC) building patches for every labelled xBD building (KR-GAT Sec. 3.1).

Patch = tight bounding box of the pre-event polygon (pixel annotation) plus a small margin
(max(8 px, 10 % of the box side) per side), cropped from the pre and the post tile and resized to
PATCH x PATCH; stored as uint8 (n, PATCH, PATCH, 6) = [pre RGB, post RGB] in data/ppc_<PATCH>.u8,
rows in node order (labelled buildings in building_id order, as data/graph.npz).
"""
import os
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

from common import DATA, IMG, PATCH, PROJ, ppc_box, ppc_patch

N_SCENES = None
OUT = f"{DATA}/ppc_{PATCH}.u8"


def work(args):
    sid, rows, boxes, n = args
    pre = np.memmap(f"{PROJ}/data/pre.u8", np.uint8, "r", shape=(N_SCENES, IMG, IMG, 3))[sid]
    post = np.memmap(f"{PROJ}/data/post.u8", np.uint8, "r", shape=(N_SCENES, IMG, IMG, 3))[sid]
    pre, post = np.ascontiguousarray(pre), np.ascontiguousarray(post)
    bank = np.memmap(OUT, np.uint8, "r+", shape=(n, PATCH, PATCH, 6))
    out = np.stack([ppc_patch(pre, post, ppc_box(*bx)) for bx in boxes])
    bank[rows[0]:rows[-1] + 1] = out
    bank.flush()
    del bank
    return len(rows)


def main():
    global N_SCENES
    t0 = time.time()
    scenes = pd.read_csv(f"{PROJ}/data/scenes.csv", usecols=["scene_id"])
    N_SCENES = len(scenes)
    b = pd.read_csv(f"{PROJ}/data/buildings.csv", usecols=["building_id", "scene_id", "label", "minx", "miny", "maxx", "maxy"])
    b = b[b.label >= 0].sort_values("building_id").reset_index(drop=True)
    g = np.load(f"{DATA}/graph.npz")
    assert (g["building_id"] == b.building_id.to_numpy()).all(), "node order differs from graph.npz"
    n = len(b)
    np.memmap(OUT, np.uint8, "w+", shape=(n, PATCH, PATCH, 6)).flush()
    jobs = []
    for sid, idx in b.groupby("scene_id").indices.items():
        assert (np.diff(idx) == 1).all()
        jobs.append((int(sid), idx, b.loc[idx, ["minx", "miny", "maxx", "maxy"]].to_numpy(np.float64), n))
    done = 0
    with Pool(len(os.sched_getaffinity(0))) as pool:
        for k, m in enumerate(pool.imap_unordered(work, jobs, chunksize=4)):
            done += m
            if (k + 1) % 1000 == 0:
                print(f"{k + 1}/{len(jobs)} scenes, {done} patches ({time.time() - t0:.0f}s)", flush=True)
    print(f"wrote {OUT}: {n} x {PATCH} x {PATCH} x 6 ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
