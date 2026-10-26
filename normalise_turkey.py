#!/usr/bin/env python
"""Linear per-channel colour normalisation of the registered Islahiye pair to the xBD statistics: each date is
mapped so that its urban area (5-95% building centroid box) has the mean and standard deviation of the xBD
images of the same date (300 random scenes). The Islahiye scene is brighter and has about 1.3x the contrast.
Usage: python normalise_turkey.py --data data_turkey_reg --out data_turkey_regnorm
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_turkey_reg")
    ap.add_argument("--out", default="data_turkey_regnorm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    s = pd.read_csv("data/scenes.csv")
    b = pd.read_csv(os.path.join(a.data, "buildings.csv"))
    x0, x1, y0, y1 = (int(v) for v in (b.cx.quantile(.05), b.cx.quantile(.95), b.cy.quantile(.05), b.cy.quantile(.95)))
    idx = np.random.default_rng(0).choice(len(s), 300, replace=False)
    stats = {}
    for k in ("pre", "post"):
        mm = np.memmap(f"data/{k}.u8", np.uint8, "r", shape=(len(s), 1024, 1024, 3))
        ref = np.stack([mm[i, ::16, ::16] for i in idx]).reshape(-1, 3).astype(np.float64)
        img = np.load(os.path.join(a.data, f"{k}.npy"))
        u = img[y0:y1:4, x0:x1:4].reshape(-1, 3).astype(np.float64)
        g, o = ref.std(0) / u.std(0), ref.mean(0) - u.mean(0) * ref.std(0) / u.std(0)
        out = np.empty_like(img)
        for r in range(0, img.shape[0], 1024):
            out[r:r + 1024] = np.clip(img[r:r + 1024] * g + o, 0, 255).round().astype(np.uint8)
        np.save(os.path.join(a.out, f"{k}.npy"), out)
        stats[k] = {"gain": g.round(4).tolist(), "offset": o.round(2).tolist(), "xbd_mean": ref.mean(0).round(1).tolist(),
                    "xbd_std": ref.std(0).round(1).tolist(), "urban_mean": u.mean(0).round(1).tolist(), "urban_std": u.std(0).round(1).tolist()}
        print(k, stats[k], flush=True)
        del img, out
    for f in os.listdir(a.data):
        if f not in ("pre.npy", "post.npy") and not os.path.exists(os.path.join(a.out, f)):
            os.symlink(os.path.realpath(os.path.join(a.data, f)), os.path.join(a.out, f))
    json.dump({"urban_box": [x0, x1, y0, y1], **stats}, open(os.path.join(a.out, "normalisation.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
