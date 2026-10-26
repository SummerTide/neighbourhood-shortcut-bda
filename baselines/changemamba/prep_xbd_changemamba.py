#!/usr/bin/env python
"""xBD in ChangeMamba's layout, for training MambaBDA under our protocol.

train  = official train + tier3, every 1024 tile split into four 512 patches named
         <stem>_{pre,post}_disaster_<i>_<j>.png (images and masks), listed as <stem>_<i>_<j>
         (ChangeMamba's DamageAssessmentDataset resolves training names that way)
hold   = model selection, full tiles as <stem>_{pre,post}_disaster.png
test   = final evaluation, same layout as hold
Masks follow xBD's target convention (pre: building 1; post: damage 1-4, un-classified 255,
0 background). The release's train / tier3 targets folders are incomplete, so training masks are
rasterised from the label JSONs; hold / test use the (complete) official targets. ChangeMamba's
loader reads masks as 3-channel images (mask[:, :, 0]), so masks are written with 3 channels.
Images are re-encoded as 3-channel PNGs: the release's PNGs carry a tRNS chunk that makes
readers (cv2 IMREAD_UNCHANGED, imageio) return RGBA. --masks-only rewrites only the masks.
Usage: python prep_xbd_changemamba.py --xbd .../Datasets/xBD --out data/xBD_cm --workers 16
"""
import argparse
import os
from multiprocessing import Pool

import json

import cv2
import numpy as np
from shapely import wkt

PATCH = 512
DAMAGE = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4, "un-classified": 255}


def rasterise(label_json, shape, kind):
    mask = np.zeros(shape, np.uint8)
    for ft in json.load(open(label_json))["features"]["xy"]:
        value = 1 if kind == "pre" else DAMAGE[ft["properties"].get("subtype", "un-classified")]
        geom = wkt.loads(ft["wkt"])
        for poly in ([geom] if geom.geom_type == "Polygon" else list(geom.geoms)):
            cv2.fillPoly(mask, [np.round(np.asarray(poly.exterior.coords)[:, :2]).astype(np.int32)], value)
    return mask


def three(mask):
    return np.repeat(mask[..., None], 3, axis=2)


def split_tile(args):
    xbd, split, stem, out, masks_only = args
    names = []
    for kind in ("pre", "post"):
        img = None if masks_only else cv2.imread(os.path.join(xbd, split, "images", f"{stem}_{kind}_disaster.png"),
                                                  cv2.IMREAD_COLOR)
        msk = rasterise(os.path.join(xbd, split, "labels", f"{stem}_{kind}_disaster.json"), (1024, 1024), kind)
        for i in range(msk.shape[0] // PATCH):
            for j in range(msk.shape[1] // PATCH):
                name = f"{stem}_{kind}_disaster_{i}_{j}.png"
                sl = (slice(i * PATCH, (i + 1) * PATCH), slice(j * PATCH, (j + 1) * PATCH))
                if not masks_only:
                    cv2.imwrite(os.path.join(out, "train", "images", name), img[sl])
                cv2.imwrite(os.path.join(out, "train", "masks", name), three(msk[sl]))
                if kind == "pre":
                    names.append(f"{stem}_{i}_{j}")
    return names


def stems(xbd, split):
    return sorted(f[: -len("_pre_disaster.png")] for f in os.listdir(os.path.join(xbd, split, "images"))
                  if f.endswith("_pre_disaster.png"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xbd", default="datasets/xBD")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--masks-only", action="store_true")
    args = ap.parse_args()
    for d in ("train/images", "train/masks", "hold/images", "hold/masks", "test/images", "test/masks"):
        os.makedirs(os.path.join(args.out, d), exist_ok=True)
    jobs = [(args.xbd, s, stem, args.out, args.masks_only) for s in ("train", "tier3") for stem in stems(args.xbd, s)]
    with Pool(args.workers) as pool:
        names = [n for ns in pool.imap(split_tile, jobs, chunksize=8) for n in ns]
    with open(os.path.join(args.out, "train_set.txt"), "w") as fh:
        fh.write("\n".join(names) + "\n")
    for split in ("hold", "test"):
        ss = stems(args.xbd, split)
        for stem in ss:
            for kind in ("pre", "post"):
                name = f"{stem}_{kind}_disaster.png"
                dst = os.path.join(args.out, split, "images", name)
                if os.path.islink(dst):
                    os.remove(dst)
                if not args.masks_only:
                    cv2.imwrite(dst, cv2.imread(os.path.join(args.xbd, split, "images", name), cv2.IMREAD_COLOR))
                dst = os.path.join(args.out, split, "masks", name)
                if os.path.islink(dst):
                    os.remove(dst)
                target = cv2.imread(os.path.join(args.xbd, split, "targets", f"{stem}_{kind}_disaster_target.png"),
                                    cv2.IMREAD_UNCHANGED)
                cv2.imwrite(dst, three(target))
        with open(os.path.join(args.out, f"{split}_set.txt"), "w") as fh:
            fh.write("\n".join(ss) + "\n")
    print(f"train patches {len(names)} from {len(jobs)} tiles; hold / test tiles linked", flush=True)


if __name__ == "__main__":
    main()
