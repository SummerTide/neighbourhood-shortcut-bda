#!/usr/bin/env python
"""Integer-pixel variant of register_turkey.py: the same displacement field, rounded to whole pixels and applied
with nearest-neighbour sampling, so the post image is shifted but not resampled (no blur).
Usage: python register_turkey_int.py --data data_turkey --field data_turkey_reg --out data_turkey_regint
"""
import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_turkey")
    ap.add_argument("--field", default="data_turkey_reg")
    ap.add_argument("--out", default="data_turkey_regint")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    z = np.load(os.path.join(a.field, "registration_field.npz"))
    fx, fy, step = z["fx"], z["fy"], int(z["step"])
    post = np.load(os.path.join(a.data, "post.npy"))
    H, W = post.shape[:2]
    reg = np.empty_like(post)
    xs = np.arange(W, dtype=np.float32)
    for y0 in range(0, H, 1024):
        y1 = min(H, y0 + 1024)
        ys = np.arange(y0, y1, dtype=np.float32)
        u, v = np.meshgrid(xs / step, ys / step)
        mx = xs[None, :] + np.round(cv2.remap(fx, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        my = ys[:, None] + np.round(cv2.remap(fy, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE))
        reg[y0:y1] = cv2.remap(post, mx.astype(np.float32), my.astype(np.float32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)
    np.save(os.path.join(a.out, "post.npy"), reg)
    for f in os.listdir(a.data):
        if f != "post.npy" and not os.path.exists(os.path.join(a.out, f)):
            os.symlink(os.path.realpath(os.path.join(a.data, f)), os.path.join(a.out, f))


if __name__ == "__main__":
    main()
