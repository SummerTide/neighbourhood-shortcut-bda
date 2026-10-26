#!/usr/bin/env python
"""Co-register the Islahiye post-event image to the pre-event image (and so to the footprints, which are drawn
on the pre-event image). The delivered pair is offset by about 5-10 px at 0.5 m, varying across the scene.

1. Phase correlation of gradient-magnitude images in 512 px windows (stride 256), Hann-weighted; a window is kept
   if its peak response is high and it has texture. Gradient images make the match robust to damage and to
   radiometric differences between the dates.
2. A smooth displacement field: on a 128 px grid, the Gaussian-weighted (sigma 600 px) median of the window offsets
   that agree with a robust quadratic fit; the offset follows the terrain, so a global polynomial is not enough.
   Local parallax of single tall buildings cannot be corrected by a field and is left as it is.
3. The post image is resampled onto the pre grid with cv2.remap (bilinear); the offset is measured again.

Writes <out>/ with post.npy registered and symlinks to the other files of <data>; registration.json and
registration_windows.csv hold the field and the residuals.
Usage: python register_turkey.py --data data_turkey --out data_turkey_reg
"""
import argparse
import json
import os

import cv2
import numpy as np
import pandas as pd

WIN, STRIDE = 256, 128
STEP, SIGMA = 128, 600.0


def grad(img):
    g = cv2.GaussianBlur(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32), (0, 0), 1.0)
    return cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0), cv2.Sobel(g, cv2.CV_32F, 0, 1))


def measure(pre_g, post_g, min_resp):
    H, W = pre_g.shape
    hann = cv2.createHanningWindow((WIN, WIN), cv2.CV_32F)
    rows = []
    for y in range(0, H - WIN + 1, STRIDE):
        for x in range(0, W - WIN + 1, STRIDE):
            a, b = pre_g[y:y + WIN, x:x + WIN], post_g[y:y + WIN, x:x + WIN]
            tex = float(min(a.std(), b.std()))
            if tex < 3:
                continue
            (dx, dy), r = cv2.phaseCorrelate(np.ascontiguousarray(a), np.ascontiguousarray(b), hann)  # shift of post relative to pre
            rows.append(dict(x=x + WIN / 2, y=y + WIN / 2, dx=dx, dy=dy, resp=r, tex=tex))
    d = pd.DataFrame(rows)
    return d[(d.resp >= min_resp) & (d.dx.abs() < 40) & (d.dy.abs() < 40)].reset_index(drop=True), d


def design(x, y, W):
    u, v = x / W * 2 - 1, y / W * 2 - 1
    return np.stack([np.ones_like(u), u, v, u * u, u * v, v * v], 1)


def fit(d, W, it=20):
    A = design(d.x.to_numpy(), d.y.to_numpy(), W)
    w = d.resp.to_numpy().copy()
    for _ in range(it):  # IRLS with Tukey biweight on the 2D residual
        sw = np.sqrt(w)[:, None]
        cx = np.linalg.lstsq(A * sw, d.dx.to_numpy() * sw[:, 0], rcond=None)[0]
        cy = np.linalg.lstsq(A * sw, d.dy.to_numpy() * sw[:, 0], rcond=None)[0]
        res = np.hypot(d.dx - A @ cx, d.dy - A @ cy).to_numpy()
        c = 4.685 * max(1.4826 * np.median(res), 0.5)
        w = d.resp.to_numpy() * np.clip(1 - (res / c) ** 2, 0, None) ** 2
    return cx, cy, res, w > 0


def wmedian(v, w):
    o = np.argsort(v)
    c = np.cumsum(w[o])
    return v[o][np.searchsorted(c, c[-1] / 2)]


def local_field(d, H, W):
    """Coarse field on a STEP grid (nodes at 0, STEP, ...), Gaussian-weighted median of the inlier windows."""
    gy, gx = np.mgrid[0:H + STEP:STEP, 0:W + STEP:STEP].astype(np.float64)
    x, y, dx, dy, r = (d[k].to_numpy() for k in ("x", "y", "dx", "dy", "resp"))
    fx, fy = np.zeros(gx.shape, np.float32), np.zeros(gx.shape, np.float32)
    for i in range(gx.shape[0]):
        for j in range(gx.shape[1]):
            w = r * np.exp(-((x - gx[i, j]) ** 2 + (y - gy[i, j]) ** 2) / (2 * SIGMA ** 2))
            fx[i, j], fy[i, j] = wmedian(dx, w), wmedian(dy, w)
    return fx, fy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_turkey")
    ap.add_argument("--out", default="data_turkey_reg")
    ap.add_argument("--min-resp", type=float, default=0.15)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    pre = np.load(os.path.join(a.data, "pre.npy"))
    post = np.load(os.path.join(a.data, "post.npy"))
    H, W = pre.shape[:2]
    pre_g = grad(pre)
    d, d_all = measure(pre_g, grad(post), a.min_resp)
    print(f"windows: {len(d_all)} textured, {len(d)} kept; median dx {d.dx.median():.2f} dy {d.dy.median():.2f}", flush=True)
    cx, cy, res, inl = fit(d, W)
    d["inlier"], d["res"] = inl, res
    print(f"field fit: {inl.sum()} inliers, median residual {np.median(res[inl]):.2f} px", flush=True)

    fx, fy = local_field(d[d.inlier], H, W)
    print(f"field dx [{fx.min():.2f}, {fx.max():.2f}] dy [{fy.min():.2f}, {fy.max():.2f}]", flush=True)
    # map(x, y) = (x + dx, y + dy) samples the post image; the coarse field is upsampled bilinearly in row strips
    reg = np.empty_like(post)
    xs = np.arange(W, dtype=np.float32)
    for y0 in range(0, H, 1024):
        y1 = min(H, y0 + 1024)
        ys = np.arange(y0, y1, dtype=np.float32)
        u, v = np.meshgrid(xs / STEP, ys / STEP)
        mx = xs[None, :] + cv2.remap(fx, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        my = ys[:, None] + cv2.remap(fy, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        reg[y0:y1] = cv2.remap(post, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    np.save(os.path.join(a.out, "post.npy"), reg)

    del post
    d2, _ = measure(pre_g, grad(reg), a.min_resp)
    after = np.hypot(d2.dx, d2.dy)
    print(f"after: {len(d2)} windows, median dx {d2.dx.median():.2f} dy {d2.dy.median():.2f}, median |d| {after.median():.2f}, "
          f"90th pct |d| {after.quantile(.9):.2f}", flush=True)
    for f in os.listdir(a.data):
        if f != "post.npy" and not os.path.exists(os.path.join(a.out, f)):
            os.symlink(os.path.abspath(os.path.join(a.data, f)), os.path.join(a.out, f))
    d.to_csv(os.path.join(a.out, "registration_windows.csv"), index=False)
    d2.to_csv(os.path.join(a.out, "registration_windows_after.csv"), index=False)
    np.savez(os.path.join(a.out, "registration_field.npz"), fx=fx, fy=fy, step=STEP)
    json.dump({"model": f"post(x+dx, y+dy) -> pre(x, y); field on a {STEP} px grid, Gaussian-weighted median (sigma {SIGMA} px) "
               "of the windows that are inliers of a robust quadratic fit", "window": WIN, "stride": STRIDE, "windows_kept": len(d), "inliers": int(inl.sum()),
               "before_median_abs": float(np.hypot(d.dx, d.dy).median()), "after_median_abs": float(after.median()),
               "after_p90_abs": float(after.quantile(.9))}, open(os.path.join(a.out, "registration.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
