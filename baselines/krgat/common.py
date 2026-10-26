"""Shared constants and helpers for the KR-GAT re-implementation (Hasan & Yeum, ECCV 2026, arXiv 2608.20548).

Node set: every labelled xBD building (label 0-3) in footprint_bda/data/buildings.csv order; the
test nodes in that order are exactly the harness.py test buildings.
"""
import os
import sys

import cv2
import numpy as np

PROJ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repository root
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = f"{ROOT}/data"
XBD = os.environ.get("XBD", f"{PROJ}/datasets/xBD")
R50 = f"{PROJ}/external/checkpoints/resnet50-11ad3fa6.pth"
sys.path.insert(0, PROJ)
sys.path.insert(0, os.path.join(PROJ, "analysis"))

IMG = 1024
PATCH = 112          # stored PPC patch side (px); the encoder sees it upsampled to INPUT
INPUT = 224
MARGIN_MIN = 8.0     # PPC crop: tight polygon box + max(MARGIN_MIN, MARGIN_FRAC * box side) per side
MARGIN_FRAC = 0.1
K_NN = 16
SPLITS = ("train", "val", "test")
# Disaster-type roll-ups of Sec. 4.1, from the event token of the xBD file names.
ROLLUPS = ("earthquake", "flood", "hurricane", "tornado", "tsunami", "volcano", "wildfire")
EVENT_ROLLUP = {
    "hurricane-florence": "hurricane", "hurricane-harvey": "hurricane", "hurricane-matthew": "hurricane",
    "hurricane-michael": "hurricane", "santa-rosa-wildfire": "wildfire", "socal-fire": "wildfire",
    "woolsey-fire": "wildfire", "pinery-bushfire": "wildfire", "portugal-wildfire": "wildfire",
    "midwest-flooding": "flood", "nepal-flooding": "flood", "moore-tornado": "tornado",
    "tuscaloosa-tornado": "tornado", "joplin-tornado": "tornado", "palu-tsunami": "tsunami",
    "sunda-tsunami": "tsunami", "guatemala-volcano": "volcano", "lower-puna-volcano": "volcano",
    "mexico-earthquake": "earthquake",
}


# ---------------------------------------------------------------- UTM (WGS84, Krueger series)
_A, _F = 6378137.0, 1 / 298.257223563


def utm_zone(lon):
    return int(np.floor((lon + 180.0) / 6.0)) + 1


def lonlat_to_utm(lon, lat, zone, south):
    """Transverse Mercator (UTM) easting/northing in metres for arrays of lon/lat (degrees)."""
    lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
    n = _F / (2 - _F)
    a_ = _A / (1 + n) * (1 + n ** 2 / 4 + n ** 4 / 64)
    alpha = (n / 2 - 2 * n ** 2 / 3 + 5 * n ** 3 / 16, 13 * n ** 2 / 48 - 3 * n ** 3 / 5, 61 * n ** 3 / 240)
    lon0 = np.radians((zone - 1) * 6 - 180 + 3)
    phi, lam = np.radians(lat), np.radians(lon) - lon0
    c = 2 * np.sqrt(n) / (1 + n)
    t = np.sinh(np.arctanh(np.sin(phi)) - c * np.arctanh(c * np.sin(phi)))
    xi = np.arctan2(t, np.cos(lam))
    eta = np.arctanh(np.sin(lam) / np.sqrt(1 + t ** 2))
    e = eta + sum(al * np.cos(2 * j * xi) * np.sinh(2 * j * eta) for j, al in enumerate(alpha, 1))
    nn = xi + sum(al * np.sin(2 * j * xi) * np.cosh(2 * j * eta) for j, al in enumerate(alpha, 1))
    k0 = 0.9996
    return 500000.0 + k0 * a_ * e, (10000000.0 if south else 0.0) + k0 * a_ * nn


# ---------------------------------------------------------------- PPC patches
def ppc_box(minx, miny, maxx, maxy):
    """Tight polygon bounding box plus a small margin (pixel coordinates of the 1024 tile)."""
    m = max(MARGIN_MIN, MARGIN_FRAC * max(maxx - minx, maxy - miny))
    return minx - m, miny - m, maxx + m, maxy + m


def crop_box(img, box, size=PATCH):
    """Crop box (x0, y0, x1, y1) from an (H, W, 3) tile (zero outside), resize to size x size."""
    x0, y0, x1, y1 = box
    c0, r0 = int(np.floor(x0)), int(np.floor(y0))
    c1, r1 = max(int(np.ceil(x1)), c0 + 2), max(int(np.ceil(y1)), r0 + 2)
    out = np.zeros((r1 - r0, c1 - c0, 3), np.uint8)
    rs, re, cs, ce = max(0, r0), min(img.shape[0], r1), max(0, c0), min(img.shape[1], c1)
    if re > rs and ce > cs:
        out[rs - r0:re - r0, cs - c0:ce - c0] = img[rs:re, cs:ce]
    interp = cv2.INTER_AREA if (out.shape[0] > size and out.shape[1] > size) else cv2.INTER_LINEAR
    return cv2.resize(out, (size, size), interpolation=interp)


def ppc_patch(pre, post, box, size=PATCH):
    """(size, size, 6) uint8: pre RGB then post RGB of the same window."""
    return np.concatenate([crop_box(pre, box, size), crop_box(post, box, size)], 2)


def inpainted_ppc_patch(pre, post, parts, box, margin=4, radius=7, size=PATCH):
    """'Building gone': the footprint plus a `margin` px rim inpainted (Telea, radius 7, as harness.py
    and analysis/context_swap.py do) in both images, then cropped like ppc_patch. Inpainting runs on
    a window around the patch, which gives the same pixels as inpainting the whole tile."""
    pad = 3 * radius + margin + 4
    x0, y0, x1, y1 = box
    c0, r0 = max(0, int(np.floor(x0)) - pad), max(0, int(np.floor(y0)) - pad)
    c1, r1 = min(IMG, int(np.ceil(x1)) + pad), min(IMG, int(np.ceil(y1)) + pad)
    mask = np.zeros((r1 - r0, c1 - c0), np.uint8)
    for p in parts:
        cv2.fillPoly(mask, [np.round(p - np.array([c0, r0], np.float32)).astype(np.int32)], 1)
    k = np.ones((2 * margin + 1, 2 * margin + 1), np.uint8)
    mask = (cv2.dilate(mask, k) > 0).astype(np.uint8)
    shifted = (x0 - c0, y0 - r0, x1 - c0, y1 - r0)
    out = [crop_box(cv2.inpaint(np.ascontiguousarray(img[r0:r1, c0:c1]), mask, radius, cv2.INPAINT_TELEA),
                    shifted, size) for img in (pre, post)]
    return np.concatenate(out, 2)
