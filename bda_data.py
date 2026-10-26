"""Building-level samples for footprint-conditioned damage assessment on xBD.

A sample is one inventory building: the footprint is the pre-event polygon, the
crop window is a square around its bounding box with CONTEXT_PX of context on
each side (so a displaced roof stays in view), and the inputs are the post-event
crop, optionally the pre-event crop of the same window, and the footprint mask,
all resized to SIZE x SIZE.
"""
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from shapely import wkt as shapely_wkt

cv2.setNumThreads(0)  # no OpenCV thread pools inside DataLoader workers

IMG = 1024
SIZE = 224
CONTEXT_PX = 32
MIN_SIDE = 64
SWAP_MARGIN = 4  # px around the footprint kept with the building when its surroundings are swapped
N_CLASSES = int(os.environ.get("BDA_N_CLASSES", 4))  # 3 for DisasterM3 (intact / damaged / destroyed)


def load_tables(data):
    scenes = pd.read_csv(os.path.join(data, "scenes.csv"))
    buildings = pd.read_csv(os.path.join(data, "buildings.csv"))
    buildings = buildings.merge(scenes[["scene_id", "official", "disaster"]], on="scene_id")
    return scenes, buildings


def polygons(wkts):
    out = []
    for w in wkts:
        g = shapely_wkt.loads(w)
        geoms = getattr(g, "geoms", [g])
        out.append([np.asarray(p.exterior.coords, np.float32) for p in geoms if not p.is_empty])
    return out


def load_conditions(cond_dir, building_ids):
    """Evaluation conditions written by make_conditions.py, for the given buildings.

    Returns the conditions (name, level, method, post/pre image paths or None for
    the original images, per-building footprint offsets in post-image pixels) and
    the scene ids addressing the rows of the condition images.
    """
    with open(os.path.join(cond_dir, "manifest.json")) as fh:
        manifest = json.load(fh)
    scene_ids = pd.read_csv(os.path.join(cond_dir, manifest["index"])).scene_id.to_numpy()
    z = np.load(os.path.join(cond_dir, manifest["offsets"]))
    pos = pd.Index(z["building_id"]).get_indexer(building_ids)
    assert (pos >= 0).all(), "buildings missing from the condition offsets"
    conds = []
    for c in manifest["conditions"]:
        path = lambda f: None if f is None else os.path.join(cond_dir, f)
        conds.append({**c, "post": path(c["post"]), "pre": path(c["pre"]), "offsets": z[c["name"]][pos]})
    return conds, scene_ids


class ImageSource:
    """uint8 images of shape (n, 1024, 1024, 3) addressed by scene id."""

    def __init__(self, path, n_rows, scene_ids=None):
        self.path, self.n_rows = path, n_rows
        self.row_of = None if scene_ids is None else {int(s): i for i, s in enumerate(scene_ids)}
        self.mm = None

    def __getitem__(self, scene_id):
        if self.mm is None:  # opened lazily in each DataLoader worker
            self.mm = np.memmap(self.path, np.uint8, "r", shape=(self.n_rows, IMG, IMG, 3))
        return self.mm[scene_id if self.row_of is None else self.row_of[scene_id]]


def crop(img, cx, cy, side):
    c0, r0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    out = np.zeros((side, side, 3), np.uint8)
    rs, re, cs, ce = max(0, r0), min(IMG, r0 + side), max(0, c0), min(IMG, c0 + side)
    if re > rs and ce > cs:
        out[rs - r0:re - r0, cs - c0:ce - c0] = img[rs:re, cs:ce]
    interp = cv2.INTER_AREA if side > SIZE else cv2.INTER_LINEAR
    return cv2.resize(out, (SIZE, SIZE), interpolation=interp), c0, r0


class BuildingDataset(torch.utils.data.Dataset):
    """tight_margin (px) blanks every crop pixel farther than that from the footprint,
    so a model sees the building itself and not its neighbourhood (context_px then
    only sets how much of the window the building fills).

    context_swap (training only) replaces, with probability context_swap_p, everything
    farther than SWAP_MARGIN px from the footprint while the label stays the building's:
      donor  the surroundings of another building (half the time a damaged one), at
             the same scale, in the post (and pre) crop
      pre    the building's own surroundings before the event (swap_pre_src), in the
             post crop: damage elsewhere in the window can no longer explain the label
    The swap can be made harder to recognise (a model may learn to ignore context only
    where it sees a seam): swap_blend "poisson" blends the building into the new
    surroundings (cv2.seamlessClone) instead of a hard cut, swap_margin (lo, hi) draws
    the kept rim width (px) per sample, and swap_donor "event" takes donors from the
    building's own disaster event (similar radiometry) instead of anywhere.

    tight_view (context_px, margin) adds a second view of the same building (keys post_t,
    mask_t, pre_t): a crop with that much context, blanked farther than margin px from the
    footprint and never context-swapped, for the building stream of a two-stream model.
    """

    def __init__(self, buildings, post_src, pre_src=None, train=False, jitter_px=0.0, offsets=None,
                 context_px=CONTEXT_PX, tight_margin=None, context_swap=None, context_swap_p=0.5,
                 swap_pre_src=None, swap_blend="hard", swap_margin=(SWAP_MARGIN, SWAP_MARGIN), swap_donor="any",
                 tight_view=None):
        b = buildings.reset_index(drop=True)
        self.polys = polygons(b.wkt_pre)
        self.scene = b.scene_id.to_numpy().astype(int)
        self.box = b[["minx", "miny", "maxx", "maxy"]].to_numpy(np.float32)
        self.labels = b.label.to_numpy().astype(int)
        self.post_src, self.pre_src, self.train, self.jitter_px = post_src, pre_src, train, jitter_px
        self.offsets = None if offsets is None else np.asarray(offsets, np.float32)
        self.context_px, self.tight_margin = context_px, tight_margin
        self.context_swap, self.context_swap_p, self.swap_pre_src = context_swap, context_swap_p, swap_pre_src
        self.swap_blend, self.swap_margin, self.swap_donor = swap_blend, tuple(swap_margin), swap_donor
        self.tight_view = tight_view
        self.damaged = np.flatnonzero(self.labels >= 1)
        if swap_donor == "event":
            ev = b.disaster.to_numpy()
            self.event = pd.factorize(ev)[0]
            self.event_all = [np.flatnonzero(self.event == e) for e in range(self.event.max() + 1)]
            self.event_dmg = [a[self.labels[a] >= 1] for a in self.event_all]

    def _donor(self, i):
        """Another building whose surroundings replace building i's: half the time a damaged one."""
        if self.swap_donor == "event":
            dmg, pool = self.event_dmg[self.event[i]], self.event_all[self.event[i]]
            return np.random.choice(dmg) if np.random.rand() < 0.5 and len(dmg) else np.random.choice(pool)
        return np.random.choice(self.damaged) if np.random.rand() < 0.5 and len(self.damaged) else \
            np.random.randint(len(self.labels))

    def _blend(self, own, other, keep):
        """own where keep (the building and its rim), other elsewhere."""
        if self.swap_blend == "poisson":
            x, y, w, h = cv2.boundingRect(keep.astype(np.uint8))
            try:  # seamlessClone centres the mask's bounding box on the given point
                return cv2.seamlessClone(np.ascontiguousarray(own), np.ascontiguousarray(other),
                                         keep.astype(np.uint8) * 255, (x + w // 2, y + h // 2), cv2.NORMAL_CLONE)
            except cv2.error:
                pass
        return np.where(keep[..., None], own, other)

    def __len__(self):
        return len(self.labels)

    def _view(self, i, context_px, dx, dy):
        """post crop, footprint mask and pre crop of building i with context_px of context,
        the post window and mask displaced by (dx, dy)."""
        minx, miny, maxx, maxy = self.box[i]
        side = int(max(MIN_SIDE, np.ceil(max(maxx - minx, maxy - miny)) + 2 * context_px))
        cx, cy = (minx + maxx) / 2, (miny + maxy) / 2
        post, c0, r0 = crop(self.post_src[self.scene[i]], cx + dx, cy + dy, side)
        scale = SIZE / side
        mask = np.zeros((SIZE, SIZE), np.uint8)
        for p in self.polys[i]:
            q = (p - np.array([c0 - dx, r0 - dy], np.float32)) * scale
            cv2.fillPoly(mask, [np.round(q).astype(np.int32)], 255)
        pre = None if self.pre_src is None else crop(self.pre_src[self.scene[i]], cx, cy, side)[0]
        return post, mask, pre, side, scale, cx, cy

    @staticmethod
    def _blank(post, mask, pre, margin, scale):
        k = 2 * int(round(margin * scale)) + 1
        outside = cv2.dilate(mask, np.ones((k, k), np.uint8)) == 0
        post[outside] = 0
        if pre is not None:
            pre[outside] = 0

    def __getitem__(self, i):
        # Displace the footprint against the post-event image (a locally rigid
        # misregistration): the post crop window and the mask move together by
        # (dx, dy), the pre crop stays at the inventory window. Offsets come from an
        # evaluation condition, random jitter from training-time augmentation.
        dx, dy = (0.0, 0.0) if self.offsets is None else self.offsets[i]
        if self.train and self.jitter_px > 0:
            jx, jy = np.random.normal(0, self.jitter_px, 2)
            dx, dy = dx + jx, dy + jy
        post, mask, pre, side, scale, cx, cy = self._view(i, self.context_px, dx, dy)
        if self.train and self.context_swap and np.random.rand() < self.context_swap_p:
            lo, hi = self.swap_margin
            margin = lo if lo == hi else np.random.uniform(lo, hi)
            k = 2 * int(round(margin * scale)) + 1
            keep = cv2.dilate(mask, np.ones((k, k), np.uint8)) > 0
            if self.context_swap == "donor":
                j = self._donor(i)
                jx, jy = self.box[j, [0, 2]].mean(), self.box[j, [1, 3]].mean()
                post = self._blend(post, crop(self.post_src[self.scene[j]], jx, jy, side)[0], keep)
                if pre is not None:
                    pre = self._blend(pre, crop(self.pre_src[self.scene[j]], jx, jy, side)[0], keep)
            else:
                post = self._blend(post, crop(self.swap_pre_src[self.scene[i]], cx + dx, cy + dy, side)[0], keep)
        if self.tight_margin is not None:
            self._blank(post, mask, pre, self.tight_margin, scale)
        out = {"post": torch.from_numpy(post), "mask": torch.from_numpy(mask), "label": int(self.labels[i]), "index": i}
        if pre is not None:
            out["pre"] = torch.from_numpy(pre)
        if self.tight_view is not None:
            post_t, mask_t, pre_t, _, scale_t, _, _ = self._view(i, self.tight_view[0], dx, dy)
            self._blank(post_t, mask_t, pre_t, self.tight_view[1], scale_t)
            out.update(post_t=torch.from_numpy(post_t), mask_t=torch.from_numpy(mask_t))
            if pre_t is not None:
                out["pre_t"] = torch.from_numpy(pre_t)
        return out
