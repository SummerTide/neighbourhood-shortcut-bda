#!/usr/bin/env python
"""Per-building evaluation of full-tile damage models (tile_models.py) on the xBD official test split.

standard        every labelled test building: footprint-mean probability over the
                four damage levels, argmax restricted to them; macro-F1, QWK and
                per-class F1, plus the neighbourhood strata of analysis/discordant.py
counterfactual  target buildings from four groups (isolated damage, concordant damage,
                isolated intact, concordant intact): the building, with a MARGIN px
                rim, pasted (pre and post) at a building of an all-intact or an
                all-major/destroyed neighbourhood of another tile, or inpainted in
                its own tile; reports the share predicted damaged per variant.
                --blend poisson pastes with Poisson blending (cv2.seamlessClone, as
                xFBD does) instead of a hard cut, so that a model trained with hard
                context swaps cannot recognise the paste by its seam; seamlessClone also
                shifts the colours of the building itself towards the new surroundings
                --blend poissonkeep blends the same way and then puts the building's own
                pixels (the footprint, without the rim) back, so only the rim is blended
                --placement vacant puts the building on open ground instead of on a donor
                building: a point of another test tile where the pasted building with its
                rim overlaps no footprint (any label, 2 px clearance) and whose labelled
                buildings within 64 px are all intact or all major/destroyed (at least 4),
                so every building of the receiving tile and its label stay intact
Usage: python harness.py --model changeos_r101 --out results/harness/changeos_r101 [--targets 500]
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
import tile_models  # noqa: E402
from bda_data import IMG, ImageSource, load_tables, polygons  # noqa: E402
from bda_metrics import building_metrics  # noqa: E402
from discordant import categories  # noqa: E402

MARGIN = 4
LEVELS = ["no", "minor", "major", "destroyed"]
VACANT_STEP, VACANT_MIN_NBRS, VACANT_CLEAR, VACANT_TRIES = 8, 4, 2, 400


def raster(polys, shift=(0, 0)):
    """Building index + 1 per pixel (0 = none) for a list of multi-part polygons."""
    canvas = np.zeros((IMG, IMG), np.int32)
    for k, parts in enumerate(polys):
        for p in parts:
            cv2.fillPoly(canvas, [np.round(p + np.asarray(shift)).astype(np.int32)], k + 1)
    return canvas


def footprint_probs(prob, ids, n):
    flat = ids.ravel()
    cnt = np.bincount(flat, minlength=n + 1)[1:].clip(min=1)
    sums = np.stack([np.bincount(flat, weights=prob[c].ravel(), minlength=n + 1)[1:] for c in range(prob.shape[0])], 1)
    return sums / cnt[:, None]


def run(model, pre, post, te, i):
    """Models that set needs_meta also get the scene's disaster name and type."""
    if getattr(model, "needs_meta", False):
        return model(pre, post, disaster=te.disaster[i], disaster_type=te.disaster_type[i])
    return model(pre, post)


def score(model, pre, post, polys, te, i, shift=(0, 0)):
    """(n, 4) probabilities of the buildings polys, displaced by shift, in a pre/post pair; te row
    i gives the scene metadata. Pixel models are averaged over each footprint; building models
    (needs_footprints, e.g. crop classifiers) are given the footprints."""
    if getattr(model, "needs_footprints", False):
        d = np.asarray(shift, np.float32)
        return model.predict_buildings(pre, post, [[p + d for p in parts] for parts in polys])
    return footprint_probs(run(model, pre, post, te, i), raster(polys, shift), len(polys))


def standard(model, te, polys, pre_src, post_src):
    probs = None
    for k, (sid, idx) in enumerate(te.groupby("scene_id").indices.items()):
        p = score(model, pre_src[int(sid)], post_src[int(sid)], [polys[i] for i in idx], te, idx[0])
        if probs is None:  # 4 levels for xBD models, 3 for DisasterM3 ones
            probs = np.zeros((len(te), p.shape[1]), np.float32)
        probs[idx] = p
        if (k + 1) % 200 == 0:
            print(f"  standard: {k + 1} tiles", flush=True)
    return probs


def standard_table(te, probs, cat):
    """Overall metrics plus, per neighbourhood stratum, accuracy and the share of each
    damage level predicted no-damage. xFBD tables (prep_xfbd.py) add the pasted building
    and the other buildings of each sample as two more rows."""
    y, pred = te.label.to_numpy(), probs.argmax(1)
    m = building_metrics(y, probs, te.disaster.to_numpy())["overall"]
    rows = [{"stratum": "all", "n": len(y), **m}]
    strata = [(c, cat == c) for c in pd.unique(cat)]
    if "pasted" in te:  # xFBD: the pasted building, and the others near it (64 px) or farther
        pasted, near = te.pasted.to_numpy(bool), te.dist_pasted.to_numpy() <= 64
        strata += [("xfbd pasted", pasted), ("xfbd others near", ~pasted & near), ("xfbd others far", ~pasted & ~near)]
    for c, s in strata:
        rows.append({"stratum": c, "n": int(s.sum()), "acc": float((pred[s] == y[s]).mean()),
                     **{f"missed_{LEVELS[l]}": float((pred[s & (y == l)] == 0).mean()) for l in (1, 2, 3)
                        if (s & (y == l)).any()}})
    return pd.DataFrame(rows), m


def test_buildings(data, split="test"):
    """The labelled buildings of a split (official xBD test by default) in harness order,
    with the scene's disaster type."""
    scenes, b = load_tables(data)
    b = b.merge(scenes[["scene_id", "disaster_type"]], on="scene_id")
    return scenes, b[(b.official == split) & (b.label >= 0)].reset_index(drop=True)


def paste(dst, src, keep, dx, dy, blend, core=None):
    """src pixels under keep, moved by (dx, dy), into a copy of dst. poissonkeep: Poisson blending, then the
    src pixels under core (the footprint) are put back unchanged."""
    out = np.array(dst)
    if blend in ("poisson", "poissonkeep"):
        x, y, w, h = cv2.boundingRect(keep.astype(np.uint8))
        try:  # seamlessClone centres the mask's bounding box on the given point
            out = cv2.seamlessClone(np.ascontiguousarray(src), out, keep.astype(np.uint8) * 255,
                                    (x + dx + w // 2, y + dy + h // 2), cv2.NORMAL_CLONE)
            if blend == "poissonkeep":
                ys, xs = np.nonzero(core)
                ok = (ys + dy >= 0) & (ys + dy < IMG) & (xs + dx >= 0) & (xs + dx < IMG)
                out[ys[ok] + dy, xs[ok] + dx] = src[ys[ok], xs[ok]]
            return out, True
        except cv2.error:
            pass  # too close to the border: hard paste
    ys, xs = np.nonzero(keep)
    ok = (ys + dy >= 0) & (ys + dy < IMG) & (xs + dx >= 0) & (xs + dx < IMG)
    out[ys[ok] + dy, xs[ok] + dx] = src[ys[ok], xs[ok]]
    return out, blend not in ("poisson", "poissonkeep")


class VacantSites:
    """Open-ground points of the test tiles, per context, for --placement vacant. The table is
    cached in <data>/vacant_sites_<split>.npz on first use and read from there afterwards, so that
    every run (in any environment: OpenCV versions rasterise footprint edges slightly differently)
    places each target at the same site."""

    def __init__(self, data, te, split):
        cache = os.path.join(data, f"vacant_sites_{split}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            self.occ = {int(s_): np.unpackbits(o)[:IMG * IMG].reshape(IMG, IMG).astype(bool) for s_, o in zip(z["occ_scenes"], z["occ"])}
            self.pool = {"intact context": [tuple(map(int, r)) for r in z["intact"]],
                         "damaged context": [tuple(map(int, r)) for r in z["damaged"]]}
            self.meta_row = {int(s_): int(r) for s_, r in z["meta_row"]}
            print(f"  vacant sites from {cache}: intact {len(self.pool['intact context'])}, "
                  f"damaged {len(self.pool['damaged context'])}", flush=True)
            return
        self._build(data, te)
        sc = sorted(self.occ)
        np.savez_compressed(cache, occ_scenes=np.array(sc), occ=np.stack([np.packbits(self.occ[k]) for k in sc]),
                            intact=np.array(self.pool["intact context"]), damaged=np.array(self.pool["damaged context"]),
                            meta_row=np.array(sorted(self.meta_row.items())))

    def _build(self, data, te):
        _, allb = load_tables(data)
        allb = allb[allb.scene_id.isin(set(te.scene_id))]
        y, cx, cy, sid = te.label.to_numpy(), te.cx.to_numpy(), te.cy.to_numpy(), te.scene_id.to_numpy()
        self.occ, self.pool, self.meta_row = {}, {"intact context": [], "damaged context": []}, {}
        g = np.arange(VACANT_STEP // 2, IMG, VACANT_STEP)
        gx, gy = np.meshgrid(g, g)
        gx, gy = gx.ravel(), gy.ravel()
        kernel = np.ones((2 * VACANT_CLEAR + 1, 2 * VACANT_CLEAR + 1), np.uint8)
        for s_, idx in pd.Series(np.arange(len(te))).groupby(sid).indices.items():
            self.meta_row[int(s_)] = int(idx[0])
            occ = (raster(polygons(allb[allb.scene_id == s_].wkt_pre)) > 0).astype(np.uint8)
            occ = cv2.dilate(occ, kernel) > 0
            free = ~occ[gy, gx]
            d = np.hypot(gx[:, None] - cx[idx][None], gy[:, None] - cy[idx][None]) < 64
            n = d.sum(1)
            intact = (d & (y[idx][None] == 0)).sum(1)
            heavy = (d & (y[idx][None] >= 2)).sum(1)
            ok_i = free & (n >= VACANT_MIN_NBRS) & (intact == n)
            ok_d = free & (n >= VACANT_MIN_NBRS) & (heavy == n)
            if ok_i.any() or ok_d.any():
                self.occ[int(s_)] = occ
            self.pool["intact context"] += [(int(s_), int(a), int(b)) for a, b in zip(gx[ok_i], gy[ok_i])]
            self.pool["damaged context"] += [(int(s_), int(a), int(b)) for a, b in zip(gx[ok_d], gy[ok_d])]
        print(f"  vacant sites: intact {len(self.pool['intact context'])}, damaged {len(self.pool['damaged context'])}", flush=True)

    def pick(self, rng, variant, ys, xs, tcx, tcy, own_scene):
        """A site where the kept pixels (ys, xs), centred on (tcx, tcy), land on open ground."""
        pool = self.pool[variant]
        for _ in range(VACANT_TRIES):
            s_, px, py = pool[int(rng.randint(len(pool)))]
            dx, dy = int(round(px - tcx)), int(round(py - tcy))
            yy, xx = ys + dy, xs + dx
            if s_ == own_scene or yy.min() < 0 or xx.min() < 0 or yy.max() >= IMG or xx.max() >= IMG:
                continue
            if not self.occ[s_][yy, xx].any():
                return s_, dx, dy
        return None


def counterfactual(model, te, polys, pre_src, post_src, cat, n_targets, rng, blend="hard", partial=None, sites=None,
                   only=None):
    """partial: CSV updated after every group; groups already in it are skipped on a rerun
    while their random draws are replayed, so every model sees the same targets and donors.
    only: run just these groups; the others are skipped the same way (draws replayed, nothing scored)."""
    y = te.label.to_numpy()
    groups = {"isolated damage": np.flatnonzero(cat == "isolated damage"),
              "concordant damage": np.flatnonzero((cat == "concordant") & (y >= 1)),
              "isolated intact": np.flatnonzero(cat == "isolated intact"),
              "concordant intact": np.flatnonzero((cat == "concordant") & (y == 0))}
    groups = {g: rng.permutation(idx)[:n_targets] for g, idx in groups.items()}
    cx, cy = te.cx.to_numpy(), te.cy.to_numpy()
    half = np.maximum(te.maxx - te.minx, te.maxy - te.miny).to_numpy() / 2 + MARGIN + 2
    donors = {"intact context": np.flatnonzero((cat == "concordant") & (y == 0)),
              "damaged context": np.flatnonzero((cat == "concordant") & (y >= 2))}
    kernel = np.ones((2 * MARGIN + 1, 2 * MARGIN + 1), np.uint8)
    rows, blended, per_target = [], [], []
    done = pd.read_csv(partial) if partial and os.path.exists(partial) else pd.DataFrame(columns=["targets"])
    # keep the per-target logs of the groups already done, so a resumed run does not overwrite them
    tlog = partial.replace(".partial.csv", "_targets.csv") if partial else None
    if tlog and os.path.exists(tlog) and len(done):
        per_target = pd.read_csv(tlog).query("`group` in @done.targets.tolist()").to_dict("records")
    for gname, idx in groups.items():
        hits = {v: [] for v in ("intact context", "damaged context", "building gone")}
        skipped = only is not None and gname not in only
        if gname in set(done.targets) or skipped:
            for t in idx:  # replay the donor draws only
                for v, pool in donors.items():
                    if sites is not None:
                        keep = cv2.dilate((raster([polys[t]]) > 0).astype(np.uint8), kernel) > 0
                        ys, xs = np.nonzero(keep)
                        sites.pick(rng, v, ys, xs, cx[t], cy[t], int(te.scene_id[t]))
                        continue
                    for _ in range(20):
                        d = int(rng.choice(pool))
                        if half[t] <= cx[d] <= IMG - half[t] and half[t] <= cy[d] <= IMG - half[t] and d != t:
                            break
            if skipped and gname not in set(done.targets):
                print(f"  counterfactual {gname}: skipped (draws replayed)", flush=True)
                continue
            rows.append(done[done.targets == gname].iloc[0].to_dict())
            print(f"  counterfactual {gname}: from {partial}", flush=True)
            continue
        for t in idx:
            sid = int(te.scene_id[t])
            pre_t, post_t = np.ascontiguousarray(pre_src[sid]), np.ascontiguousarray(post_src[sid])
            core = raster([polys[t]]) > 0
            keep = cv2.dilate(core.astype(np.uint8), kernel) > 0
            for v, pool in donors.items():
                if sites is not None:  # open ground of another tile: no footprint is covered
                    ys, xs = np.nonzero(keep)
                    site = sites.pick(rng, v, ys, xs, cx[t], cy[t], sid)
                    if site is None:
                        continue
                    sd, dx, dy = site
                    d = sites.meta_row[sd]
                else:
                    for _ in range(20):  # a donor far enough from the border for the pasted building
                        d = int(rng.choice(pool))
                        if half[t] <= cx[d] <= IMG - half[t] and half[t] <= cy[d] <= IMG - half[t] and d != t:
                            break
                    dx, dy = int(round(cx[d] - cx[t])), int(round(cy[d] - cy[t]))
                    sd = int(te.scene_id[d])
                pre_c, ok_pre = paste(pre_src[sd], pre_t, keep, dx, dy, blend, core)
                post_c, ok_post = paste(post_src[sd], post_t, keep, dx, dy, blend, core)
                blended.append(ok_pre and ok_post)
                p = score(model, pre_c, post_c, [polys[t]], te, d, (dx, dy))[0]
                hits[v].append(p.argmax() >= 1)
                per_target.append({"building_id": int(te.building_id[t]), "group": gname, "variant": v,
                                   "donor_building_id": -1 if sites is not None else int(te.building_id[d]),
                                   "donor_scene": sd, "dx": dx, "dy": dy, "pred": int(p.argmax()),
                                   "p_damaged": float(1 - p[0] / max(p.sum(), 1e-9))})
            m8 = keep.astype(np.uint8)
            gone = score(model, cv2.inpaint(pre_t, m8, 7, cv2.INPAINT_TELEA), cv2.inpaint(post_t, m8, 7, cv2.INPAINT_TELEA),
                         [polys[t]], te, t)[0]
            hits["building gone"].append(gone.argmax() >= 1)
            per_target.append({"building_id": int(te.building_id[t]), "group": gname, "variant": "building gone",
                               "donor_building_id": -1, "pred": int(gone.argmax()),
                               "p_damaged": float(1 - gone[0] / max(gone.sum(), 1e-9))})
        rows.append({"targets": gname, "n": len(idx), **{v: float(np.mean(h)) for v, h in hits.items()}})
        print(f"  counterfactual {gname}: done", flush=True)
        if partial:
            pd.DataFrame(rows).to_csv(partial, index=False)
            # every target and variant of the groups run here, for breakdowns (e.g. by disaster type)
            pd.DataFrame(per_target).to_csv(partial.replace(".partial.csv", "_targets.csv"), index=False)
    if blend in ("poisson", "poissonkeep"):
        print(f"  pastes blended as requested: {np.mean(blended):.3f} (the rest hard, at tile borders)", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=tile_models.available())
    ap.add_argument("--data", default="data")
    ap.add_argument("--image-dir", default=None)
    ap.add_argument("--split", default="test", help="scenes.csv 'official' value to score (xfbd_test for prep_xfbd.py)")
    ap.add_argument("--max-dist-pasted", type=float, default=None,
                    help="xFBD: score only buildings within this distance (px) of the pasted one")
    ap.add_argument("--out", required=True)
    ap.add_argument("--targets", type=int, default=500, help="counterfactual targets per group (0 = skip)")
    ap.add_argument("--blend", choices=["hard", "poisson", "poissonkeep"], default="hard", help="counterfactual paste")
    ap.add_argument("--placement", choices=["building", "vacant"], default="building",
                    help="counterfactual site: on a donor building (default) or on open ground (no footprint covered)")
    ap.add_argument("--cf-only", action="store_true", help="skip the standard pass (counterfactual only)")
    ap.add_argument("--cf-groups", default=None,
                    help="':'-separated groups to run, '_' for spaces (e.g. concordant_damage:concordant_intact)")
    ap.add_argument("--average", nargs="+", default=None,
                    help="score the mean of these runs' harness_preds.npz instead of running --model")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    scenes, te = test_buildings(args.data, args.split)
    if args.max_dist_pasted is not None:
        te = te[te.dist_pasted <= args.max_dist_pasted].reset_index(drop=True)
    polys = polygons(te.wkt_pre)
    img_dir = args.image_dir or args.data
    pre_src, post_src = (ImageSource(os.path.join(img_dir, f"{k}.u8"), len(scenes)) for k in ("pre", "post"))
    t0 = time.time()
    if args.average:
        runs = [np.load(os.path.join(r, "harness_preds.npz")) for r in args.average]
        assert all((r["building_id"] == te.building_id.to_numpy()).all() for r in runs)
        probs, model = np.mean([r["probs"] for r in runs], 0), None
        args.model, args.targets = "average of " + ", ".join(args.average), 0
    else:
        model = tile_models.build(args.model)
    cat = categories(te, 64)
    if not args.cf_only:
        probs = probs if args.average else standard(model, te, polys, pre_src, post_src)
        np.savez_compressed(os.path.join(args.out, "harness_preds.npz"), building_id=te.building_id.to_numpy(),
                            label=te.label.to_numpy(), probs=probs)
        std, m = standard_table(te, probs, cat)
        std.to_csv(os.path.join(args.out, "harness_standard.csv"), index=False)
        print(f"{args.model}: macro-F1 {m['macro_f1']:.3f} QWK {m['qwk']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        print(std[[c for c in ["stratum", "n", "acc", "missed_minor", "missed_major", "missed_destroyed"] if c in std]]
              .round(3).to_string(index=False), flush=True)
    if args.targets:
        suffix = ("" if args.blend == "hard" else f"_{args.blend}") + ("" if args.placement == "building" else "_vacant")
        sites = VacantSites(args.data, te, args.split) if args.placement == "vacant" else None
        cf = counterfactual(model, te, polys, pre_src, post_src, cat, args.targets, np.random.RandomState(0),
                            args.blend, os.path.join(args.out, f"harness_counterfactual{suffix}.partial.csv"), sites,
                            only=None if not args.cf_groups else [g.replace("_", " ") for g in args.cf_groups.split(":")])
        cf.to_csv(os.path.join(args.out, f"harness_counterfactual{suffix}.csv"), index=False)
        print(cf.round(3).to_string(index=False), flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
