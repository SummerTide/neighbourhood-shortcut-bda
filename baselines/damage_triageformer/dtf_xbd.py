#!/usr/bin/env python
"""Damage-TriageFormer crop mode retrained on xBD as a four-level building damage baseline.

Model and recipe are the repo's primary crop configuration (Xiao et al., arXiv 2606.12248, revised
manuscript; run crop_dtf = scripts/train_crop.py --arch dinov3_vitl16 --gated, reproduced by
scripts/train_crop_r2.py --variant gated_aux --recipe crop --crop-mode upsampled). The repo's
DinoCrop, make_crop, macro_f1 and predict are imported from train_crop_r2.py unchanged; this file
holds the xBD adapter, the repo's training loop for that configuration, and the project's evaluation.

  prepare  the repo's --prepare stage: one JPEG quality-90 crop per labelled building, the pixel
           bbox of its pre-event footprint + 32 px, clipped to the tile, resized to 224 x 224 with
           INTER_AREA (post-event image only: crop mode is mono-temporal). Splits: train = official
           train (train + tier3) with area >= 30 px (the repo's MIN_AREA), val = xBD hold, test =
           official test; val and test keep every labelled building, test in harness.py order.
  train    DINOv3 ViT-L/16 (LVD-1689M, as the repo), class token -> gate (damaged?) + 3-way leaf
           (minor / major / destroyed) + aux severity, all parameters fine-tuned; loss = BCE(gate)
           + 2.0 CE(leaf on damaged buildings; inverse-sqrt-frequency weights, label smoothing 0.1)
           + 0.5 smooth-L1(aux); AdamW lr 3e-5, weight decay 1e-4, cosine over 20 epochs stepped
           per epoch, batch 48, random h/v flips, seed 42; checkpoint = best val macro-F1.
           Probabilities are [1 - g, g * softmax(leaf)].
  eval     test probabilities -> harness_preds.npz and harness_standard.csv (as harness.py), and
           the crop-level counterfactual context test of analysis/context_swap.py on this model's
           crops -> harness_counterfactual.csv.

Changes made for xBD (everything else is the repo's code and settings):
  - 4 ordinal classes instead of the 5 typology classes: the leaf is 3-way and the aux severity
    targets are 0, 1/3, 2/3, 1 (the repo uses 0, .3, .7, .5, 1 for its typology).
  - The footprint is the xBD pre-event polygon; the floor/ceil of its bounds (as
    scripts/predict_event.py computes it) replaces the connected-component bbox of the
    repo's instance index.
  - The same DINOv3 weights are loaded from the official checkpoint through the reference dinov3
    code instead of the Hugging Face copy (offline compute nodes, gated HF repo). The class token is
    the final-norm CLS, as HF last_hidden_state[:, 0].
  - Matmuls run in TF32 (--precision tf32; the repo used strict fp32), which is about 3x faster on A100.
  - Crops are packed in one file per split instead of one JPEG file per crop. The bytes are the same.
--weights sat swaps in the satellite DINOv3 ViT-L/16 (SAT-493M) with its own normalisation.

  python dtf_xbd.py prepare
  python dtf_xbd.py train --run runs/dtf_lvd [--crops /dev/shm/...]
  python dtf_xbd.py eval --run runs/dtf_lvd --out ../../footprint_bda/results/harness/damage_triageformer
"""
import argparse
import contextlib
import json
import multiprocessing as mp
import os
import sys
import time
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

PROJ = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))  # repository root
REPO = os.path.join(PROJ, "external/Damage-TriageFormer")
DINOV3_REPO = os.path.join(PROJ, "external/dinov3")
CKPT_DIR = os.path.join(PROJ, "external/checkpoints")
HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (DINOV3_REPO, os.path.join(PROJ, "analysis"), PROJ, os.path.join(REPO, "scripts")):
    sys.path.insert(0, _p)
import train_crop_r2 as dtf  # noqa: E402  the repo's crop mode
from bda_data import IMG, load_tables, polygons  # noqa: E402
from bda_metrics import building_metrics  # noqa: E402
from discordant import categories  # noqa: E402

cv2.setNumThreads(1)  # one thread per process (data workers, prepare pool)
N_CLASSES = 4
dtf.NUM_CLASSES = N_CLASSES  # read by DinoCrop at construction: gate + (N_CLASSES - 1)-way leaf
SEV_TARGETS = [0.0, 1 / 3, 2 / 3, 1.0]
LEVELS = ["no", "minor", "major", "destroyed"]
HF_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
WEIGHTS = {"lvd": "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth", "sat": "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"}
NORM = {"lvd": (dtf.MEAN, dtf.STD),  # ImageNet statistics, as the repo
        "sat": (np.array([0.430, 0.411, 0.296], np.float32), np.array([0.213, 0.156, 0.143], np.float32))}
CF_MARGIN = 4  # px around the footprint that stays with the building (analysis/context_swap.py)


# ---------------------------------------------------------------- crops

def pixel_boxes(df):
    """Pixel bbox (x0, y0, x1, y1; x1 and y1 exclusive) of each pre-event footprint."""
    x0 = np.floor(df.minx.to_numpy() + 1e-6)
    y0 = np.floor(df.miny.to_numpy() + 1e-6)
    x1 = np.maximum(np.ceil(df.maxx.to_numpy() - 1e-6), x0 + 1)
    y1 = np.maximum(np.ceil(df.maxy.to_numpy() - 1e-6), y0 + 1)
    return np.stack([x0, y0, x1, y1], 1).astype(np.int64)


def crop_window(box):
    """The tile window make_crop resizes: bbox + MARGIN px, clipped to the tile."""
    x0, y0, x1, y1 = (int(v) for v in box)
    return max(0, x0 - dtf.MARGIN), max(0, y0 - dtf.MARGIN), min(IMG, x1 + dtf.MARGIN), min(IMG, y1 + dtf.MARGIN)


def dtf_crop(tile, box):
    """The repo's crop of one building (make_crop, upsampled mode), in the tile's channel order."""
    x0, y0, x1, y1 = (int(v) for v in box)
    return dtf.make_crop(tile, {"x0": x0, "y0": y0, "x1": x1, "y1": y1}, "upsampled")


def jpeg(rgb):
    """JPEG quality-90 bytes of an RGB crop, encoded from BGR as the repo writes its crops."""
    ok, buf = cv2.imencode(".jpg", np.ascontiguousarray(rgb[:, :, ::-1]), [cv2.IMWRITE_JPEG_QUALITY, 90])
    assert ok
    return buf


def unjpeg(buf):
    """Decoded RGB crop, as the repo's CropDS reads it (cv2.imread then [:, :, ::-1])."""
    return cv2.imdecode(np.asarray(buf, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1]


def normalise(rgb, mean, std):
    """CropDS's model input: [0, 1] floats standardised per channel, CHW."""
    x = (rgb.astype(np.float32) / 255.0 - mean) / std
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def split_tables(b):
    lab = b[b.label >= 0]
    return {"train": lab[(lab.official == "train") & (lab.area >= dtf.MIN_AREA)].reset_index(drop=True),
            "val": lab[lab.official == "val"].reset_index(drop=True),
            "test": b[(b.official == "test") & (b.label >= 0)].reset_index(drop=True)}


def _prepare_scene(job):
    path, n_rows, sid, boxes = job
    tile = np.array(np.memmap(path, np.uint8, "r", shape=(n_rows, IMG, IMG, 3))[sid])
    return [jpeg(dtf_crop(tile, box)).tobytes() for box in boxes]


def prepare(args):
    scenes, b = load_tables(args.data)
    post = os.path.join(args.image_dir or args.data, "post.u8")
    os.makedirs(args.crops, exist_ok=True)
    for name, df in split_tables(b).items():
        if os.path.exists(os.path.join(args.crops, f"{name}.npz")):
            print(f"{name}: already prepared", flush=True)
            continue
        boxes = pixel_boxes(df)
        groups = list(df.groupby("scene_id").indices.items())
        jobs = [(post, len(scenes), int(s), boxes[idx]) for s, idx in groups]
        off, size = np.zeros(len(df), np.int64), np.zeros(len(df), np.int64)
        pos, t0 = 0, time.time()
        tmp = os.path.join(args.crops, f"{name}.bin.tmp")
        with open(tmp, "wb") as fh, mp.Pool(len(os.sched_getaffinity(0))) as pool:
            for k, ((_, idx), blobs) in enumerate(zip(groups, pool.imap(_prepare_scene, jobs, chunksize=4))):
                for i, blob in zip(idx, blobs):
                    fh.write(blob)
                    off[i], size[i] = pos, len(blob)
                    pos += len(blob)
                if (k + 1) % 2000 == 0:
                    print(f"  {name}: {k + 1}/{len(groups)} tiles ({time.time() - t0:.0f}s)", flush=True)
        os.replace(tmp, os.path.join(args.crops, f"{name}.bin"))
        np.savez(os.path.join(args.crops, f"{name}.npz"), building_id=df.building_id.to_numpy(),
                 label=df.label.to_numpy(), scene_id=df.scene_id.to_numpy(), box=boxes, offset=off, length=size)
        print(f"{name}: {len(df)} crops, {pos / 2 ** 30:.2f} GiB, labels {np.bincount(df.label, minlength=4).tolist()} "
              f"({time.time() - t0:.0f}s)", flush=True)


class PackedCrops(Dataset):
    """The repo's CropDS (random h/v flips in training, normalised CHW floats) reading the JPEG crops
    from one packed file per split instead of one file per crop."""

    def __init__(self, crops_dir, split, train, norm, limit=0):
        z = np.load(os.path.join(crops_dir, f"{split}.npz"))
        keep = slice(None) if not limit else np.sort(np.random.RandomState(0).permutation(len(z["label"]))[:limit])
        self.off, self.size = z["offset"][keep], z["length"][keep]
        self.labels, self.building_id = z["label"][keep].astype(np.int64), z["building_id"][keep]
        self.path, self.train, (self.mean, self.std), self.buf = os.path.join(crops_dir, f"{split}.bin"), train, norm, None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, k):
        if self.buf is None:  # opened lazily in each DataLoader worker
            self.buf = np.memmap(self.path, np.uint8, "r")
        img = unjpeg(self.buf[self.off[k]:self.off[k] + self.size[k]])
        if self.train:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1, :]
        return normalise(img, self.mean, self.std), int(self.labels[k])


# ---------------------------------------------------------------- model

class LocalDinoV3(nn.Module):
    """Stands in for transformers.AutoModel.from_pretrained(HF_NAME) inside the repo's DinoCrop: the
    official DINOv3 ViT-L/16 checkpoint run by the reference dinov3 code. The output has the HF layout
    (final-norm [CLS, 4 storage tokens, patch tokens]), so last_hidden_state[:, 0] is the class token."""

    def __init__(self, weights):
        super().__init__()
        from dinov3.hub import backbones
        net = backbones.dinov3_vitl16(pretrained=False, weights=WEIGHTS[weights])
        net.load_state_dict(torch.load(os.path.join(CKPT_DIR, WEIGHTS[weights]), map_location="cpu"), strict=True)
        self.net, self.config = net, SimpleNamespace(hidden_size=net.embed_dim)

    def forward(self, pixel_values):
        o = self.net.forward_features(pixel_values)
        tokens = torch.cat([o["x_norm_clstoken"][:, None], o["x_storage_tokens"], o["x_norm_patchtokens"]], 1)
        return SimpleNamespace(last_hidden_state=tokens)


def build_model(weights):
    """The repo's DinoCrop('gated_aux'), i.e. Damage-TriageFormer crop mode (train_crop.py --gated)."""
    from transformers import AutoModel

    def local(name, *a, **k):
        assert name == HF_NAME, name
        return LocalDinoV3(weights)

    with mock.patch.object(AutoModel, "from_pretrained", local):
        model = dtf.DinoCrop("gated_aux")
    return model.cuda()


def set_precision(precision):
    torch.backends.cuda.matmul.allow_tf32 = precision != "fp32"


def autocast(precision):
    return torch.autocast("cuda", dtype=torch.bfloat16) if precision == "bf16" else contextlib.nullcontext()


def gated_loss(model, x, y, leaf_crit, sev_t):
    """train_crop_r2.main for variant gated_aux, recipe crop (= train_crop.py --gated)."""
    g_logit, logits, aux_pred = model.heads(x)
    logits, g_logit = logits.float(), g_logit.float()
    loss = F.binary_cross_entropy_with_logits(g_logit, (y > 0).float())
    dmg = y > 0
    if dmg.any():
        loss = loss + 2.0 * leaf_crit(logits[dmg], y[dmg] - 1)
    return loss + 0.5 * F.smooth_l1_loss(aux_pred.float(), sev_t[y])


def n_workers(args):
    return args.workers or max(1, len(os.sched_getaffinity(0)) - 1)


# ---------------------------------------------------------------- training

def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    set_precision(args.precision)
    os.makedirs(args.run, exist_ok=True)
    norm, dev, workers = NORM[args.weights], "cuda", n_workers(args)
    tr = PackedCrops(args.crops, "train", True, norm, args.limit)
    va = PackedCrops(args.crops, "val", False, norm, args.limit)
    tl = DataLoader(tr, batch_size=args.batch, shuffle=True, num_workers=workers, pin_memory=True, drop_last=True,
                    persistent_workers=True)
    vl = DataLoader(va, batch_size=args.eval_batch, shuffle=False, num_workers=workers, pin_memory=True)
    print(f"train {len(tr)}, val {len(va)} crops; {workers} workers; {torch.cuda.get_device_name()}", flush=True)
    model = build_model(args.weights)

    cnt = np.bincount(tr.labels, minlength=N_CLASSES)
    w = 1.0 / np.sqrt(np.maximum(cnt, 1))
    w = w / w.sum() * N_CLASSES
    leaf_w = w[1:] / w[1:].sum() * (N_CLASSES - 1)
    print("class counts:", cnt.tolist(), " weights:", np.round(w, 3).tolist(), " leaf:", np.round(leaf_w, 3).tolist())
    leaf_crit = nn.CrossEntropyLoss(weight=torch.tensor(leaf_w, dtype=torch.float32, device=dev), label_smoothing=0.1)
    sev_t = torch.tensor(SEV_TARGETS, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    config = {"model": "Damage-TriageFormer crop mode (train_crop_r2.DinoCrop gated_aux, recipe crop, upsampled crops)",
              "backbone": f"DINOv3 ViT-L/16 {WEIGHTS[args.weights]}", "weights": args.weights,
              "classes": LEVELS, "sev_targets": SEV_TARGETS, "epochs": args.epochs, "batch": args.batch, "lr": args.lr,
              "weight_decay": 1e-4, "label_smoothing": 0.1, "leaf_loss_weight": 2.0, "aux_loss_weight": 0.5,
              "schedule": "CosineAnnealingLR(T_max=epochs), stepped per epoch", "precision": args.precision,
              "seed": args.seed, "train_n": len(tr), "val_n": len(va), "class_counts": cnt.tolist(),
              "crop": "pre-event footprint bbox + 32 px, clipped, resized to 224x224 INTER_AREA, JPEG q90, post image",
              "selection": "best val (xBD hold) macro F1 over epochs", "limit": args.limit}
    cfg_path = os.path.join(args.run, "config.json")
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        diff = {k: (old.get(k), v) for k, v in config.items() if old.get(k) != v}
        assert not diff, f"run directory holds another configuration: {diff}"
    json.dump(config, open(cfg_path, "w"), indent=2)

    start, best, best_ep, hist = 0, -1.0, -1, []
    last = os.path.join(args.run, "last.pth")
    if os.path.exists(last):
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        start, best, best_ep, hist = ck["epoch"], ck["best"], ck["best_ep"], ck["hist"]
        torch.set_rng_state(ck["rng_torch"])
        torch.cuda.set_rng_state(ck["rng_cuda"])
        np.random.set_state(ck["rng_np"])
        del ck
        print(f"resumed after epoch {start} (best val macro-F1 {best:.4f} at epoch {best_ep})", flush=True)

    for ep in range(start, args.epochs):
        if args.stop_after and ep >= start + args.stop_after:
            print(f"stopping after {args.stop_after} epoch(s); resume with the same command", flush=True)
            return
        model.train()
        t0, tot, n = time.time(), torch.zeros((), device=dev), 0
        for it, (x, y) in enumerate(tl):
            opt.zero_grad(set_to_none=True)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with autocast(args.precision):
                loss = gated_loss(model, x, y, leaf_crit, sev_t)
            loss.backward()
            opt.step()
            tot += loss.detach() * len(y)
            n += len(y)
            if (it + 1) % args.log_every == 0:
                print(f"  ep {ep + 1} it {it + 1}/{len(tl)} loss {tot.item() / n:.4f} "
                      f"{n / (time.time() - t0):.0f} img/s", flush=True)
            if args.max_iters and it + 1 >= args.max_iters:
                break
        lr_used = opt.param_groups[0]["lr"]
        sched.step()
        t_train = time.time() - t0
        probs = dtf.predict(model, vl, dev)
        vf1, per = dtf.macro_f1(va.labels, probs.argmax(1), N_CLASSES)
        m = building_metrics(va.labels, probs)["overall"]
        hist.append({"epoch": ep + 1, "loss": tot.item() / max(n, 1), "lr": lr_used,
                     "val_macro_f1": vf1, **{f"val_f1_{c}": f for c, f in enumerate(per)}, "val_qwk": m["qwk"],
                     "train_s": t_train, "val_s": time.time() - t0 - t_train})
        print(f"ep {ep + 1}/{args.epochs} loss {tot.item() / max(n, 1):.4f} val macroF1 {vf1:.4f} "
              f"(F1 {' '.join(f'{f:.3f}' for f in per)}) QWK {m['qwk']:.4f} "
              f"[train {t_train:.0f}s, {n / t_train:.0f} img/s; val {hist[-1]['val_s']:.0f}s]", flush=True)
        if vf1 > best:
            best, best_ep = vf1, ep + 1
            torch.save(model.state_dict(), os.path.join(args.run, "best.pth"))
            np.savez_compressed(os.path.join(args.run, "val_preds.npz"), building_id=va.building_id, label=va.labels,
                                probs=probs, epoch=ep + 1)
        pd.DataFrame(hist).to_csv(os.path.join(args.run, "val_log.csv"), index=False)
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep + 1,
                    "best": best, "best_ep": best_ep, "hist": hist, "rng_torch": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(), "rng_np": np.random.get_state()}, last + ".tmp")
        os.replace(last + ".tmp", last)
    json.dump({**config, "best_epoch": best_ep, "best_val_macro_f1": best, "history": hist},
              open(os.path.join(args.run, "results.json"), "w"), indent=2)
    print(f"training done: best val macro-F1 {best:.4f} at epoch {best_ep}", flush=True)


# ---------------------------------------------------------------- evaluation

def counterfactual(model, te, cat, post_path, n_scenes, norm, n_concordant=1500, limit=0, batch=128):
    """analysis/context_swap.py on this model's crops. Targets and donors are drawn exactly as there
    (RandomState(0), the same calls in the same order). Every variant is a crop of the repo's geometry:
      original         the target's crop
      intact context   the footprint dilated by CF_MARGIN px (in tile pixels) kept, the rest of the crop
                       from the crop of a building in an all-intact neighbourhood
      damaged context  the same with a building in an all-major/destroyed neighbourhood
      building gone    the footprint and its margin inpainted (Telea, radius 7) in the target's crop
    Each composite then goes through the JPEG round trip and normalisation of every other crop.
    Returns the share predicted damaged (argmax >= minor) per group and variant, and the original
    probabilities of the targets."""
    rng = np.random.RandomState(0)
    y = te.label.to_numpy()
    groups = {"isolated damage": np.flatnonzero(cat == "isolated damage"),
              "concordant damage": rng.permutation(np.flatnonzero((cat == "concordant") & (y >= 1)))[:n_concordant],
              "isolated intact": np.flatnonzero(cat == "isolated intact"),
              "concordant intact": rng.permutation(np.flatnonzero((cat == "concordant") & (y == 0)))[:n_concordant]}
    if limit:
        groups = {g: idx[:limit] for g, idx in groups.items()}
    donors = {"intact context": np.flatnonzero((cat == "concordant") & (y == 0)),
              "damaged context": np.flatnonzero((cat == "concordant") & (y >= 2))}
    boxes, sid = pixel_boxes(te), te.scene_id.to_numpy()
    mm = np.memmap(post_path, np.uint8, "r", shape=(n_scenes, IMG, IMG, 3))
    need = np.unique(np.concatenate(list(groups.values())))
    polys = dict(zip(need, polygons(te.wkt_pre.to_numpy()[need])))
    size = dtf.SIZE

    def target(i):
        crop = dtf_crop(np.asarray(mm[sid[i]]), boxes[i])
        xa, ya, xb, yb = crop_window(boxes[i])
        sx, sy = size / (xb - xa), size / (yb - ya)
        mask = np.zeros((size, size), np.uint8)
        for p in polys[i]:
            cv2.fillPoly(mask, [np.round((p - [xa, ya]) * [sx, sy]).astype(np.int32)], 1)
        kernel = np.ones((2 * int(round(CF_MARGIN * sy)) + 1, 2 * int(round(CF_MARGIN * sx)) + 1), np.uint8)
        return crop, cv2.dilate(mask, kernel) > 0

    @torch.no_grad()
    def score(crops):
        x = torch.stack([normalise(unjpeg(jpeg(c)), *norm) for c in crops]).cuda()
        return model(x).float().cpu().numpy()

    model.eval()
    rows, originals = [], {}
    variants = ["original", "intact context", "damaged context", "building gone"]
    for gname, idx in groups.items():
        out = {v: [] for v in variants}
        for s in range(0, len(idx), batch):
            items = {v: [] for v in variants}
            for i in idx[s:s + batch]:
                crop, keep = target(int(i))
                items["original"].append(crop)
                for v in ("intact context", "damaged context"):
                    d = int(rng.choice(donors[v]))
                    items[v].append(np.where(keep[..., None], crop, dtf_crop(np.asarray(mm[sid[d]]), boxes[d])))
                items["building gone"].append(cv2.inpaint(np.ascontiguousarray(crop), keep.astype(np.uint8), 7,
                                                          cv2.INPAINT_TELEA))
            for v in variants:
                out[v].append(score(items[v]))
        rows.append({"targets": gname, "n": len(idx),
                     **{v: float((np.concatenate(ps).argmax(1) >= 1).mean()) for v, ps in out.items()}})
        originals[gname] = (idx, np.concatenate(out["original"]))
        print(f"  counterfactual {gname}: done", flush=True)
    return pd.DataFrame(rows), originals


def evaluate(args):
    cfg = json.load(open(os.path.join(args.run, "config.json")))
    set_precision(cfg["precision"])
    norm = NORM[cfg["weights"]]
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    scenes, b = load_tables(args.data)
    b = b.merge(scenes[["scene_id", "disaster_type"]], on="scene_id")  # as harness.py
    te = b[(b.official == "test") & (b.label >= 0)].reset_index(drop=True)
    ds = PackedCrops(args.crops, "test", False, norm)
    assert (ds.building_id == te.building_id.to_numpy()).all(), "test crops are not in harness order"
    sel = np.arange(len(te)) if not args.limit else np.sort(np.random.RandomState(0).permutation(len(te))[:args.limit])
    model = build_model(cfg["weights"])
    model.load_state_dict(torch.load(os.path.join(args.run, "best.pth"), map_location="cuda"))
    loader = DataLoader(Subset(ds, sel), batch_size=args.eval_batch, shuffle=False, num_workers=n_workers(args),
                        pin_memory=True)
    probs = dtf.predict(model, loader, "cuda").astype(np.float32)
    cat_all = categories(te, 64)
    sub, y, cat = te.iloc[sel].reset_index(drop=True), te.label.to_numpy()[sel], cat_all[sel]
    np.savez_compressed(os.path.join(args.out, "harness_preds.npz"), building_id=sub.building_id.to_numpy(), label=y,
                        probs=probs)
    # the standard table of harness.py
    pred = probs.argmax(1)
    m = building_metrics(y, probs, sub.disaster.to_numpy())["overall"]
    rows = [{"stratum": "all", "n": len(y), **m}]
    for c in pd.unique(cat):
        s = cat == c
        rows.append({"stratum": c, "n": int(s.sum()), "acc": float((pred[s] == y[s]).mean()),
                     **{f"missed_{LEVELS[l]}": float((pred[s & (y == l)] == 0).mean()) for l in (1, 2, 3)
                        if (s & (y == l)).any()}})
    std = pd.DataFrame(rows)
    std.to_csv(os.path.join(args.out, "harness_standard.csv"), index=False)
    print(f"damage_triageformer ({args.run}): macro-F1 {m['macro_f1']:.3f} QWK {m['qwk']:.3f} "
          f"F1 {m['f1_0']:.3f}/{m['f1_1']:.3f}/{m['f1_2']:.3f}/{m['f1_3']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    print(std[["stratum", "n", "acc", "missed_minor", "missed_major", "missed_destroyed"]].round(3).to_string(index=False),
          flush=True)
    if args.targets:
        post = os.path.join(args.image_dir or args.data, "post.u8")
        cf, originals = counterfactual(model, te, cat_all, post, len(scenes), norm, args.targets, args.cf_limit)
        cf.to_csv(os.path.join(args.out, "harness_counterfactual.csv"), index=False)
        print(cf.round(3).to_string(index=False), flush=True)
        if not args.limit:  # the "original" variant must reproduce the standard predictions
            for g, (idx, p) in originals.items():
                agree = (p.argmax(1) == probs[idx].argmax(1)).mean()
                print(f"  {g}: original vs standard argmax agreement {agree:.4f}, "
                      f"max |dp| {np.abs(p - probs[idx]).max():.2e}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- throughput check

def bench(args):
    """Training and inference throughput of the crop model at each precision (smoke test)."""
    norm, dev = NORM[args.weights], "cuda"
    ds = PackedCrops(args.crops, "train", True, norm, 20000)
    tl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=n_workers(args), pin_memory=True,
                    drop_last=True, persistent_workers=True)
    t0, n = time.time(), 0
    for it, (x, _) in enumerate(tl):
        n += len(x)
        if it == 60:
            break
    print(f"data loader alone: {n / (time.time() - t0):.0f} img/s (incl. worker start-up)", flush=True)
    model = build_model(args.weights).train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    leaf_crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    sev_t = torch.tensor(SEV_TARGETS, device=dev)
    for precision in ("tf32", "bf16", "fp32"):
        set_precision(precision)
        torch.cuda.reset_peak_memory_stats()
        model.train()
        n, t0 = 0, None
        for it, (x, y) in enumerate(tl):
            if it == 5:
                torch.cuda.synchronize()
                t0 = time.time()
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with autocast(precision):
                loss = gated_loss(model, x, y, leaf_crit, sev_t)
            loss.backward()
            opt.step()
            if t0 is not None:
                n += len(y)
            if it == 25:
                break
        torch.cuda.synchronize()
        tr_rate = n / (time.time() - t0)
        model.eval()
        x = torch.randn(args.eval_batch, 3, 224, 224, device=dev)
        with torch.no_grad(), autocast(precision):
            model(x)
            torch.cuda.synchronize()
            t1 = time.time()
            for _ in range(5):
                model(x)
            torch.cuda.synchronize()
        inf_rate = 5 * args.eval_batch / (time.time() - t1)
        print(f"{precision}: train {tr_rate:.0f} img/s (batch {args.batch}), inference {inf_rate:.0f} img/s, "
              f"peak memory {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GiB, last loss {loss.item():.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prepare", "train", "eval", "bench"])
    ap.add_argument("--data", default=os.path.join(PROJ, "data"))
    ap.add_argument("--image-dir", default=None, help="where post.u8 lives (default: --data)")
    ap.add_argument("--crops", default=os.path.join(HERE, "crops"), help="packed crops (prepare output)")
    ap.add_argument("--run", default=os.path.join(HERE, "runs/dtf_lvd"))
    ap.add_argument("--out", default=os.path.join(PROJ, "results/harness/damage_triageformer"))
    ap.add_argument("--weights", choices=sorted(WEIGHTS), default="lvd")
    ap.add_argument("--precision", choices=["fp32", "tf32", "bf16"], default="tf32")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-batch", type=int, default=128)
    ap.add_argument("--workers", type=int, default=0, help="0 = allocated cores - 1")
    ap.add_argument("--log-every", type=int, default=500)
    ap.add_argument("--stop-after", type=int, default=0, help="train at most this many epochs in this call")
    ap.add_argument("--targets", type=int, default=1500, help="concordant counterfactual targets per group (0 = skip)")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: random subset of this many crops")
    ap.add_argument("--max-iters", type=int, default=0, help="smoke test: iterations per epoch")
    ap.add_argument("--cf-limit", type=int, default=0, help="smoke test: counterfactual targets per group")
    args = ap.parse_args()
    {"prepare": prepare, "train": train, "eval": evaluate, "bench": bench}[args.cmd](args)


if __name__ == "__main__":
    main()
