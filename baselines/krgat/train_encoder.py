#!/usr/bin/env python
"""The KR-GAT patch encoder: a ResNet-50 with a bitemporal (6-channel) input stem on PPC patches.

The paper uses a pretrained, frozen "Res50 (xView2 #1)" encoder (Fig. 1) whose features already
separate the damage levels (patch-only head: 0.822 macro-F1 on hold). We train that encoder as a
patch classifier on official train (train + tier3), keep the epoch with the best val (hold)
macro-F1, freeze it and store the 2048-d embedding of every node (data/feats_ppc_r50.f16).
--imagenet instead stores frozen ImageNet ResNet-50 embeddings [f_pre, f_post] (4096-d,
data/feats_in1k_r50.f16), a no-xBD-training sensitivity check.

Loss as the paper's heads: effective-number class-weighted CE + 0.25 EMD (beta = 0.9999).
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from common import DATA, INPUT, PATCH, R50

MEAN = torch.tensor([0.485, 0.456, 0.406] * 2).view(1, 6, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225] * 2).view(1, 6, 1, 1)


def class_weights(labels, beta, n_classes=4):
    """Effective-number weights (Cui et al. 2019), normalised to sum to n_classes."""
    n = np.bincount(labels, minlength=n_classes).astype(np.float64)
    w = (1 - beta) / (1 - np.power(beta, n))
    return torch.tensor(w / w.sum() * n_classes, dtype=torch.float32)


def cls_loss(logits, y, w, lam_emd=0.25):
    """Eq. 11 + lam_emd * Eq. 12: (1/B) sum w_y (-log p_y) and the squared CDF (EMD) loss."""
    logp = F.log_softmax(logits.float(), 1)
    ce = -(w[y] * logp.gather(1, y[:, None])[:, 0]).mean()
    cdf = logp.exp().cumsum(1)
    target = (torch.arange(logits.shape[1], device=y.device)[None] >= y[:, None]).float()
    emd = ((cdf - target) ** 2).sum(1).mean()
    return ce + lam_emd * emd, ce, emd


class PPCEncoder(torch.nn.Module):
    def __init__(self, weights=R50):
        super().__init__()
        net = torchvision.models.resnet50()
        net.load_state_dict(torch.load(weights, map_location="cpu"))
        w = net.conv1.weight.data
        net.conv1 = torch.nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        net.conv1.weight.data = torch.cat([w, w], 1) / 2  # pre and post share the ImageNet stem at start
        net.fc = torch.nn.Identity()
        self.backbone, self.head = net, torch.nn.Linear(2048, 4)

    def forward(self, x):
        f = self.backbone(x)
        return self.head(f), f


def to_input(xb, augment=False):
    """uint8 (B, P, P, 6) -> normalised float (B, 6, INPUT, INPUT); training: random dihedral
    transform plus a slight zoom/shift, all in one grid_sample."""
    x = xb.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
    x = (x - MEAN.cuda()) / STD.cuda()
    if not augment:
        x = F.interpolate(x, size=(INPUT, INPUT), mode="bilinear", align_corners=False)
    else:
        b = x.shape[0]
        ang = torch.randint(4, (b,), device=x.device).float() * (math.pi / 2)
        flip = torch.where(torch.rand(b, device=x.device) < 0.5, -1.0, 1.0)
        s = torch.empty(b, device=x.device).uniform_(0.9, 1.0)
        t = torch.empty(b, 2, device=x.device).uniform_(-0.05, 0.05)
        cos, sin = torch.cos(ang) * s, torch.sin(ang) * s
        theta = torch.stack([torch.stack([cos * flip, -sin, t[:, 0]], 1), torch.stack([sin * flip, cos, t[:, 1]], 1)], 1)
        grid = F.affine_grid(theta, (b, 6, INPUT, INPUT), align_corners=False)
        x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def run(model, bank, idx, batch=512):
    model.eval()
    logits, feats = [], []
    for s in range(0, len(idx), batch):
        xb = torch.from_numpy(bank[np.sort(idx[s:s + batch])] if False else bank[idx[s:s + batch]])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo, f = model(to_input(xb))
        logits.append(lo.float().cpu())
        feats.append(f.float().cpu())
    model.train()
    return torch.cat(logits).numpy(), torch.cat(feats).numpy()


def macro_f1(y, logits):
    from bda_metrics import building_metrics
    return building_metrics(y, torch.softmax(torch.from_numpy(logits), 1).numpy())["overall"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.9999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=f"{DATA}/encoder")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: train/val subsample size")
    ap.add_argument("--extract-only", action="store_true")
    ap.add_argument("--imagenet", action="store_true", help="only store frozen ImageNet [f_pre, f_post]")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)
    g = np.load(f"{DATA}/graph.npz")
    split, y_all = g["split"], g["label"].astype(np.int64)
    n = len(split)
    t0 = time.time()
    bank = np.fromfile(f"{DATA}/ppc_{PATCH}.u8", np.uint8).reshape(n, PATCH, PATCH, 6)
    print(f"bank loaded: {bank.shape} ({time.time() - t0:.0f}s)", flush=True)

    if args.imagenet:
        net = torchvision.models.resnet50()
        net.load_state_dict(torch.load(R50, map_location="cpu"))
        net.fc = torch.nn.Identity()
        net = net.cuda().eval().to(memory_format=torch.channels_last)
        out = np.memmap(f"{DATA}/feats_in1k_r50.f16", np.float16, "w+", shape=(n, 4096))
        with torch.no_grad():
            for s in range(0, n, 512):
                x = to_input(torch.from_numpy(bank[s:s + 512]))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = torch.cat([net(x[:, :3].contiguous(memory_format=torch.channels_last)),
                                   net(x[:, 3:].contiguous(memory_format=torch.channels_last))], 1)
                out[s:s + 512] = f.float().cpu().numpy().astype(np.float16)
        out.flush()
        print(f"wrote feats_in1k_r50.f16 ({time.time() - t0:.0f}s)", flush=True)
        return

    tr, va = np.flatnonzero(split == 0), np.flatnonzero(split == 1)
    if args.limit:
        rng = np.random.RandomState(0)
        tr, va = np.sort(rng.choice(tr, args.limit, False)), np.sort(rng.choice(va, args.limit // 4, False))
    model = PPCEncoder().cuda().to(memory_format=torch.channels_last)
    if not args.extract_only:
        w = class_weights(y_all[split == 0], args.beta).cuda()
        print("class weights", w.cpu().numpy().round(3).tolist(), flush=True)
        groups = [{"params": model.backbone.parameters(), "lr": args.lr},
                  {"params": model.head.parameters(), "lr": args.head_lr}]
        opt = torch.optim.AdamW(groups, weight_decay=args.wd)
        steps = len(tr) // args.batch
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr, args.head_lr], total_steps=args.epochs * steps,
                                                    pct_start=0.05)
        best, log = -1, []
        y_t = torch.from_numpy(y_all)
        for ep in range(args.epochs):
            perm = np.random.permutation(tr)
            run_loss = 0.0
            for it in range(steps):
                idx = np.sort(perm[it * args.batch:(it + 1) * args.batch])
                x = to_input(torch.from_numpy(bank[idx]), augment=True)
                yb = y_t[idx].cuda(non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits, _ = model(x)
                loss, ce, emd = cls_loss(logits, yb, w)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                sched.step()
                run_loss += loss.item() if it % 50 == 0 else 0.0
                if (it + 1) % 500 == 0:
                    print(f"  ep {ep + 1} it {it + 1}/{steps} loss {loss.item():.3f} ({time.time() - t0:.0f}s)", flush=True)
            lv, _ = run(model, bank, va)
            m = macro_f1(y_all[va], lv)
            log.append({"epoch": ep + 1, **m})
            print(f"epoch {ep + 1}: val macro-F1 {m['macro_f1']:.4f} F1 {m['f1_0']:.3f}/{m['f1_1']:.3f}/{m['f1_2']:.3f}/"
                  f"{m['f1_3']:.3f} QWK {m['qwk']:.3f} ({time.time() - t0:.0f}s)", flush=True)
            if m["macro_f1"] > best:
                best = m["macro_f1"]
                torch.save(model.state_dict(), f"{args.out}/best.pt")
        with open(f"{args.out}/val_log.json", "w") as fh:
            json.dump({"args": vars(args), "log": log}, fh, indent=1)
    model.load_state_dict(torch.load(f"{args.out}/best.pt", map_location="cuda"))
    if args.limit:
        return
    # Frozen embeddings (and the encoder's own patch-classifier logits) of every node.
    feats = np.memmap(f"{DATA}/feats_ppc_r50.f16", np.float16, "w+", shape=(n, 2048))
    logits = np.zeros((n, 4), np.float32)
    for s in range(0, n, 8192):
        idx = np.arange(s, min(n, s + 8192))
        lo, f = run(model, bank, idx)
        feats[idx], logits[idx] = f.astype(np.float16), lo
    feats.flush()
    np.save(f"{args.out}/logits.npy", logits)
    for k, name in enumerate(("train", "val", "test")):
        m = macro_f1(y_all[split == k], logits[split == k])
        print(f"encoder head on {name}: macro-F1 {m['macro_f1']:.4f} acc {m['acc']:.4f} QWK {m['qwk']:.3f}", flush=True)
    print(f"wrote feats_ppc_r50.f16 ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
