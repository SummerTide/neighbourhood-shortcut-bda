#!/usr/bin/env python
"""Footprint-conditioned crop baselines on the official xBD splits.

Trains on official train (train + tier3), selects the checkpoint on val (hold),
and evaluates every test building under each evaluation condition of
make_conditions.py (misregistration level x registration method, as footprint
offsets on the untouched post-event image). --eval-only re-evaluates a trained
model on another set of conditions.

--inputs post      ResNet-50 on [post RGB, footprint mask]
--inputs pre+post  Siamese ResNet-50 (shared weights) on [pre RGB, mask] and
                   [post RGB, mask]; the head sees [f_post, f_post - f_pre]
--separate-pre     a second ResNet-50 for the pre-event image (no weight sharing), for
                   different sensors before and after the event (BRIGHT: optical pre, SAR post)
--backbone dinov3_vitl16_sat fine-tunes the satellite DINOv3 ViT-L/16 instead.
--backbone resnet18|resnet101|convnext_tiny|swin_t swaps the ResNet-50 for another ImageNet backbone.
--context-swap trains with the surroundings of the footprint swapped (bda_data.py).
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision

from bda_data import CONTEXT_PX, N_CLASSES, BuildingDataset, ImageSource, load_conditions, load_tables
from bda_metrics import building_metrics

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def resnet50_4ch(weights):
    net = torchvision.models.resnet50()
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    conv = net.conv1
    new = torch.nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = 0.0  # the mask channel starts as a no-op
    net.conv1, net.fc = new, torch.nn.Identity()
    return net


# other torchvision backbones: ImageNet weights in the torch hub cache
TV_WEIGHTS = {"resnet18": "resnet18-f37072fd.pth", "resnet101": "resnet101-63fe2227.pth",
              "convnext_tiny": "convnext_tiny-983f1562.pth", "swin_t": "swin_t-704ceda3.pth"}


def tv_4ch(arch, weights):
    """A torchvision backbone on [RGB, mask], as resnet50_4ch; returns the net and its feature size."""
    if arch == "resnet50":
        return resnet50_4ch(weights), 2048
    net = getattr(torchvision.models, arch)()
    net.load_state_dict(torch.load(weights, map_location="cpu"))
    if arch.startswith("resnet"):
        conv, dim = net.conv1, net.fc.in_features
    else:  # ConvNeXt and Swin: patchify stem at features[0][0]
        conv = net.features[0][0]
        dim = net.classifier[2].in_features if arch.startswith("convnext") else net.head.in_features
    new = torch.nn.Conv2d(4, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=conv.bias is not None)
    with torch.no_grad():
        new.weight[:, :3] = conv.weight
        new.weight[:, 3:] = 0.0
        if conv.bias is not None:
            new.bias.copy_(conv.bias)
    if arch.startswith("resnet"):
        net.conv1, net.fc = new, torch.nn.Identity()
    else:
        net.features[0][0] = new
        if arch.startswith("convnext"):
            net.classifier[2] = torch.nn.Identity()
        else:
            net.head = torch.nn.Identity()
    return net, dim


class CropNet(torch.nn.Module):
    def __init__(self, weights, use_pre, n_classes=N_CLASSES, separate_pre=False, arch="resnet50"):
        super().__init__()
        net, dim = tv_4ch(arch, weights)
        self.backbone, self.use_pre = net, use_pre
        if use_pre and separate_pre:
            self.backbone_pre = tv_4ch(arch, weights)[0]
        dim = dim * (2 if use_pre else 1)
        self.head = torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(512, n_classes))

    def forward(self, post, mask, pre=None):
        f_post = self.backbone(torch.cat([post, mask], 1))
        if not self.use_pre:
            return self.head(f_post)
        f_pre = getattr(self, "backbone_pre", self.backbone)(torch.cat([pre, mask], 1))
        return self.head(torch.cat([f_post, f_post - f_pre], 1))


class ViTCropNet(torch.nn.Module):
    """DINOv3 ViT on [RGB, footprint mask]: the mask enters through a zero-initialised
    fourth channel of the patch embedding; a crop is the class token and the mean patch token."""

    SAT_MEAN, SAT_STD = (0.430, 0.411, 0.296), (0.213, 0.156, 0.143)

    def __init__(self, hub_dir, repo, ckpt, use_pre, n_classes=N_CLASSES):
        super().__init__()
        sys.path.insert(0, repo)
        from dinov3.hub import backbones
        net = backbones.dinov3_vitl16(pretrained=False, weights=ckpt)
        net.load_state_dict(torch.load(os.path.join(hub_dir, "checkpoints", ckpt), map_location="cpu"), strict=True)
        pe = net.patch_embed.proj
        new = torch.nn.Conv2d(4, pe.out_channels, pe.kernel_size, pe.stride, bias=pe.bias is not None)
        with torch.no_grad():
            new.weight[:, :3] = pe.weight
            new.weight[:, 3:] = 0.0
            if pe.bias is not None:
                new.bias.copy_(pe.bias)
        net.patch_embed.proj, net.patch_embed.in_chans = new, 4
        self.backbone, self.use_pre = net, use_pre
        dim = 2 * net.embed_dim * (2 if use_pre else 1)
        self.head = torch.nn.Sequential(torch.nn.Linear(dim, 512), torch.nn.ReLU(), torch.nn.Dropout(0.2),
                                        torch.nn.Linear(512, n_classes))
        # to_input normalises with ImageNet statistics; the SAT model expects its own.
        self.register_buffer("scale", (STD.view(3) / torch.tensor(self.SAT_STD)).view(1, 3, 1, 1))
        self.register_buffer("shift", ((MEAN.view(3) - torch.tensor(self.SAT_MEAN)) / torch.tensor(self.SAT_STD)).view(1, 3, 1, 1))

    def embed(self, img, mask):
        out = self.backbone.forward_features(torch.cat([img * self.scale + self.shift, mask], 1))
        return torch.cat([out["x_norm_clstoken"], out["x_norm_patchtokens"].mean(1)], 1)

    def forward(self, post, mask, pre=None):
        f_post = self.embed(post, mask)
        if not self.use_pre:
            return self.head(f_post)
        return self.head(torch.cat([f_post, f_post - self.embed(pre, mask)], 1))


class TwoStreamNet(torch.nn.Module):
    """Building evidence and neighbourhood context as two experts whose logits add up (a
    product of experts): `context` sees the usual crop, `building` a zoomed crop blanked
    outside the footprint rim (the tight_view of BuildingDataset). Each expert also gets
    its own loss in training, so neither can leave the decision to the other."""

    def __init__(self, context, building):
        super().__init__()
        self.context, self.building = context, building

    def forward(self, post, mask, pre=None, post_t=None, mask_t=None, pre_t=None, experts=False):
        z_c, z_b = self.context(post, mask, pre), self.building(post_t, mask_t, pre_t)
        return (z_c, z_b) if experts else z_c + z_b


def build_model(args, n_classes=N_CLASSES):
    use_pre = args.inputs == "pre+post"

    def one():
        if args.backbone == "resnet50":
            return CropNet(args.weights, use_pre, n_classes, getattr(args, "separate_pre", False)).cuda().to(memory_format=torch.channels_last)
        if args.backbone in TV_WEIGHTS:
            w = os.path.join(args.hub_dir, "checkpoints", TV_WEIGHTS[args.backbone])
            return CropNet(w, use_pre, n_classes, getattr(args, "separate_pre", False),
                           args.backbone).cuda().to(memory_format=torch.channels_last)
        return ViTCropNet(args.hub_dir, args.dinov3_repo, "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth", use_pre,
                          n_classes).cuda()
    return TwoStreamNet(one(), one()) if getattr(args, "two_stream", False) else one()


def param_groups(model, args):
    """Backbone(s) at lr, head(s) at lr x head_lr_mult."""
    nets = [model.context, model.building] if isinstance(model, TwoStreamNet) else [model]
    enc = [p for n in nets for m in (n.backbone, getattr(n, "backbone_pre", None)) if m is not None for p in m.parameters()]
    return [{"params": enc, "lr": args.lr},
            {"params": [p for n in nets for p in n.head.parameters()], "lr": args.lr * args.head_lr_mult}]


def model_inputs(batch, augment):
    """Positional inputs of the model for a batch: (post, mask, pre) and, when the batch
    carries a tight view, (post_t, mask_t, pre_t) with its own random flips/rotation."""
    x = to_input(batch, augment)
    if "post_t" in batch:
        x = x + to_input({k[:-2]: v for k, v in batch.items() if k.endswith("_t")}, augment)
    return x


def to_input(batch, augment):
    def img(x):
        x = x.cuda(non_blocking=True).permute(0, 3, 1, 2).float().div_(255)
        return (x - MEAN.cuda()) / STD.cuda()
    post = img(batch["post"])
    mask = batch["mask"].cuda(non_blocking=True).float().div_(255).unsqueeze(1)
    pre = img(batch["pre"]) if "pre" in batch else None
    if augment:  # the same flips/rotation for every input of a sample batch
        k = int(torch.randint(4, (1,)))
        flip = bool(torch.rand(1) < 0.5)
        def geo(x):
            if x is None:
                return None
            x = torch.rot90(x, k, dims=(2, 3))
            return x.flip(3) if flip else x
        post, mask, pre = geo(post), geo(mask), geo(pre)
    cl = torch.channels_last
    return post.contiguous(memory_format=cl), mask.contiguous(memory_format=cl), \
        None if pre is None else pre.contiguous(memory_format=cl)


@torch.no_grad()
def predict(model, loader):
    model.eval()
    probs, idx = [], []
    for batch in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs.append(F.softmax(model(*model_inputs(batch, False)).float(), 1).cpu())
        idx.append(batch["index"])
    model.train()
    p = torch.cat(probs).numpy()
    out = np.zeros_like(p)
    out[torch.cat(idx).numpy()] = p
    return out


def crop_args(args):
    out = {"context_px": args.context_px, "tight_margin": args.tight_margin}
    if getattr(args, "two_stream", False):
        out["tight_view"] = (args.tight_view_px, args.tight_view_margin)
    return out


def discordance_weights(tr, weight):
    """Per-building sampling weights: `weight` for buildings whose damage disagrees with
    their labelled neighbours (isolated damage / intact, level discordant), 1 otherwise."""
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis"))
    from discordant import categories
    cat = categories(tr.reset_index(drop=True), 64)
    hard = np.isin(cat, ["isolated damage", "isolated intact", "level discordant"])
    print(f"discordance weighting: {hard.sum()} of {len(tr)} training buildings x{weight}", flush=True)
    return np.where(hard, weight, 1.0)


def extra_dataset(args):
    """Extra training buildings from another table + memmap set (e.g. the xFBD synthetic
    single-building tier1 set of prep_xfbd.py: Melamed et al.'s augmentation). For xFBD only
    the pasted building and the buildings within --extra-max-dist px of it are used."""
    scenes, b = load_tables(args.extra_train)
    b = b[b.label >= 0]
    if "dist_pasted" in b:
        b = b[b.dist_pasted <= args.extra_max_dist]
    post = ImageSource(os.path.join(args.extra_train, "post.u8"), len(scenes))
    pre = ImageSource(os.path.join(args.extra_train, "pre.u8"), len(scenes)) if args.inputs == "pre+post" else None
    print(f"extra training buildings from {args.extra_train}: {len(b)} "
          f"(levels {np.bincount(b.label, minlength=N_CLASSES).tolist()})", flush=True)
    return BuildingDataset(b, post, pre, train=True, jitter_px=args.jitter_px, **crop_args(args))


def train(args, model, tr, va, post_src, pre_src, dl, swap_pre_src=None):
    swap = {} if args.context_swap == "none" else {
        "context_swap": args.context_swap, "context_swap_p": args.context_swap_p, "swap_pre_src": swap_pre_src,
        "swap_blend": args.swap_blend, "swap_donor": args.swap_donor,
        "swap_margin": tuple(float(v) for v in (args.swap_margin.replace(":", ",").split(",") * 2)[:2])}
    train_ds = BuildingDataset(tr, post_src, pre_src, train=True, jitter_px=args.jitter_px, **crop_args(args), **swap)
    n_extra = 0
    if args.extra_train:
        extra_ds = extra_dataset(args)
        n_extra = len(extra_ds)
        train_ds = torch.utils.data.ConcatDataset([train_ds, extra_ds])
    if args.discordance_weight != 1 or n_extra:
        w = discordance_weights(tr, args.discordance_weight) if args.discordance_weight != 1 else np.ones(len(tr))
        w = torch.as_tensor(np.concatenate([w, np.ones(n_extra)]), dtype=torch.double)
        train_dl = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch, sampler=torch.utils.data.WeightedRandomSampler(w, len(w)),
            drop_last=True, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    else:
        train_dl = dl(train_ds, True)
    val_dl = dl(BuildingDataset(va, post_src, pre_src, **crop_args(args)), False)
    cw = None
    if args.class_weight == "balanced":
        cnt = np.bincount(tr.label, minlength=N_CLASSES)
        cw = torch.tensor(len(tr) / (N_CLASSES * cnt), dtype=torch.float32).cuda()
    groups = param_groups(model, args)
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=args.epochs * len(train_dl), pct_start=0.05)
    best, t0, log = -1.0, time.time(), []
    for ep in range(args.epochs):
        for batch in train_dl:
            x, y = model_inputs(batch, True), batch["label"].cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                if isinstance(model, TwoStreamNet):
                    z_c, z_b = (z.float() for z in model(*x, experts=True))
                    loss = F.cross_entropy(z_c + z_b, y, weight=cw) + args.expert_loss * (
                        F.cross_entropy(z_c, y, weight=cw) + F.cross_entropy(z_b, y, weight=cw))
                else:
                    loss = F.cross_entropy(model(*x).float(), y, weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
        p = predict(model, val_dl)
        m = building_metrics(va.label.to_numpy(), p, va.disaster.to_numpy())
        log.append({"epoch": ep + 1, "loss": float(loss), **m["overall"]})
        print(f"epoch {ep + 1}/{args.epochs} loss {loss.item():.3f} val macro-F1 {m['overall']['macro_f1']:.3f} "
              f"QWK {m['overall']['qwk']:.3f} ({time.time() - t0:.0f}s)", flush=True)
        if m["overall"]["macro_f1"] > best:
            best = m["overall"]["macro_f1"]
            torch.save(model.state_dict(), os.path.join(args.out, "best.pt"))
    pd.DataFrame(log).to_csv(os.path.join(args.out, "val_log.csv"), index=False)
    with open(os.path.join(args.out, "args.json"), "w") as fh:
        json.dump(vars(args), fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--inputs", choices=["post", "pre+post"], default="post")
    ap.add_argument("--class-weight", choices=["none", "balanced"], default="none")
    ap.add_argument("--jitter-px", type=float, default=0.0, help="training-time misregistration (std, px)")
    ap.add_argument("--context-px", type=int, default=CONTEXT_PX, help="crop margin around the footprint box (px)")
    ap.add_argument("--tight-margin", type=float, default=None,
                    help="blank crop pixels farther than this from the footprint (px): no neighbourhood context")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--head-lr-mult", type=float, default=1.0, help="head learning rate = lr x this")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--backbone", choices=["resnet50", "resnet18", "resnet101", "convnext_tiny", "swin_t", "dinov3_vitl16_sat"], default="resnet50")
    ap.add_argument("--separate-pre", action="store_true", help="separate ResNet-50 for the pre-event image")
    ap.add_argument("--hub-dir", default="external/torch_hub")
    ap.add_argument("--dinov3-repo", default="external/dinov3")
    ap.add_argument("--context-swap", choices=["none", "donor", "pre"], default="none",
                    help="train with the surroundings of the footprint swapped (see bda_data.BuildingDataset)")
    ap.add_argument("--context-swap-p", type=float, default=0.5)
    ap.add_argument("--swap-blend", choices=["hard", "poisson"], default="hard",
                    help="how the building is put into the swapped surroundings")
    ap.add_argument("--swap-margin", default="4", help="kept rim width (px), or a range 'lo,hi' / 'lo:hi' drawn per sample")
    ap.add_argument("--swap-donor", choices=["any", "event"], default="any", help="donor pool for --context-swap donor")
    ap.add_argument("--two-stream", action="store_true",
                    help="context crop + tight building crop as two experts whose logits add (TwoStreamNet)")
    ap.add_argument("--tight-view-px", type=int, default=8, help="two-stream: context of the building crop (px)")
    ap.add_argument("--tight-view-margin", type=float, default=4.0, help="two-stream: rim kept around the footprint (px)")
    ap.add_argument("--expert-loss", type=float, default=1.0, help="two-stream: weight of each expert's own loss")
    ap.add_argument("--extra-train", default=None, help="extra training set (table + memmap dir, e.g. data_xfbd_tier1)")
    ap.add_argument("--extra-max-dist", type=float, default=64.0, help="xFBD: keep buildings this close to the pasted one")
    ap.add_argument("--discordance-weight", type=float, default=1.0,
                    help="oversample training buildings whose damage disagrees with their neighbours by this factor")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--image-dir", default=None, help="where pre.u8/post.u8 live (e.g. a copy in /dev/shm)")
    ap.add_argument("--weights", default="external/checkpoints/resnet50-11ad3fa6.pth")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond-dir", default=None, help="evaluation conditions (default: <data>/conditions_v2)")
    ap.add_argument("--eval-only", action="store_true", help="evaluate <out>/best.pt without training")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: at most this many buildings per split")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True
    os.makedirs(args.out, exist_ok=True)

    scenes, buildings = load_tables(args.data)
    n_scenes = len(scenes)
    img_dir = args.image_dir or args.data
    post_src = ImageSource(os.path.join(img_dir, "post.u8"), n_scenes)
    pre_src = ImageSource(os.path.join(img_dir, "pre.u8"), n_scenes) if args.inputs == "pre+post" else None
    lab = buildings[buildings.label >= 0]
    tr, va, te = (lab[lab.official == s] for s in ("train", "val", "test"))
    if args.limit:
        tr, va, te = (d.sample(min(len(d), args.limit), random_state=0).sort_index() for d in (tr, va, te))
    print(f"train {len(tr)}, val {len(va)}, test {len(te)} buildings; inputs={args.inputs}", flush=True)

    dl = lambda ds, shuffle: torch.utils.data.DataLoader(ds, batch_size=args.batch if shuffle else 256, shuffle=shuffle,
                                                         drop_last=shuffle, num_workers=args.workers, pin_memory=True,
                                                         persistent_workers=shuffle)
    model = build_model(args)
    t0 = time.time()
    if not args.eval_only:
        swap_pre = ImageSource(os.path.join(img_dir, "pre.u8"), n_scenes) if args.context_swap == "pre" else None
        train(args, model, tr, va, post_src, pre_src, dl, swap_pre)

    model.load_state_dict(torch.load(os.path.join(args.out, "best.pt"), map_location="cuda"))
    cond_dir = args.cond_dir or os.path.join(args.data, "conditions_v2")
    if not os.path.exists(os.path.join(cond_dir, "manifest.json")):
        print(f"no conditions in {cond_dir}; evaluate later with --eval-only", flush=True)
        return
    conds, scene_ids = load_conditions(cond_dir, te.building_id.to_numpy())
    rows, preds = [], {}
    for c in conds:
        src = post_src if c["post"] is None else ImageSource(c["post"], len(scene_ids), scene_ids)
        p = predict(model, dl(BuildingDataset(te, src, pre_src, offsets=c["offsets"], **crop_args(args)), False))
        preds[c["name"]] = p
        m = building_metrics(te.label.to_numpy(), p, te.disaster.to_numpy())
        rows.append({"condition": c["name"], **m["overall"]})
        print(f"test {c['name']}: macro-F1 {m['overall']['macro_f1']:.3f} QWK {m['overall']['qwk']:.3f} "
              f"F1 minor/major/destroyed {m['overall']['f1_1']:.3f}/{m['overall']['f1_2']:.3f}/{m['overall']['f1_3']:.3f}",
              flush=True)
    tag = os.path.basename(os.path.normpath(cond_dir))
    pd.DataFrame(rows).to_csv(os.path.join(args.out, f"test_{tag}.csv"), index=False)
    np.savez_compressed(os.path.join(args.out, f"test_{tag}_preds.npz"), building_id=te.building_id.to_numpy(),
                        label=te.label.to_numpy(), **preds)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
