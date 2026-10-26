#!/usr/bin/env python
"""KR-GAT (Hasan & Yeum, ECCV 2026, arXiv 2608.20548) and its patch-only / vanilla-GAT ablations,
on frozen PPC patch embeddings and per-event k-NN building graphs (data/graph.npz).

  patch   no graph: the same residual stack with a self-only update v <- v + elu(W v)
  gat     vanilla GAT (Velickovic et al.): e_ij = LeakyReLU(a_d.Wv_i + a_s.Wv_j), softmax over kNN(i),
          v_i <- v_i + elu(sum_j alpha_ij W v_j); no edge features, no disaster token, no kernel
  krgat   Eq. 2-9: disaster-type token z (7 roll-ups), DCMS kernel K_ij = sum_m pi_m exp(-d_ij/rho_m)
          (M = 3), compatibility psi_ij = MLP([W_q v_i | W_k v_j | u_ij | z]), logit psi + lam_K log(K + eps),
          v_i <- v_i + elu(sum_j alpha_ij W_v v_j); plus the residual Moran loss (Eq. 13-14)
All: input = [frozen embedding, standardised log area, log perimeter] -> Linear(256) -> LN -> GELU,
two layers (4 heads), LN -> Linear(4); loss = effective-number weighted CE + 0.25 EMD (+ 0.1 L_res for
krgat). Mini-batches are groups of training tiles with their 2-hop neighbourhoods (exact for 2 layers).
Train on official train, select epoch and seed on val (hold), test once.

Stages: --stage train (all models x seeds), --stage test (best seed per model: harness outputs +
counterfactual on the official test split, written to footprint_bda/results/harness/<name>/).

Paper recipe -> implementation (choices the paper leaves open are marked *):
  PPC patch      tight box of the polygon + small margin*(max(8 px, 10 % of the box side)), pre and post
                 crops resized* to 112 px (fed at 224) -> 6-channel tensor          [make_patches.py]
  encoder        "pretrained, frozen Res50 (xView2 #1)" with a bitemporal input stem: no released
                 weights, so a 6-channel ResNet-50 (ImageNet init) is trained as a PPC patch classifier
                 on official train, the best-on-hold epoch frozen; 2048-d embedding  [train_encoder.py]
  node features  embedding + area and perimeter (m, log, standardised*)
  graph          centroids of the lng_lat polygons in UTM (zone of the event's median longitude),
                 directed kNN, k = 16, one graph per event and official split*; u_ij = [d_ij / 100 m*,
                 unit displacement]                                                    [prep_graph.py]
  DCMS kernel    7 roll-up tokens (Sec. 4.1), d_z = 16*, M = 3, rho_min = 1 m*, init rho = 10/40/160 m*,
                 distances in units of 100 m*; kernel + token parameters at 10x lr, no weight decay*
  attention      a_phi = 2-layer MLP (hidden 64*), 4 heads*, lam_K = 1*, 2 layers*, width 256*
  losses         CE with effective-number weights (beta = 0.9999*, sum 4) + 0.25 EMD + 0.1 L_res
                 (L_res per event over the core nodes of a mini-batch, events with >= 10 nodes*)
  optimiser      AdamW* lr 1e-3, wd 1e-2, 30 epochs, cosine, 1-epoch warm-up, 48 tiles per batch*
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, IMG, PATCH, PROJ, ROLLUPS, ROOT, inpainted_ppc_patch, ppc_box, ppc_patch
from train_encoder import PPCEncoder, class_weights, cls_loss, to_input

FEATS = {"ppc_r50": ("feats_ppc_r50.f16", 2048), "in1k_r50": ("feats_in1k_r50.f16", 4096)}
HARNESS_NAME = {"krgat": "krgat", "patch": "krgat_patch_only", "gat": "krgat_vanilla_gat"}
LEVELS = ["no", "minor", "major", "destroyed"]
DEV = os.environ.get("KRGAT_DEVICE", "cuda")
FAKE = bool(os.environ.get("KRGAT_FAKE"))  # smoke tests: random embeddings, no encoder


# ------------------------------------------------------------------------------------------ data
class SplitGraph:
    """Nodes of one official split, the per-event k-NN tables in split-local indices, edge features."""

    def __init__(self, g, s, X):
        idx = np.flatnonzero(g["split"] == s)
        loc = np.full(len(g["split"]), -1, np.int64)
        loc[idx] = np.arange(len(idx))
        nb = g["nbr"][idx]
        valid = nb >= 0
        nbl = np.where(valid, loc[np.maximum(nb, 0)], -1)
        assert (nbl[valid] >= 0).all(), "edges must stay inside the split"
        x, y = g["x"][idx], g["y"][idx]
        dx = np.where(valid, x[np.maximum(nbl, 0)] - x[:, None], 0.0)
        dy = np.where(valid, y[np.maximum(nbl, 0)] - y[:, None], 0.0)
        dist = np.hypot(dx, dy)
        safe = np.maximum(dist, 1e-6)
        u = np.stack([dist / 100.0, np.where(dist > 0, dx / safe, 0), np.where(dist > 0, dy / safe, 0)], -1)
        self.idx, self.n = idx, len(idx)
        self.nbr_np = nbl
        self.nbr = torch.from_numpy(nbl).to(DEV)
        self.d = torch.from_numpy(dist.astype(np.float32)).to(DEV)
        self.u = torch.from_numpy(u.astype(np.float32)).to(DEV)
        self.rollup = torch.from_numpy(g["rollup"][idx].astype(np.int64)).to(DEV)
        self.event_np = g["event"][idx].astype(np.int64)
        self.event = torch.from_numpy(self.event_np).to(DEV)
        self.y_np = g["label"][idx].astype(np.int64)
        self.y = torch.from_numpy(self.y_np).to(DEV)
        self.scene = g["scene_id"][idx]
        self.x = torch.from_numpy(np.ascontiguousarray(X[idx])).to(DEV)  # float16
        # tiles (scenes) as contiguous node ranges, for mini-batches
        starts = np.flatnonzero(np.r_[True, self.scene[1:] != self.scene[:-1]])
        self.tile_ranges = np.stack([starts, np.r_[starts[1:], self.n]], 1)


def load_inputs(feat):
    g = dict(np.load(f"{DATA}/graph.npz"))
    fname, dim = FEATS[feat]
    n = len(g["split"])
    if FAKE:
        rs = np.random.RandomState(0)
        emb = rs.standard_normal(size=(n, 64)).astype(np.float16)
        emb[np.arange(n), g["label"]] += 3.0  # some signal
    else:
        emb = np.memmap(f"{DATA}/{fname}", np.float16, "r", shape=(n, dim))
    tr = g["split"] == 0
    meta = np.stack([np.log(g["area_m2"] + 1.0), np.log(g["perim_m"] + 1.0)], 1)
    meta = (meta - meta[tr].mean(0)) / meta[tr].std(0)
    X = np.concatenate([np.asarray(emb), meta.astype(np.float16)], 1)
    return g, X


# ------------------------------------------------------------------------------------------ model
class DCMSKernel(nn.Module):
    """pi = softmax(W_pi z + b_pi), rho_m = rho_min + softplus(w_m.z + b_m), K = sum_m pi_m exp(-d / rho_m).
    Distances and length scales are in units of 100 m (as the edge features), so that rho can move
    within the few epochs before early stopping; rho_min = 1 m, initial rho = 10 / 40 / 160 m."""

    UNIT = 100.0

    def __init__(self, d_z=16, m=3, rho_min=0.01, rho_init=(0.1, 0.4, 1.6)):
        super().__init__()
        self.pi, self.rho, self.rho_min = nn.Linear(d_z, m), nn.Linear(d_z, m), rho_min
        for lin in (self.pi, self.rho):
            nn.init.normal_(lin.weight, std=0.01)
        nn.init.zeros_(self.pi.bias)
        with torch.no_grad():  # inverse softplus, in float64 (expm1 of large values overflows float32)
            r = torch.tensor(rho_init, dtype=torch.float64) - rho_min
            self.rho.bias.copy_((r + torch.log(-torch.expm1(-r))).float())

    def params(self, z):
        """Mixture weights and length scales (metres)."""
        return torch.softmax(self.pi(z), -1), (self.rho_min + F.softplus(self.rho(z))) * self.UNIT

    def forward(self, z, d):
        pi, rho = self.params(z)
        return (pi[:, None, :] * torch.exp(-d[..., None] / (rho[:, None, :] + 1e-4))).sum(-1)


class GraphLayer(nn.Module):
    def __init__(self, kind, dim=256, heads=4, d_a=64, d_z=16, p=0.1):
        super().__init__()
        self.kind, self.h, self.dh = kind, heads, dim // heads
        self.norm, self.drop = nn.LayerNorm(dim), nn.Dropout(p)
        if kind == "patch":
            self.v = nn.Linear(dim, dim)
        elif kind == "gat":
            self.w = nn.Linear(dim, dim, bias=False)
            self.a_dst = nn.Parameter(torch.empty(heads, self.dh))
            self.a_src = nn.Parameter(torch.empty(heads, self.dh))
            nn.init.xavier_uniform_(self.a_dst)
            nn.init.xavier_uniform_(self.a_src)
        else:
            self.q, self.k = nn.Linear(dim, d_a), nn.Linear(dim, d_a, bias=False)
            self.e, self.zp = nn.Linear(3, d_a, bias=False), nn.Linear(d_z, d_a, bias=False)
            self.a = nn.Linear(d_a, heads)
            self.v = nn.Linear(dim, dim)
        self.alpha = None

    def forward(self, h, nbr, u, z, logk, keep_alpha=False):
        hn = self.norm(h)
        if self.kind == "patch":
            return h + self.drop(F.elu(self.v(hn)))
        n = h.shape[0]
        valid = nbr >= 0
        j = nbr.clamp(min=0)
        if self.kind == "gat":
            wh = self.w(hn).view(n, self.h, self.dh)
            e = F.leaky_relu((wh * self.a_dst).sum(-1)[:, None, :] + (wh * self.a_src).sum(-1)[j], 0.2)
            vals = wh[j]
        else:
            hid = self.q(hn)[:, None, :] + self.k(hn)[j] + self.e(u) + self.zp(z)[:, None, :]
            e = self.a(F.leaky_relu(hid, 0.2))
            if logk is not None:
                e = e + logk[..., None]
            vals = self.v(hn).view(n, self.h, self.dh)[j]
        e = e.masked_fill(~valid[..., None], -1e4)
        alpha = torch.softmax(e, 1) * valid[..., None]
        if keep_alpha:
            self.alpha = alpha.detach()
        msg = (alpha[..., None] * vals).sum(1).reshape(n, -1)
        return h + self.drop(F.elu(msg))


class Net(nn.Module):
    def __init__(self, kind, d_in, dim=256, heads=4, layers=2, d_a=64, d_z=16, m=3, p_in=0.2, p=0.1, lam_k=1.0):
        super().__init__()
        self.kind, self.lam_k = kind, lam_k
        self.inp = nn.Sequential(nn.Dropout(p_in), nn.Linear(d_in, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(p))
        self.layers = nn.ModuleList([GraphLayer(kind, dim, heads, d_a, d_z, p) for _ in range(layers)])
        if kind == "krgat":
            self.z = nn.Embedding(len(ROLLUPS), d_z)
            self.kernel = DCMSKernel(d_z, m)
        self.out = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 4))

    def forward(self, x, nbr, u, d, rollup, keep_alpha=False):
        h = self.inp(x.float())
        z = logk = None
        if self.kind == "krgat":
            z = self.z(rollup)
            logk = self.lam_k * torch.log(self.kernel(z, d) + 1e-6)
        for layer in self.layers:
            h = layer(h, nbr, u, z, logk, keep_alpha)
        return self.out(h)


# ------------------------------------------------------------------------------------------ graph ops
def closure(G, core, hops):
    s, fr = core, core
    for _ in range(hops):
        nb = G.nbr[fr].flatten()
        fr = torch.unique(nb[nb >= 0])
        s = torch.cat([s, fr])
    return torch.unique(s)


def sub_forward(model, G, nodes, x=None, rollup=None, keep_alpha=False):
    """Logits of the subgraph induced by `nodes` (split-local, unique); edges leaving it are dropped."""
    m = torch.full((G.n,), -1, dtype=torch.long, device=DEV)
    m[nodes] = torch.arange(len(nodes), device=DEV)
    nb = G.nbr[nodes]
    snb = torch.where(nb >= 0, m[nb.clamp(min=0)], -1)
    return model(G.x[nodes] if x is None else x, snb, G.u[nodes], G.d[nodes],
                 G.rollup[nodes] if rollup is None else rollup, keep_alpha), m


def moran_loss(r, ev, cnb, min_n=10):
    """Eq. 13-14 on the core nodes of a mini-batch: residuals r, event ids ev, core-core neighbour
    table cnb (positions in r, -1 = none; W row-normalised over the kept neighbours)."""
    uniq, inv = torch.unique(ev, return_inverse=True)
    ne = len(uniq)
    one = torch.ones_like(r)
    cnt_e = torch.zeros(ne, device=DEV).index_add_(0, inv, one)
    mean_e = torch.zeros(ne, device=DEV).index_add_(0, inv, r) / cnt_e
    rt = r - mean_e[inv]
    valid = cnb >= 0
    cnt = valid.sum(1)
    lag = (rt[cnb.clamp(min=0)] * valid).sum(1) / cnt.clamp(min=1)
    has = (cnt > 0).float()
    num = torch.zeros(ne, device=DEV).index_add_(0, inv, rt * lag * has)
    den = torch.zeros(ne, device=DEV).index_add_(0, inv, rt * rt)
    s0 = torch.zeros(ne, device=DEV).index_add_(0, inv, has)
    ok = (cnt_e >= min_n) & (s0 >= 5)
    if not ok.any():
        return r.sum() * 0.0
    i_e = cnt_e[ok] / s0[ok] * num[ok] / (den[ok] + 1e-6)
    return F.relu(i_e).mean()


@torch.no_grad()
def predict(model, G, keep_alpha=False):
    """Class probabilities of every node, one closed per-event graph at a time."""
    model.eval()
    out = torch.zeros(G.n, 4, device=DEV)
    alphas = {}
    for e in np.unique(G.event_np):
        nodes = torch.from_numpy(np.flatnonzero(G.event_np == e)).to(DEV)
        lo, _ = sub_forward(model, G, nodes, keep_alpha=keep_alpha)
        out[nodes] = torch.softmax(lo.float(), 1)
        if keep_alpha and model.kind != "patch":
            alphas[int(e)] = (nodes.cpu().numpy(), [l.alpha.mean(-1).cpu().numpy() for l in model.layers])
    model.train()
    return (out.cpu().numpy(), alphas) if keep_alpha else out.cpu().numpy()


def residual_moran(G, probs):
    """Mean over events of Moran's I of the severity residuals y - E[y] on the event k-NN graph
    (row-normalised W), and the pooled ordinal MAE of the argmax."""
    r = G.y_np - probs @ np.arange(4)
    vals = []
    for e in np.unique(G.event_np):
        idx = np.flatnonzero(G.event_np == e)
        if len(idx) < 10:
            continue
        loc = np.full(G.n, -1)
        loc[idx] = np.arange(len(idx))
        nb = loc[np.maximum(G.nbr_np[idx], 0)]
        ok = (G.nbr_np[idx] >= 0) & (nb >= 0)
        rt = r[idx] - r[idx].mean()
        lag = (rt[np.maximum(nb, 0)] * ok).sum(1) / np.maximum(ok.sum(1), 1)
        vals.append(float((rt * lag).sum() / ((rt ** 2).sum() + 1e-9)))
    return float(np.mean(vals))


def scores(G, probs, disasters):
    from bda_metrics import building_metrics
    m = building_metrics(G.y_np, probs, disasters)["overall"]
    m["severity_mae"] = float(np.abs(probs.argmax(1) - G.y_np).mean())
    m["residual_moran"] = residual_moran(G, probs)
    return m


# ------------------------------------------------------------------------------------------ training
def train_run(kind, seed, Gtr, Gva, d_in, args, dis_va, w):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(kind, d_in, lam_k=args.lam_k, p_in=args.p_in).to(DEV)
    prior = [p for n, p in model.named_parameters() if n.startswith(("kernel.", "z."))]
    rest = [p for n, p in model.named_parameters() if not n.startswith(("kernel.", "z."))]
    groups = [{"params": rest, "lr": args.lr, "weight_decay": args.wd}]
    if prior:  # the low-dimensional DCMS prior learns at kernel_lr_mult x lr, without weight decay
        groups.append({"params": prior, "lr": args.lr * args.kernel_lr_mult, "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    n_tiles = len(Gtr.tile_ranges)
    steps = int(np.ceil(n_tiles / args.tiles))
    total, warm = args.epochs * steps, steps
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: min(1.0, (t + 1) / warm) * 0.5 * (1 + np.cos(np.pi * min(1.0, t / total))))
    pos_in_core = torch.full((Gtr.n,), -1, dtype=torch.long, device=DEV)
    best, best_ep, best_state, log, t0 = -1.0, 0, None, [], time.time()
    hops = len(model.layers) if kind != "patch" else 0
    for ep in range(args.epochs):
        perm = np.random.permutation(n_tiles)
        tot = {"loss": 0.0, "ce": 0.0, "emd": 0.0, "res": 0.0}
        for it in range(min(steps, args.max_steps or steps)):
            rng_ = Gtr.tile_ranges[perm[it * args.tiles:(it + 1) * args.tiles]]
            core = torch.from_numpy(np.concatenate([np.arange(a, b) for a, b in rng_])).to(DEV)
            nodes = closure(Gtr, core, hops) if hops else core
            logits, m = sub_forward(model, Gtr, nodes)
            lc = logits[m[core]]
            y = Gtr.y[core]
            loss, ce, emd = cls_loss(lc, y, w, args.lam_emd)
            res = torch.zeros((), device=DEV)
            if kind == "krgat" and args.lam_res > 0:
                pos_in_core[core] = torch.arange(len(core), device=DEV)
                cn = Gtr.nbr[core]
                cnb = torch.where(cn >= 0, pos_in_core[cn.clamp(min=0)], -1)
                pos_in_core[core] = -1
                s_hat = torch.softmax(lc.float(), 1) @ torch.arange(4.0, device=DEV)
                res = moran_loss(y.float() - s_hat, Gtr.event[core], cnb)
                loss = loss + args.lam_res * res
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            for k, v in (("loss", loss), ("ce", ce), ("emd", emd), ("res", res)):
                tot[k] += float(v) / steps
        p = predict(model, Gva)
        sc = scores(Gva, p, dis_va)
        log.append({"epoch": ep + 1, **tot, **sc})
        if sc["macro_f1"] > best:
            best, best_ep = sc["macro_f1"], ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if args.patience and ep - best_ep >= args.patience:  # (CPU fallback only; the schedule is unchanged)
            print(f"    {kind} s{seed}: stop at epoch {ep + 1}, no val gain for {args.patience} epochs", flush=True)
            break
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"    {kind} s{seed} ep {ep + 1}: loss {tot['loss']:.3f} res {tot['res']:.3f} | val macro-F1 "
                  f"{sc['macro_f1']:.4f} QWK {sc['qwk']:.3f} Moran {sc['residual_moran']:.3f} ({time.time() - t0:.0f}s)",
                  flush=True)
    model.load_state_dict(best_state)
    return model, pd.DataFrame(log)


def stage_train(args):
    g, X = load_inputs(args.feat)
    Gtr, Gva = SplitGraph(g, 0, X), SplitGraph(g, 1, X)
    events = g["events"]
    dis_va = events[Gva.event_np]
    w = class_weights(Gtr.y_np, args.beta).to(DEV)
    print(f"[{args.feat}] train {Gtr.n} nodes / {len(Gtr.tile_ranges)} tiles, val {Gva.n}; class weights "
          f"{w.cpu().numpy().round(3).tolist()}", flush=True)
    rows = []
    for kind in args.kinds:
        for seed in args.seeds:
            out = f"{ROOT}/results/{args.feat}/{kind}_s{seed}"
            os.makedirs(out, exist_ok=True)
            model, log = train_run(kind, seed, Gtr, Gva, X.shape[1], args, dis_va, w)
            torch.save(model.state_dict(), f"{out}/best.pt")
            log.to_csv(f"{out}/val_log.csv", index=False)
            b = log.loc[log.macro_f1.idxmax()]
            rows.append({"feat": args.feat, "kind": kind, "seed": seed, "best_epoch": int(b.epoch),
                         **{k: b[k] for k in ("macro_f1", "f1_0", "f1_1", "f1_2", "f1_3", "qwk", "acc",
                                              "severity_mae", "residual_moran")}})
            print(f"  {kind} s{seed}: best val macro-F1 {b.macro_f1:.4f} at epoch {int(b.epoch)} "
                  f"(F1 {b.f1_0:.3f}/{b.f1_1:.3f}/{b.f1_2:.3f}/{b.f1_3:.3f}, QWK {b.qwk:.3f}, Moran {b.residual_moran:.3f})",
                  flush=True)
            with open(f"{out}/args.json", "w") as fh:
                json.dump(vars(args), fh)
    df = pd.DataFrame(rows)
    path = f"{ROOT}/results/{args.feat}/val_summary.csv"
    if os.path.exists(path):
        old = pd.read_csv(path)
        df = pd.concat([old[~old.set_index(["kind", "seed"]).index.isin(df.set_index(["kind", "seed"]).index)], df])
    df.to_csv(path, index=False)
    print(df.groupby("kind")[["macro_f1", "f1_1", "f1_2", "qwk", "residual_moran"]].agg(["mean", "std"]).round(4).to_string())


# ------------------------------------------------------------------------------------------ test + harness
def ego_union(G, centres, hops=2):
    """Disjoint union of the `hops`-hop in-neighbourhoods of the centres (split-local ids)."""
    nodes, nbrs, cpos, off = [], [], [], 0
    nb_all = G.nbr_np
    for c in centres:
        s = np.array([c])
        fr = s
        for _ in range(hops):
            fr = nb_all[fr].ravel()
            fr = np.unique(fr[fr >= 0])
            s = np.concatenate([s, fr])
        s = np.unique(s)
        nb = nb_all[s]
        pos = np.minimum(np.searchsorted(s, np.maximum(nb, 0)), len(s) - 1)
        ok = (nb >= 0) & (s[pos] == nb)
        nodes.append(s)
        nbrs.append(np.where(ok, pos + off, -1))
        cpos.append(off + int(np.searchsorted(s, c)))
        off += len(s)
    return np.concatenate(nodes), np.concatenate(nbrs), np.array(cpos)


@torch.no_grad()
def ego_predict(model, G, centres, centre_x=None, centre_rollup=None, batch=256):
    """Probabilities at the centres, each in its own ego graph; optionally with the centre's input
    features / disaster type replaced (centre_x: (n, d) tensor, centre_rollup: (n,) tensor)."""
    model.eval()
    hops = len(model.layers) if model.kind != "patch" else 0
    out = []
    for s in range(0, len(centres), batch):
        c = centres[s:s + batch]
        nodes, nbr, cpos = ego_union(G, c, hops)
        nodes_t, cpos_t = torch.from_numpy(nodes).to(DEV), torch.from_numpy(cpos).to(DEV)
        x = G.x[nodes_t].float()
        r = G.rollup[nodes_t].clone()
        if centre_x is not None:
            x[cpos_t] = centre_x[s:s + batch].float()
        if centre_rollup is not None:
            r[cpos_t] = centre_rollup[s:s + batch]
        lo = model(x, torch.from_numpy(nbr).to(DEV), G.u[nodes_t], G.d[nodes_t], r)
        out.append(torch.softmax(lo.float(), 1)[cpos_t].cpu().numpy())
    return np.concatenate(out)


def frozen_encoder(feat):
    """The frozen patch encoder of `feat` as a function uint8 (B, P, P, 6) tensor -> (B, d) float16
    array, computed exactly as the stored embeddings were (train_encoder.py)."""
    if feat == "ppc_r50":
        enc = PPCEncoder().to(DEV).to(memory_format=torch.channels_last)
        enc.load_state_dict(torch.load(f"{DATA}/encoder/best.pt", map_location=DEV))
        net = lambda x: enc(x)[1]
    else:
        import torchvision
        from common import R50
        enc = torchvision.models.resnet50()
        enc.load_state_dict(torch.load(R50, map_location="cpu"))
        enc.fc = nn.Identity()
        enc = enc.to(DEV).to(memory_format=torch.channels_last)
        cl = torch.channels_last
        net = lambda x: torch.cat([enc(x[:, :3].contiguous(memory_format=cl)), enc(x[:, 3:].contiguous(memory_format=cl))], 1)
    enc.eval()

    def encode(arr):
        fs = []
        with torch.no_grad():
            for s in range(0, len(arr), 512):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    f = net(to_input(torch.from_numpy(arr[s:s + 512])))
                fs.append(f.float().cpu().numpy().astype(np.float16))
        return np.concatenate(fs)
    return encode


def inpainted_features(te, targets, args):
    """Frozen-encoder embeddings of the targets' PPC patches with the building (+4 px) inpainted."""
    from bda_data import polygons
    scenes = pd.read_csv(f"{PROJ}/data/scenes.csv", usecols=["scene_id"])
    mm = {k: np.memmap(f"{PROJ}/data/{k}.u8", np.uint8, "r", shape=(len(scenes), IMG, IMG, 3)) for k in ("pre", "post")}
    encode = frozen_encoder(args.feat)
    n_all = len(np.load(f"{DATA}/graph.npz")["split"])
    bank = np.memmap(f"{DATA}/ppc_{PATCH}.u8", np.uint8, "r", shape=(n_all, PATCH, PATCH, 6))
    fname, dim = FEATS[args.feat]
    stored = np.memmap(f"{DATA}/{fname}", np.float16, "r", shape=(n_all, dim))
    polys = polygons(te.wkt_pre.to_numpy()[targets])
    boxes = te[["minx", "miny", "maxx", "maxy"]].to_numpy(np.float64)[targets]
    patches, plain = np.zeros((len(targets), PATCH, PATCH, 6), np.uint8), np.zeros((len(targets), PATCH, PATCH, 6), np.uint8)
    sid = te.scene_id.to_numpy()[targets]
    for s in np.unique(sid):
        pre, post = np.ascontiguousarray(mm["pre"][s]), np.ascontiguousarray(mm["post"][s])
        for k in np.flatnonzero(sid == s):
            box = ppc_box(*boxes[k])
            patches[k] = inpainted_ppc_patch(pre, post, polys[k], box)
            plain[k] = ppc_patch(pre, post, box)
    gid = args.test_global[targets]
    inv = np.argsort(np.argsort(gid))
    assert (plain == bank[np.sort(gid)][inv]).all(), "PPC crop differs from the bank"
    check = encode(plain[:256])
    ref = np.asarray(stored[np.sort(gid[:256])])[np.argsort(np.argsort(gid[:256]))].astype(np.float32)
    print(f"  re-encoded vs stored embeddings: max abs diff {np.abs(check.astype(np.float32) - ref).max():.4f} "
          f"(mean |f| {np.abs(ref).mean():.3f})", flush=True)
    np.savez_compressed(f"{ROOT}/results/{args.feat}/inpainted_targets.npz", building_id=te.building_id.to_numpy()[targets],
                        patches=patches[:64], plain=plain[:64])
    return encode(patches)


def harness_tables(te, probs, cat):
    from bda_metrics import building_metrics
    y = te.label.to_numpy()
    pred = probs.argmax(1)
    m = building_metrics(y, probs, te.disaster.to_numpy())["overall"]
    rows = [{"stratum": "all", "n": len(y), **m}]
    for c in pd.unique(cat):
        s = cat == c
        rows.append({"stratum": c, "n": int(s.sum()), "acc": float((pred[s] == y[s]).mean()),
                     **{f"missed_{LEVELS[l]}": float((pred[s & (y == l)] == 0).mean()) for l in (1, 2, 3)
                        if (s & (y == l)).any()}})
    return pd.DataFrame(rows), m


def stage_test(args):
    """Best-on-hold seed per model -> test predictions in harness format, plus the counterfactual on
    the targets and donors of analysis/context_swap.py (seed 0; isolated damage / isolated intact all,
    1500 concordant damage / intact). Share predicted damaged (argmax >= minor) for:
      original         the target in its own test graph
      intact context   the target node (its embedding, area/perimeter and disaster token) moved into
                       the place of a donor building of an all-intact neighbourhood: every neighbour
                       (1 and 2 hops, with the donor's geometry) is the donor's
      damaged context  the same with a donor of an all-major/destroyed neighbourhood
      building gone    the target's embedding replaced by that of its PPC patch with the footprint
                       (+4 px) inpainted (Telea, as harness.py); neighbours unchanged
      building gone (neighbour-mean feature)  ... replaced by the mean embedding of its 16 neighbours
    """
    from bda_data import load_tables
    from discordant import categories
    g, X = load_inputs(args.feat)
    Gva, Gte = SplitGraph(g, 1, X), SplitGraph(g, 2, X)
    events = g["events"]
    scenes, b = load_tables(f"{PROJ}/data")
    b = b.merge(scenes[["scene_id", "disaster_type"]], on="scene_id")
    te = b[(b.official == "test") & (b.label >= 0)].reset_index(drop=True)
    assert (g["building_id"][Gte.idx] == te.building_id.to_numpy()).all(), "test node order != harness order"
    args.test_global = Gte.idx
    cat = categories(te, 64)
    y = te.label.to_numpy()
    summ = pd.read_csv(f"{ROOT}/results/{args.feat}/val_summary.csv")
    # counterfactual targets and donors exactly as analysis/context_swap.py (seed 0)
    rng = np.random.RandomState(0)
    groups = {"isolated damage": np.flatnonzero(cat == "isolated damage"),
              "concordant damage": rng.permutation(np.flatnonzero((cat == "concordant") & (y >= 1)))[:args.n_concordant],
              "isolated intact": np.flatnonzero(cat == "isolated intact"),
              "concordant intact": rng.permutation(np.flatnonzero((cat == "concordant") & (y == 0)))[:args.n_concordant]}
    pools = {"intact context": np.flatnonzero((cat == "concordant") & (y == 0)),
             "damaged context": np.flatnonzero((cat == "concordant") & (y >= 2))}
    donors = {gname: {v: [] for v in pools} for gname in groups}
    for gname, idx in groups.items():
        for _ in idx:
            for v in pools:
                donors[gname][v].append(int(rng.choice(pools[v])))
    targets = np.concatenate(list(groups.values()))
    gone_feat = None
    if args.cf:
        t0 = time.time()
        gone_feat = np.random.RandomState(1).normal(size=(len(targets), X.shape[1] - 2)).astype(np.float16) if FAKE \
            else inpainted_features(te, targets, args)
        print(f"  inpainted embeddings for {len(targets)} targets ({time.time() - t0:.0f}s)", flush=True)
        comp = []
        for v, lab_ok in (("intact context", lambda l: l == 0), ("damaged context", lambda l: l >= 2)):
            dd = np.concatenate([donors[gn][v] for gn in groups])
            nb = Gte.nbr_np[dd]
            comp.append(f"{v}: donors' graph neighbours with the donor-pool level {lab_ok(Gte.y_np[nb]).mean():.3f}, "
                        f"any damage {(Gte.y_np[nb] >= 1).mean():.3f}")
        print("  " + "; ".join(comp), flush=True)
    meta_te = Gte.x[:, -2:]
    rows_all = []
    for kind in args.kinds:
        sk = summ[summ.kind == kind].sort_values("macro_f1", ascending=False)
        seed = int(sk.iloc[0].seed)
        model = Net(kind, X.shape[1], lam_k=args.lam_k, p_in=args.p_in).to(DEV)
        model.load_state_dict(torch.load(f"{ROOT}/results/{args.feat}/{kind}_s{seed}/best.pt", map_location=DEV))
        if kind == "krgat":
            with torch.no_grad():
                pi, rho = model.kernel.params(model.z.weight)
            kt = pd.DataFrame({"rollup": ROLLUPS, **{f"pi_{k}": pi[:, k].cpu().numpy() for k in range(pi.shape[1])},
                               **{f"rho_{k}_m": rho[:, k].cpu().numpy() for k in range(rho.shape[1])}})
            kt.to_csv(f"{ROOT}/results/{args.feat}/{kind}_s{seed}/learned_kernels.csv", index=False)
            print("learned DCMS kernels (mixture weights, length scales in m):\n" + kt.round(3).to_string(index=False), flush=True)
        pva = predict(model, Gva)
        sva = scores(Gva, pva, events[Gva.event_np])
        pte = predict(model, Gte)
        ste = scores(Gte, pte, events[Gte.event_np])
        name = HARNESS_NAME[kind] + ("" if args.feat == "ppc_r50" else f"_{args.feat}")
        out = f"{args.harness_root}/{name}" if args.feat == "ppc_r50" else f"{ROOT}/results/{args.feat}/harness/{name}"
        os.makedirs(out, exist_ok=True)
        np.savez_compressed(f"{out}/harness_preds.npz", building_id=te.building_id.to_numpy(), label=y, probs=pte.astype(np.float32))
        std, m = harness_tables(te, pte, cat)
        std.to_csv(f"{out}/harness_standard.csv", index=False)
        np.savez_compressed(f"{ROOT}/results/{args.feat}/{kind}_s{seed}/val_preds.npz",
                            building_id=g["building_id"][Gva.idx], label=Gva.y_np, probs=pva.astype(np.float32))
        print(f"{name} (seed {seed}): hold macro-F1 {sva['macro_f1']:.4f} | test macro-F1 {m['macro_f1']:.4f} QWK {m['qwk']:.4f}",
              flush=True)
        print(std[["stratum", "n", "acc", "missed_minor", "missed_major", "missed_destroyed"]].round(3).to_string(index=False),
              flush=True)
        rows_all.append({"model": name, "seed": seed, **{f"hold_{k}": v for k, v in sva.items()},
                         **{f"test_{k}": v for k, v in ste.items()}})
        if args.cf:
            rows = []
            pos = 0
            for gname, idx in groups.items():
                n_ = len(idx)
                base = ego_predict(model, Gte, idx)
                assert np.abs(base - pte[idx]).max() < 1e-3, "ego-graph prediction differs from the full graph"
                r = {"targets": gname, "n": n_, "original": float((pte[idx].argmax(1) >= 1).mean())}
                cx = Gte.x[torch.from_numpy(idx).to(DEV)]
                cr = Gte.rollup[torch.from_numpy(idx).to(DEV)]
                for v in pools:
                    p = ego_predict(model, Gte, np.array(donors[gname][v]), centre_x=cx, centre_rollup=cr)
                    r[v] = float((p.argmax(1) >= 1).mean())
                gf = torch.from_numpy(gone_feat[pos:pos + n_]).to(DEV)
                p = ego_predict(model, Gte, idx, centre_x=torch.cat([gf, meta_te[idx]], 1))
                r["building gone"] = float((p.argmax(1) >= 1).mean())
                nb = Gte.nbr[torch.from_numpy(idx).to(DEV)]
                nbm = (Gte.x[nb.clamp(min=0)][..., :-2].float() * (nb >= 0)[..., None]).sum(1) / (nb >= 0).sum(1, keepdim=True)
                p = ego_predict(model, Gte, idx, centre_x=torch.cat([nbm, meta_te[idx].float()], 1))
                r["building gone (neighbour-mean feature)"] = float((p.argmax(1) >= 1).mean())
                rows.append(r)
                pos += n_
            cf = pd.DataFrame(rows)
            cf.to_csv(f"{out}/harness_counterfactual.csv", index=False)
            print(cf.round(3).to_string(index=False), flush=True)
    path = f"{ROOT}/results/{args.feat}/hold_test_summary.csv"
    df = pd.DataFrame(rows_all)
    if os.path.exists(path):
        old_df = pd.read_csv(path)
        df = pd.concat([old_df[~old_df.model.isin(df.model)], df])
    df.to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "test"], required=True)
    ap.add_argument("--feat", choices=list(FEATS), default="ppc_r50")
    ap.add_argument("--kinds", nargs="+", default=["patch", "gat", "krgat"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--tiles", type=int, default=48, help="training tiles per mini-batch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--p-in", type=float, default=0.2)
    ap.add_argument("--beta", type=float, default=0.9999)
    ap.add_argument("--lam-emd", type=float, default=0.25)
    ap.add_argument("--lam-res", type=float, default=0.1)
    ap.add_argument("--lam-k", type=float, default=1.0)
    ap.add_argument("--kernel-lr-mult", type=float, default=10.0)
    ap.add_argument("--n-concordant", type=int, default=1500)
    ap.add_argument("--no-cf", dest="cf", action="store_false")
    ap.add_argument("--max-steps", type=int, default=0, help="smoke test: steps per epoch")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many epochs without val gain (0 = off)")
    ap.add_argument("--harness-root", default=f"{PROJ}/results/harness")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = True
    if args.stage == "train":
        stage_train(args)
    else:
        stage_test(args)


if __name__ == "__main__":
    main()
