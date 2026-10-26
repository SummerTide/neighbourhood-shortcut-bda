#!/usr/bin/env python
"""Score a trained crop model (train_bda.py best.pt) on any table + memmap set in harness.py's
format, e.g. the xFBD synthetic single-building test set (prep_xfbd.py).
Usage: python analysis/crop_eval.py --model results/rn50_post_none --inputs post \
           --data data_xfbd_test --split xfbd_test --out results/harness_xfbd/rn50_post
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bda_data import BuildingDataset, ImageSource  # noqa: E402
from discordant import categories  # noqa: E402
from harness import standard_table, test_buildings  # noqa: E402
from train_bda import build_model, crop_args, predict  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="results dir holding best.pt (and args.json)")
    ap.add_argument("--inputs", choices=["post", "pre+post"], required=True)
    ap.add_argument("--backbone", choices=["resnet50", "dinov3_vitl16_sat"], default="resnet50")
    ap.add_argument("--data", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", required=True)
    ap.add_argument("--weights", default="external/checkpoints/resnet50-11ad3fa6.pth")
    ap.add_argument("--hub-dir", default="external/torch_hub")
    ap.add_argument("--dinov3-repo", default="external/dinov3")
    ap.add_argument("--context-px", type=int, default=None, help="default: the model's training value")
    ap.add_argument("--tight-margin", type=float, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-dist-pasted", type=float, default=None,
                    help="xFBD: score only buildings within this distance (px) of the pasted one")
    args = ap.parse_args()
    trained = os.path.join(args.model, "args.json")
    if os.path.exists(trained):  # crop geometry as in training
        import json
        t = json.load(open(trained))
        args.context_px = t.get("context_px") if args.context_px is None else args.context_px
        args.tight_margin = t.get("tight_margin") if args.tight_margin is None else args.tight_margin
    if args.context_px is None:
        from bda_data import CONTEXT_PX
        args.context_px = CONTEXT_PX
    os.makedirs(args.out, exist_ok=True)
    scenes, te = test_buildings(args.data, args.split)
    if args.max_dist_pasted is not None:
        te = te[te.dist_pasted <= args.max_dist_pasted].reset_index(drop=True)
    post = ImageSource(os.path.join(args.data, "post.u8"), len(scenes))
    pre = ImageSource(os.path.join(args.data, "pre.u8"), len(scenes)) if args.inputs == "pre+post" else None
    model = build_model(args).eval()
    model.load_state_dict(torch.load(os.path.join(args.model, "best.pt"), map_location="cuda"))
    loader = torch.utils.data.DataLoader(BuildingDataset(te, post, pre, **crop_args(args)), batch_size=256,
                                         num_workers=args.workers)
    probs = predict(model, loader)
    np.savez_compressed(os.path.join(args.out, "harness_preds.npz"), building_id=te.building_id.to_numpy(),
                        label=te.label.to_numpy(), probs=probs)
    std, m = standard_table(te, probs, categories(te, 64))
    std.to_csv(os.path.join(args.out, "harness_standard.csv"), index=False)
    print(f"{args.model} on {args.data}/{args.split}: macro-F1 {m['macro_f1']:.3f}")
    print(std[["stratum", "n", "acc", "missed_major", "missed_destroyed"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
