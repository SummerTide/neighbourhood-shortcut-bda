"""Our crop classifiers (train_bda.py) behind the harness interface, so that the tile-level
counterfactual test (paste into another neighbourhood, inpaint) is the same for every model.

A building model sets needs_footprints and maps a pre/post tile pair plus footprints (lists of
polygon parts in tile pixels) to (n, 4) probabilities; crops are cut exactly as in training
(bda_data.BuildingDataset, the run's context_px / tight_margin).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from shapely.geometry import MultiPolygon, Polygon

from tile_models import register

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RN50 = "external/checkpoints/resnet50-11ad3fa6.pth"


class CropModel:
    needs_footprints = True

    def __init__(self, run, inputs, backbone="resnet50"):
        from bda_data import CONTEXT_PX
        from train_bda import build_model
        run = os.path.join(PROJ, "results", run)
        trained = json.load(open(os.path.join(run, "args.json")))
        self.args = argparse.Namespace(inputs=inputs, backbone=backbone, weights=RN50,
                                       hub_dir="external/torch_hub",
                                       dinov3_repo="external/dinov3",
                                       context_px=trained.get("context_px", CONTEXT_PX),
                                       tight_margin=trained.get("tight_margin"),
                                       two_stream=trained.get("two_stream", False),
                                       separate_pre=trained.get("separate_pre", False),
                                       tight_view_px=trained.get("tight_view_px", 8),
                                       tight_view_margin=trained.get("tight_view_margin", 4.0))
        assert trained.get("inputs", inputs) == inputs and trained.get("backbone", backbone) == backbone, trained
        self.model = build_model(self.args).eval()
        self.model.load_state_dict(torch.load(os.path.join(run, "best.pt"), map_location="cuda"))

    @torch.inference_mode()
    def predict_buildings(self, pre, post, polys):
        from bda_data import BuildingDataset
        from train_bda import crop_args, model_inputs
        geoms = [MultiPolygon([Polygon(p) for p in parts]) if len(parts) > 1 else Polygon(parts[0]) for parts in polys]
        b = pd.DataFrame({"scene_id": 0, "wkt_pre": [g.wkt for g in geoms], "label": 0,
                          **dict(zip(["minx", "miny", "maxx", "maxy"], np.array([g.bounds for g in geoms]).T))})
        ds = BuildingDataset(b, [post], [pre] if self.args.inputs == "pre+post" else None, **crop_args(self.args))
        out = []
        for s in range(0, len(ds), 256):
            items = [ds[i] for i in range(s, min(s + 256, len(ds)))]
            batch = {k: torch.stack([it[k] for it in items]) for k in items[0] if k not in ("label", "index")}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out.append(F.softmax(self.model(*model_inputs(batch, False)).float(), 1).cpu().numpy())
        return np.concatenate(out)


def _crop(name, run, inputs, backbone="resnet50"):
    @register(name)
    class Crop(CropModel):
        __doc__ = f"train_bda.py run results/{run} ({inputs}, {backbone})"

        def __init__(self):
            super().__init__(run, inputs, backbone)
    return Crop


def _discover():
    """crop_<run> for every train_bda.py run in results/ (best.pt + args.json, 4 xBD levels)."""
    import glob
    out = []
    for f in sorted(glob.glob(os.path.join(PROJ, "results", "*", "best.pt"))):
        run = os.path.basename(os.path.dirname(f))
        meta = os.path.join(os.path.dirname(f), "args.json")
        if run.startswith(("joint_", "dm3")) or not os.path.exists(meta):
            continue
        a = json.load(open(meta))
        if "inputs" in a and "dm3_footprints" not in a:
            out.append((f"crop_{run}", run, a["inputs"], a.get("backbone", "resnet50")))
    return out


for _name, _run, _inputs, _backbone in _discover() + [
        ("crop_rn50_post", "rn50_post_none", "post", "resnet50"),
        ("crop_rn50_pre_post", "rn50_pre_post_none", "pre+post", "resnet50"),
        ("crop_rn50_post_cswap", "rn50_post_cswap", "post", "resnet50"),
        ("crop_rn50_pre_post_cswap", "rn50_pre_post_cswap", "pre+post", "resnet50"),
        ("crop_rn50_post_preswap", "rn50_post_preswap", "post", "resnet50"),
        ("crop_rn50_pre_post_preswap", "rn50_pre_post_preswap", "pre+post", "resnet50"),
        ("crop_rn50_post_tight", "rn50_post_tight", "post", "resnet50"),
        ("crop_rn50_pre_post_tight", "rn50_pre_post_tight", "pre+post", "resnet50"),
        ("crop_dinov3ft_post", "dinov3ft_post", "post", "dinov3_vitl16_sat"),
        ("crop_dinov3ft_post_cswap", "dinov3ft_post_cswap", "post", "dinov3_vitl16_sat"),
        ("crop_dinov3ft_pre_post", "dinov3ft_pre_post", "pre+post", "dinov3_vitl16_sat"),
        ("crop_dinov3ft_pre_post_cswap", "dinov3ft_pre_post_cswap", "pre+post", "dinov3_vitl16_sat")]:
    _crop(_name, _run, _inputs, _backbone)
