"""MambaBDA-Small (ChangeMamba, Chen et al., TGRS 2024), retrained on xBD train + tier3 with the
published recipe and the checkpoint selected on hold (baselines/run_mambabda.sh).

Inference as in the repo's BDAInferer: RGB, ImageNet mean / std on the 0-255 scale, full
1024 tiles, fp32, no TTA; channels 1-4 of the damage head's softmax are the damage levels.
"""
import argparse
import glob
import sys

import numpy as np
import torch

from tile_models import register

REPO = "external/ChangeMamba"
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


@register("mambabda_small")
class MambaBDASmall:
    def __init__(self, checkpoint=None):
        sys.path.insert(0, REPO)
        from changedetection.checkpoints import extract_model_state_dict
        from changedetection.configs.config import get_config
        from changedetection.models.ChangeMambaBDA import ChangeMambaBDA
        from changedetection.script.script_utils import get_vssm_kwargs
        checkpoint = checkpoint or sorted(glob.glob(f"{REPO}/saved_models/xBD/MambaBDA_Small_*/best_model.pth"))[-1]
        config = get_config(argparse.Namespace(cfg=f"{REPO}/changedetection/configs/vssm1/vssm_small_224.yaml",
                                               opts=None))
        model = ChangeMambaBDA(output_building=2, output_damage=5, pretrained=None, **get_vssm_kwargs(config))
        # our own training checkpoint (it also stores numpy evaluation records, which the default
        # weights_only load refuses); the model weights must match exactly
        state = extract_model_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()}, strict=True)
        print(f"mambabda_small: {checkpoint}", flush=True)
        self.model = model.eval().cuda()
        self.mean, self.std = MEAN.cuda(), STD.cuda()

    @torch.inference_mode()
    def __call__(self, pre, post):
        pre, post = ((torch.from_numpy(np.ascontiguousarray(a)).cuda().permute(2, 0, 1)[None].float() - self.mean)
                     / self.std for a in (pre, post))
        _, dmg = self.model(pre, post)
        return torch.softmax(dmg[0].float(), 0)[1:5].cpu().numpy()
