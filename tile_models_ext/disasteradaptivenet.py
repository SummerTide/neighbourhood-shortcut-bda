"""Strong Baseline and DisasterAdaptiveNet (Hafner, Gerard et al., JAG 2025), retrained with the
repo's strong_baseline_xview2 / disasteradaptivenet_xview2 configs: 40 epochs on 90 % of
train + tier3, checkpoint with the best validation loss on a stratified 10 % of train + tier3
(baselines/dan_work, driver scripts/train_dan.py).

Inference as in the repo's create_submission.py: BGR (it reads with cv2), x / 127 - 1, pre and
post stacked to 6 channels, full 1024 tiles, fp32, sigmoid outputs averaged over the identity and
three flips (INFERENCE.USE_ALL_FLIPS). Output channel 0 is building presence; channels 1-4 are
one-vs-rest sigmoids for no / minor / major / destroyed, returned as they are (the repo's damage
map is their argmax). DisasterAdaptiveNet also conditions the post-image decoder features (FiLM)
on the event's hazard: the repo's DATASET.EVENT_CONDITIONING by event name (tsunamis as
earthquakes, Florence and Harvey as floods, Matthew, Michael and the tornadoes as storms), with
the xBD disaster type as a fallback for events outside that table.
"""
import importlib
import os
import sys
import types

import numpy as np
import torch

from tile_models import register

REPO = "external/DisasterAdaptiveNet"
OUTPUT = "external/dan_work/output"
TYPE_TO_HAZARD = {"fire": "wildfire", "volcano": "volcano", "earthquake": "earthquake", "tsunami": "earthquake",
                  "wind": "storm", "flooding": "flood"}


def _repo_modules():
    """The repo's utils.models and utils.experiment_manager, imported without clashing with any
    other top-level 'utils' package already loaded in the process."""
    saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k == "utils" or k.startswith("utils.")}
    sys.path.insert(0, REPO)
    try:
        models = importlib.import_module("utils.models")
        experiment_manager = importlib.import_module("utils.experiment_manager")
    finally:
        sys.path.remove(REPO)
        for k in [k for k in sys.modules if k == "utils" or k.startswith("utils.")]:
            del sys.modules[k]
        sys.modules.update(saved)
    return models, experiment_manager


class _DisasterAdaptiveNetRepo:
    CONFIG = None
    RUN = None  # checkpoint name when it differs from the config name

    def __init__(self, weights=None, tta=True):
        models, experiment_manager = _repo_modules()
        cfg = experiment_manager.new_config()
        cfg.merge_from_file(os.path.join(REPO, "configs", f"{self.CONFIG}.yaml"))
        cfg.NAME = self.CONFIG
        self.cfg = cfg
        self.conditioned = bool(cfg.DATASET.INCLUDE_CONDITIONING_INFORMATION)
        self.tta = tta and bool(cfg.INFERENCE.USE_ALL_FLIPS)
        weights = weights or os.path.join(OUTPUT, "networks", f"{self.RUN or self.CONFIG}.pt")
        # every weight comes from the checkpoint: skip the ImageNet download of the encoder init
        models.ResNet34_Weights = types.SimpleNamespace(IMAGENET1K_V1=None)
        net = models.create_network(cfg)  # nn.DataParallel, matching the checkpoint's keys
        checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
        net.load_state_dict(checkpoint["network"])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = net.module.eval().to(self.device)
        print(f"{self.RUN or self.CONFIG}: {weights} (epoch {checkpoint['epoch']}), flips {self.tta}", flush=True)

    def hazard_id(self, disaster, disaster_type):
        hazard = self.cfg.DATASET.EVENT_CONDITIONING.get(disaster) or TYPE_TO_HAZARD[disaster_type]
        return int(self.cfg.DATASET.CONDITIONING_KEY[hazard])

    @torch.inference_mode()
    def __call__(self, pre, post, disaster=None, disaster_type=None):
        x = np.concatenate([pre[..., ::-1], post[..., ::-1]], 2).astype(np.float32) / 127 - 1
        x = torch.from_numpy(np.ascontiguousarray(x)).to(self.device).permute(2, 0, 1)
        flips = [None, [1], [2], [1, 2]] if self.tta else [None]
        inp = torch.stack([x if f is None else torch.flip(x, dims=f) for f in flips])
        if self.conditioned:
            cond = torch.full((len(flips),), self.hazard_id(disaster, disaster_type), dtype=torch.long,
                              device=inp.device)
            y = torch.sigmoid(self.net(inp, cond))
        else:
            y = torch.sigmoid(self.net(inp))
        y = torch.stack([y[k] if f is None else torch.flip(y[k], dims=f) for k, f in enumerate(flips)]).mean(0)
        return y[1:5].float().cpu().numpy()


@register("strong_baseline")
class StrongBaseline(_DisasterAdaptiveNetRepo):
    """Strong Baseline (Siamese ResNet34 U-Net, xView2 split) from the DisasterAdaptiveNet repo."""

    CONFIG = "strong_baseline_xview2"


@register("disasteradaptivenet")
class DisasterAdaptiveNet(_DisasterAdaptiveNetRepo):
    """DisasterAdaptiveNet (Strong Baseline + hazard-type FiLM conditioning, xView2 split)."""

    CONFIG = "disasteradaptivenet_xview2"
    needs_meta = True


@register("strong_baseline_cf")
class StrongBaselineCF(_DisasterAdaptiveNetRepo):
    """Strong Baseline trained with counterfactual copy-paste of buildings into discordant
    neighbourhoods (Poisson-blended, same event) and x5 discordance tile sampling
    (baselines/dan_work/scripts/cf_aug.py); same recipe, selection and inference otherwise."""

    CONFIG = "strong_baseline_xview2"
    RUN = "strong_baseline_cf"
