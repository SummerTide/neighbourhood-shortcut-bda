"""Full-tile damage models behind one interface, for harness.py.

A model maps a pre/post pair of uint8 RGB tiles (1024 x 1024 x 3) to per-pixel
probabilities over the four xBD damage levels (no, minor, major, destroyed),
shape (4, H, W). Localisation outputs are ignored: buildings are scored inside
the ground-truth footprints, with the argmax restricted to the damage levels.
"""
import os

import numpy as np
import torch

MODELS = {}


def register(name):
    def wrap(cls):
        MODELS[name] = cls
        return cls
    return wrap


@register("changeos_r101")
class ChangeOSR101:
    """Official ChangeOS R101 TorchScript release (Zheng et al., RSE 2021)."""

    MEAN = torch.tensor([123.675, 116.28, 103.53] * 2).view(1, 6, 1, 1)
    STD = torch.tensor([58.395, 57.12, 57.375] * 2).view(1, 6, 1, 1)

    def __init__(self, weights="external/checkpoints/changeos_r101_official.pt"):
        self.model = torch.jit.load(weights, map_location="cuda").eval()
        self.mean, self.std = self.MEAN.cuda(), self.STD.cuda()

    @torch.inference_mode()
    def __call__(self, pre, post):
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate([pre, post], 2))).cuda().permute(2, 0, 1)[None].float()
        with torch.autocast("cuda", dtype=torch.float16):
            _, dmg = self.model((x - self.mean) / self.std)
        return torch.softmax(dmg[0].float(), 0)[1:5].cpu().numpy()


class _TorChange:
    """TorChange models: ImageNet-normalised pre/post stacked to 6 channels; the eval
    output's change_prediction holds softmax probabilities over (background, no, minor,
    major, destroyed). Weights come from the Hugging Face cache (set HF_HUB_OFFLINE=1 on
    compute nodes)."""

    MEAN = torch.tensor([0.485, 0.456, 0.406] * 2).view(1, 6, 1, 1) * 255
    STD = torch.tensor([0.229, 0.224, 0.225] * 2).view(1, 6, 1, 1) * 255

    def __init__(self):
        self.mean, self.std = self.MEAN.cuda(), self.STD.cuda()

    @torch.inference_mode()
    def __call__(self, pre, post):
        x = torch.from_numpy(np.ascontiguousarray(np.concatenate([pre, post], 2))).cuda().permute(2, 0, 1)[None].float()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = self.model((x - self.mean) / self.std)
        return out["change_prediction"][0, 1:5].float().cpu().numpy()


@register("changeos_swint")
class ChangeOSSwinT(_TorChange):
    """TorChange ChangeOS with a Swin-T encoder (checkpoint selected on the test split)."""

    def __init__(self, repo="EVER-Z/torchange_example_changeos_swint_on_xview2_best42k"):
        super().__init__()
        from torchange.models.changeos import ChangeOS
        self.model = ChangeOS.from_pretrained(repo).eval().cuda()


@register("changestar_changen2")
class ChangeStarChangen2(_TorChange):
    """Changen2-pretrained ChangeStar 1x256 (SAM ViT-B encoder) fine-tuned on xView2 (Zheng et al., TPAMI 2025)."""

    def __init__(self):
        super().__init__()
        from torchange.models.changen2 import s0_init_xView2_ft_s1c5_changestar_vitb_1x256
        self.model = s0_init_xView2_ft_s1c5_changestar_vitb_1x256().eval().cuda()


def _load_checkpoint(path):
    """weights_only load of checkpoints that also store numpy scalars (e.g. best_score)."""
    core = getattr(np, "_core", None) or np.core
    allow = [(core.multiarray.scalar, "numpy.core.multiarray.scalar"), np.dtype,
             (type(np.dtype(np.float64)), "numpy.dtype[float64]")]
    with torch.serialization.safe_globals(allow):
        return torch.load(path, map_location="cpu", weights_only=True)


@register("xview2_first")
class XView2First:
    """xView2 1st place (Durnov 2020): Siamese U-Nets on ResNet34, SE-ResNeXt50, DPN92 and SENet154,
    three folds each, averaged with flip TTA as in its predict*cls.py. Inputs are BGR (it reads
    with cv2) scaled to x / 127 - 1; channels 1-4 of the softmax are the damage levels."""

    REPO = "external/xView2_first_place"
    SPECS = [("Res34_Unet_Double", "res34_cls2_{}_tuned_best", False),
             ("SeResNext50_Unet_Double", "res50_cls_cce_{}_tuned_best", None),
             ("Dpn92_Unet_Double", "dpn92_cls_cce_{}_tuned_best", None),
             ("SeNet154_Unet_Double", "se154_cls_cce_{}_tuned_best", None)]

    def __init__(self, seeds=(0, 1, 2), tta=True, weights_dir=None, families=None):
        import glob
        import sys
        sys.path.insert(0, self.REPO)
        from zoo import models as zoo
        weights_dir = weights_dir or os.path.dirname(glob.glob(f"{self.REPO}/weights_final/**/res34_cls2_0_tuned_best",
                                                               recursive=True)[0])
        self.models, self.tta = [], tta
        for cls, snap, pretrained in self.SPECS:
            if families and not any(cls.startswith(f) for f in families):
                continue
            for seed in seeds:
                model = getattr(zoo, cls)(pretrained=pretrained)
                state = _load_checkpoint(os.path.join(weights_dir, snap.format(seed)))["state_dict"]
                model.load_state_dict({k.replace("module.", "", 1): v for k, v in state.items()})
                self.models.append(model.eval().cuda())

    @torch.inference_mode()
    def __call__(self, pre, post):
        x = np.concatenate([pre[..., ::-1], post[..., ::-1]], 2).astype(np.float32) / 127 - 1
        x = torch.from_numpy(np.ascontiguousarray(x)).cuda().permute(2, 0, 1)[None]
        flips = [(), (2,), (3,), (2, 3)] if self.tta else [()]
        inp = torch.cat([x.flip(f) if f else x for f in flips])
        acc = 0
        for model in self.models:
            with torch.autocast("cuda", dtype=torch.float16):
                p = torch.softmax(model(inp).float(), 1)
            acc = acc + sum((p[k:k + 1].flip(f) if f else p[k:k + 1]) for k, f in enumerate(flips))
        return (acc[0, 1:5] / (len(self.models) * len(flips))).cpu().numpy()


def _family(name, prefix):
    @register(f"xview2_first_{name}")
    class Family(XView2First):
        __doc__ = f"{prefix} encoder of the 1st place only, three folds with TTA (average the four families' "\
                  "probabilities with harness.py --average for the full ensemble)."

        def __init__(self):
            super().__init__(families=(prefix,))
    return Family


for _name, _prefix in [("res34", "Res34"), ("res50", "SeResNext50"), ("dpn92", "Dpn92"), ("se154", "SeNet154")]:
    _family(_name, _prefix)


@register("xview2_first_light")
class XView2FirstLight(XView2First):
    """One fold per encoder and no TTA, for the counterfactual test."""

    def __init__(self):
        super().__init__(seeds=(0,), tta=False)


def _load_extensions():
    """Import tile_models_ext/*.py so that baseline wrappers kept in their own files register too."""
    import importlib
    import pkgutil
    import sys
    here = os.path.dirname(os.path.abspath(__file__))
    if here not in sys.path:
        sys.path.insert(0, here)
    for mod in pkgutil.iter_modules([os.path.join(here, "tile_models_ext")]):
        importlib.import_module(f"tile_models_ext.{mod.name}")


def build(name, **kwargs):
    return MODELS[name](**kwargs)


def available():
    return sorted(MODELS)


_load_extensions()


if __name__ == "__main__":
    print(available(), os.getcwd())
