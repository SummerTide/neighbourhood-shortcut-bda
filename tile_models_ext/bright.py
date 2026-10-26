"""BRIGHT benchmark models (Chen et al., ESSD 2025) with the authors' released checkpoints for the
standard ML split (Zenodo 15349462), behind the harness interface.

Inference as in bda_benchmark/script/standard_ML/infer_*.py: pre-event optical RGB and post-event SAR
(one band copied to three channels), ImageNet mean / std on the 0-255 scale, full 1024 tiles, no TTA.
The damage head has four classes (background, intact, damaged, destroyed); channels 1-3 are the
three BRIGHT damage levels.
"""
import os
import sys

import numpy as np
import torch

from tile_models import register

REPO = "external/BRIGHT/bda_benchmark"
CKPT = "external/BRIGHT_ckpt/ckpt_{}_standard_ML_split.pth"
MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


class _Bright:
    ARCH = None
    SIAMESE = False      # model(pre, post) instead of model(cat(pre, post))
    TWO_OUTPUTS = False  # returns (localisation, damage)

    def __init__(self):
        sys.path.insert(0, REPO)
        self.model = self.build().eval().cuda()
        state = torch.load(CKPT.format(self.ARCH), map_location="cpu")
        self.model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        self.mean, self.std = MEAN.cuda(), STD.cuda()
        print(f"bright {self.ARCH}: {CKPT.format(self.ARCH)}", flush=True)

    @torch.inference_mode()
    def __call__(self, pre, post):
        a, b = ((torch.from_numpy(np.ascontiguousarray(v)).cuda().permute(2, 0, 1)[None].float() - self.mean) / self.std
                for v in (pre, post))
        out = self.model(a, b) if self.SIAMESE else self.model(torch.cat([a, b], 1))
        if self.TWO_OUTPUTS:
            out = out[1]
        return torch.softmax(out[0].float(), 0)[1:4].cpu().numpy()


@register("bright_damageformer")
class BrightDamageFormer(_Bright):
    ARCH, SIAMESE, TWO_OUTPUTS = "DamageFormer", True, True

    def build(self):
        from model.DamageFormer import DamageFormer
        return DamageFormer()


@register("bright_siamcrnn")
class BrightSiamCRNN(_Bright):
    ARCH, SIAMESE, TWO_OUTPUTS = "SiamCRNN", True, True

    def build(self):
        from model.SiamCRNN import SiamCRNN
        return SiamCRNN()


@register("bright_siamattnunet")
class BrightSiamAttnUNet(_Bright):
    ARCH, SIAMESE = "SiamAttnUNet", True

    def build(self):
        from model.SiamAttnUNet import SiamAttnUNet
        return SiamAttnUNet(in_channels=3, num_classes=4)


@register("bright_deeplabv3plus")
class BrightDeepLabV3Plus(_Bright):
    ARCH = "DeepLabV3Plus"

    def build(self):
        from model.DeepLabV3Plus import DeepLabV3Plus
        return DeepLabV3Plus(in_channels=6, num_classes=4)


@register("bright_unet")
class BrightUNet(_Bright):
    ARCH = "UNet"

    def build(self):
        from model.UNet import UNet
        return UNet(in_channels=6, num_classes=4)
