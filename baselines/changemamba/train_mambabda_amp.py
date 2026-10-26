#!/usr/bin/env python
"""ChangeMamba's train_MambaBDA.py with bf16 autocast and TF32 matmuls, nothing else changed.

The repo trains in fp32: on a 40 GB A100 the published recipe (batch 16, 512 crops, 80k
iterations) runs at ~4 s per iteration (~90 h). Mixed precision keeps the recipe (data,
losses, optimiser, schedule, evaluation and selection) and only changes arithmetic precision.
EVAL_INTERVAL in the environment overrides the evaluation interval (smoke tests).
Usage: same arguments as ChangeMamba/changedetection/script/train_MambaBDA.py
"""
import os
import runpy
import sys

import torch

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ChangeMamba")
sys.path.insert(0, REPO)

from changedetection.engine import base as engine_base  # noqa: E402
from changedetection.tasks import bda as bda_task  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def autocast(fn):
    def wrapped(self, *args, **kwargs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return fn(self, *args, **kwargs)
    return wrapped


# losses inside train_step run in fp32 under autocast (cross_entropy, softmax are fp32 ops)
bda_task.BDATrainer.train_step = autocast(bda_task.BDATrainer.train_step)
bda_task.BDATrainer.evaluate_loader = autocast(bda_task.BDATrainer.evaluate_loader)

if os.environ.get("EVAL_INTERVAL"):
    _init = engine_base.BaseTrainer.__init__

    def init(self, args):
        _init(self, args)
        self.eval_interval = int(os.environ["EVAL_INTERVAL"])
    engine_base.BaseTrainer.__init__ = init

sys.argv[0] = os.path.join(REPO, "changedetection", "script", "train_MambaBDA.py")
os.chdir(os.path.join(REPO, "changedetection"))
runpy.run_path(sys.argv[0], run_name="__main__")
