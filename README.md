# Neighbourhood shortcuts in building damage assessment

Code accompanying the manuscript *Neighbourhood shortcuts in building damage assessment from very-high-resolution satellite imagery: diagnosis and a building-evidence plug-in*.

**Authors:** JianChong Guo, YuTing Wan, Kefan Li, AiLong Ma, and YanFei Zhong.

**Publication status:** Unpublished manuscript (2026); not yet accepted for publication. An arXiv link and updated citation will be added when available.

## Citation

If you use the evaluation protocol, building-evidence plug-in, or experimental code in academic work, please cite the manuscript below. Please also cite the original datasets and baseline methods used in your experiments.

```bibtex
@unpublished{guo2026neighbourhood,
  title  = {Neighbourhood shortcuts in building damage assessment from very-high-resolution satellite imagery: diagnosis and a building-evidence plug-in},
  author = {Guo, JianChong and Wan, YuTing and Li, Kefan and Ma, AiLong and Zhong, YanFei},
  year   = {2026},
  note   = {Unpublished manuscript},
  url    = {https://github.com/SummerTide/neighbourhood-shortcut-bda}
}
```

The repository URL above identifies the accompanying code. This is a provisional manuscript citation; it does not indicate journal acceptance or an assigned arXiv identifier.

## Overview

The repository contains three parts.

| Part | What it does | Main files |
|---|---|---|
| Evaluation protocol | Scores any damage model per building on the same footprints, stratifies buildings by their agreement with their neighbours and runs the counterfactual relocation tests | `harness.py`, `analysis/discordant.py`, `analysis/cf_by_type.py` |
| Building-evidence plug-in | Trains the tight-crop and the swap-trained expert and combines them with an existing model as a product of experts, with the existing model alone on floods | `train_bda.py`, `bda_data.py`, `analysis/paper_tables.py` |
| Experiments | Wrappers of the nine diagnosed models, BRIGHT, xFBD and the İslahiye scene, and the scripts that produce the tables | `tile_models.py`, `tile_models_ext/`, `baselines/`, `analysis/` |

## Setup

Python 3.10 with PyTorch 2 and CUDA. Install the packages with

```
pip install -r requirements.txt
```

The wrapped models use their official code and weights, which are cloned into `external/`.

| Model | Repository | Weights |
|---|---|---|
| ChangeOS R101 | https://github.com/Z-Zheng/ChangeOS | `external/checkpoints/changeos_r101_official.pt` |
| ChangeOS Swin-T, ChangeStar (Changen2) | https://github.com/Z-Zheng/pytorch-change-models | Hugging Face hub, released by the authors |
| xView2 first place | https://github.com/DIUx-xView/xView2_first_place | released by the authors |
| Strong Baseline | https://github.com/PaulBorneP/Xview2_Strong_Baseline | released by the authors |
| DisasterAdaptiveNet | https://github.com/SebastianHafner/DisasterAdaptiveNet | released by the authors |
| MambaBDA-S | https://github.com/ChenHongruixuan/ChangeMamba | trained with `baselines/changemamba/` |
| Damage-TriageFormer | https://github.com/YimingXiao98/Damage-TriageFormer | trained with `baselines/damage_triageformer/` |
| KR-GAT | reimplemented in `baselines/krgat/` | trained with `baselines/krgat/` |
| BRIGHT benchmark models | https://github.com/ChenHongruixuan/BRIGHT | released by the authors |
| DINOv3 experts | https://github.com/facebookresearch/dinov3 | DINOv3-SAT ViT-L/16, from Meta |

The ResNet-50 experts start from the torchvision ImageNet weights `resnet50-11ad3fa6.pth` in `external/checkpoints/`.

## Data

| Dataset | Source | Preparation |
|---|---|---|
| xBD | https://xview2.org | `python prep_official.py --root datasets/xBD --out data` |
| xFBD | Melamed et al., arXiv:2212.13876 | `python prep_xfbd.py --xfbd <xfbd> --split test --out data_xfbd_test` |
| BRIGHT | https://github.com/ChenHongruixuan/BRIGHT | `python prep_bright.py --root <bright> --splits external/BRIGHT/bda_benchmark/dataset/splitname/standard_ML --out data_bright` |
| İslahiye, Türkiye, 2023 | Maxar Open Data Program, Turkey and Syria Earthquake 2023, ARD RGB visual product, CC BY-NC 4.0. Building instances from BRIGHT | `python prep_turkey.py --out data_turkey`, then `python register_turkey.py --data data_turkey --out data_turkey_reg` |

`prep_official.py` writes the building tables and two memory-mapped image files, `data/pre.u8` and `data/post.u8`, of about 35 GB each. The job scripts copy them to `/dev/shm` for training.

## Usage

Score an existing model, with the counterfactual tests on 500 targets per direction

```
python harness.py --model changeos_r101 --targets 500 --placement vacant --out results/harness/changeos_r101
```

Train the two experts

```
# tight-crop expert
python train_bda.py --data data --inputs pre+post --context-px 8 --tight-margin 4 --out results/rn50_pre_post_tight
# swap-trained expert
python train_bda.py --data data --inputs pre+post --context-swap donor --swap-blend poisson --swap-margin 2,12 \
    --swap-donor event --discordance-weight 5 --out results/rn50_pre_post_pswap_dw5
```

and score them with `python harness.py --model crop_<run> --targets 0 --out results/harness/crop_<run>`.

Combine the experts with every existing model and write the tables of the paper

```
python analysis/paper_tables.py --out tables
python analysis/cf_plugin.py --placement vacant --tex tables/tab_cf_plugin.tex
python analysis/turkey_paper.py
```

`scripts/` holds the shell scripts used for the paper. Each run used one NVIDIA A100 GPU with 40 GB of memory, and the image files are copied to `/dev/shm` first, which needs about 70 GB of host memory. Set `PROJ`, `H_HOME` and the conda environment at the top of each script.

## Licence

The code is released under the MIT licence. The datasets and the wrapped models keep their own licences. The İslahiye imagery is © Maxar Technologies, released under CC BY-NC 4.0 through the Maxar Open Data Program, and may be used for non-commercial purposes only.
