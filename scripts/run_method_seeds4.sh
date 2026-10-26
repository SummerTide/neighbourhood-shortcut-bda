#!/bin/bash
# Seeds 1 and 2 of the ResNet-50 crop configurations whose differences are a few points of
# discordant balanced accuracy (single runs vary by ~3 points between seeds), one run per GPU
# of a full node: tight crop and v1 (Poisson event swap + discordance x5), post and pre+post.
# Then each model goes through harness.py (standard + hard and Poisson counterfactuals + xFBD).
set -euo pipefail
export H_HOME=${H_HOME:-$HOME}
PROJ=${PROJ:-$PWD}
exec >"$PROJ/logs/method_seeds_${PART}_${JOB_ID:-$$}.log" 2>&1
export OMP_NUM_THREADS=1
source "$H_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bda
export PYTHONPATH=${PYLIB:-}
export TORCH_HOME=$H_HOME/.cache/torch XDG_CACHE_HOME=$H_HOME/.cache MPLCONFIGDIR=$PROJ/.mplconfig
cd "$PROJ"
J=${JOB_ID:-$$}
echo "Job $J on $(hostname), $(nvidia-smi -L | wc -l) GPUs, $(date)"
SHM=/dev/shm/fbda_$J
mkdir -p "$SHM" && cp data/pre.u8 data/post.u8 "$SHM"/ && trap 'rm -rf "$SHM"' EXIT
echo "images staged at $(date)"
COMMON="--data data --image-dir $SHM --epochs 8 --workers 8"
SWAP="--context-swap donor --swap-blend poisson --swap-margin 2,12 --swap-donor event"
DW="--discordance-weight 5"
TIGHT="--context-px 8 --tight-margin 4"
NAMES=(rn50_post_tight_s1 rn50_post_tight_s2 rn50_pre_post_tight_s1 rn50_pre_post_tight_s2
       rn50_post_pswap_dw5_s1 rn50_post_pswap_dw5_s2 rn50_pre_post_pswap_dw5_s1 rn50_pre_post_pswap_dw5_s2)
OPTS=("--inputs post $TIGHT --seed 1" "--inputs post $TIGHT --seed 2" "--inputs pre+post $TIGHT --seed 1"
      "--inputs pre+post $TIGHT --seed 2" "--inputs post $SWAP $DW --seed 1" "--inputs post $SWAP $DW --seed 2"
      "--inputs pre+post $SWAP $DW --seed 1" "--inputs pre+post $SWAP $DW --seed 2")
# PART=a: runs 0-3 (tight crops), PART=b: runs 4-7 (v1), on four GPUs each
OFF=$([ "$PART" = b ] && echo 4 || echo 0)
NAMES=("${NAMES[@]:$OFF:4}"); OPTS=("${OPTS[@]:$OFF:4}")
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g python train_bda.py $COMMON ${OPTS[$g]} --out "results/${NAMES[$g]}" \
      >"logs/${NAMES[$g]}_$J.log" 2>&1 &
done
wait
echo "training done at $(date)"
for g in 0 1 2 3; do
  m=crop_${NAMES[$g]}
  (CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 500 --image-dir "$SHM" --out results/harness/$m &&
   CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 500 --image-dir "$SHM" --out results/harness/$m \
       --blend poisson --cf-only &&
   CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 0 --data data_xfbd_test --split xfbd_test \
       --max-dist-pasted 64 --out results/harness_xfbd/$m) >"logs/harness_${m}_$J.log" 2>&1 &
done
wait
echo "Job finished at $(date)"
