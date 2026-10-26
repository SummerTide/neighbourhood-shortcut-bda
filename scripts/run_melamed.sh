#!/bin/bash
# Melamed et al. (xFBD) augmentation on our ResNet-50 crop host: xBD train + the xFBD tier1
# synthetic single-building samples (Palu; pasted building and its neighbours within 64 px).
set -euo pipefail
export H_HOME=${H_HOME:-$HOME}
PROJ=${PROJ:-$PWD}
exec >"$PROJ/logs/melamed_${JOB_ID:-$$}.log" 2>&1
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
NAMES=(rn50_post_xfbd rn50_pre_post_xfbd)
OPTS=("--inputs post --extra-train data_xfbd_tier1" "--inputs pre+post --extra-train data_xfbd_tier1")
for g in 0 1; do
  CUDA_VISIBLE_DEVICES=$g python train_bda.py $COMMON ${OPTS[$g]} --out "results/${NAMES[$g]}" \
      >"logs/${NAMES[$g]}_$J.log" 2>&1 &
done
wait
echo "training done at $(date)"
for g in 0 1; do
  m=crop_${NAMES[$g]}
  (CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 500 --image-dir "$SHM" --out results/harness/$m &&
   CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 500 --image-dir "$SHM" --out results/harness/$m \
       --blend poisson --cf-only &&
   CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 0 --data data_xfbd_test --split xfbd_test \
       --max-dist-pasted 64 --out results/harness_xfbd/$m) >"logs/harness_${m}_$J.log" 2>&1 &
done
wait
echo "Job finished at $(date)"
