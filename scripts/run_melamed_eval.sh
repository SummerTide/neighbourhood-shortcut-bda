#!/bin/bash
# Evaluation of the Melamed (xFBD tier1) ResNet-50 hosts after run_melamed.sh hit its time limit:
# standard + hard and Poisson counterfactuals + the xFBD single-building test, one model per GPU.
set -euo pipefail
export H_HOME=${H_HOME:-$HOME}
PROJ=${PROJ:-$PWD}
exec >"$PROJ/logs/melamed_eval_${JOB_ID:-$$}.log" 2>&1
export OMP_NUM_THREADS=1
source "$H_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bda
export PYTHONPATH=${PYLIB:-}
export TORCH_HOME=$H_HOME/.cache/torch XDG_CACHE_HOME=$H_HOME/.cache MPLCONFIGDIR=$PROJ/.mplconfig
cd "$PROJ"
g=0
for m in crop_rn50_post_xfbd crop_rn50_pre_post_xfbd; do
  (CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 500 --out results/harness/$m &&
   CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 500 --out results/harness/$m --blend poisson --cf-only &&
   CUDA_VISIBLE_DEVICES=$g python harness.py --model $m --targets 0 --data data_xfbd_test --split xfbd_test \
       --max-dist-pasted 64 --out results/harness_xfbd/$m) >"logs/harness_${m}_${JOB_ID:-$$}.log" 2>&1 &
  g=$((g + 1))
done
wait
echo "Job finished at $(date)"
