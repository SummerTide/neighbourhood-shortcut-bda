#!/bin/bash
# BRIGHT, separate pre-event encoder (as run_bright_method2.sh): the missing seeds, so that context,
# tight and swap-trained experts all have seeds 0, 1 and 2. Then harness.py, standard pass only
# (counterfactuals run separately). One GPU; the runs listed in IDX (default all five) one after another:
#   IDX=0:1:2 bash scripts/run_bright_seeds.sh
set -euo pipefail
export H_HOME=${H_HOME:-$HOME}
PROJ=${PROJ:-$PWD}
exec >"$PROJ/logs/bright_seeds_${JOB_ID:-$$}.log" 2>&1
export OMP_NUM_THREADS=1 BDA_N_CLASSES=3
source "$H_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bda
export PYTHONPATH=${PYLIB:-}
export TORCH_HOME=$H_HOME/.cache/torch XDG_CACHE_HOME=$H_HOME/.cache MPLCONFIGDIR=$PROJ/.mplconfig
cd "$PROJ"
J=${JOB_ID:-$$}
IDX=${IDX:-0:1:2:3:4}
echo "Job $J on $(hostname), $(nvidia-smi -L | wc -l) GPUs, $(date)"
SHM=/dev/shm/fbda_brts_$J
mkdir -p "$SHM" && cp data_bright/pre.u8 data_bright/post.u8 "$SHM"/ && trap 'rm -rf "$SHM"' EXIT
COMMON="--data data_bright --image-dir $SHM --epochs 8 --workers 8 --inputs pre+post --separate-pre"
SWAP="--context-swap donor --swap-blend poisson --swap-margin 2,12 --swap-donor event --discordance-weight 5"
NAMES=(brt2_rn50_pre_post_s2 brt2_rn50_pre_post_tight_s1 brt2_rn50_pre_post_tight_s2
       brt2_rn50_pre_post_pswap_dw5_s1 brt2_rn50_pre_post_pswap_dw5_s2)
OPTS=("--seed 2" "--seed 1 --context-px 8 --tight-margin 4" "--seed 2 --context-px 8 --tight-margin 4"
      "--seed 1 $SWAP" "--seed 2 $SWAP")
run() {  # run <index> <gpu>: train, then score on the BRIGHT test buildings
  local n=${NAMES[$1]}
  CUDA_VISIBLE_DEVICES=$2 python train_bda.py $COMMON ${OPTS[$1]} --out "results/$n" >"logs/${n}_$J.log" 2>&1
  CUDA_VISIBLE_DEVICES=$2 python harness.py --model "crop_$n" --targets 0 --data data_bright --image-dir "$SHM" \
      --out "results/harness_bright/crop_$n" >"logs/harness_crop_${n}_$J.log" 2>&1
  echo "$n done at $(date)"
}
for i in ${IDX//:/ }; do run "$i" 0; done
echo "Job finished at $(date)"
