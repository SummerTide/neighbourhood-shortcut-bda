#!/bin/bash
# Islahiye (Turkiye 2023) evaluation, turkey_eval.py: every model in its own conda env, one GPU.
#   DATA, OUT, TAG  input dir (default data_turkey), output dir, log tag
#   SMOKE=1      a few models on a 1500 px region of the urban crop, to results/turkey_smoke
set -uo pipefail
export H_HOME=${H_HOME:-$HOME}
PROJ=${PROJ:-$PWD}
exec >"$PROJ/logs/turkey${SMOKE:+_smoke}${TAG:+_$TAG}_${JOB_ID:-$$}.log" 2>&1
export OMP_NUM_THREADS=4 HF_HOME=$H_HOME/huggingface_cache HF_HUB_OFFLINE=1
export TORCH_HOME=$H_HOME/.cache/torch XDG_CACHE_HOME=$H_HOME/.cache MPLCONFIGDIR=$PROJ/.mplconfig
source "$H_HOME/miniconda3/etc/profile.d/conda.sh"
cd "$PROJ"
if [ -n "${SMOKE:-}" ]; then ARGS="--region 8125,8125,1500,1500 --out results/turkey_smoke"; else ARGS="--data ${DATA:-data_turkey} --out ${OUT:-results/turkey} --skip-done"; fi
part() {  # part <env> <models...>
  local env=$1; shift
  conda activate "$env"
  if [ "$env" = bda ]; then export PYTHONPATH=${PYLIB:-}; else unset PYTHONPATH; fi
  echo "=== $env: $* ($(date))"
  python turkey_eval.py --model "$@" $ARGS || echo "=== part $env FAILED with status $?"
}
if [ -n "${SMOKE:-}" ]; then
  part bda changeos_r101 crop_rn50_pre_post_tight crop_rn50_pre_post_pswap_dw5_s1
  part bda_tc changeos_swint
  part bda_baselines mambabda_small
  part bda_dan disasteradaptivenet
else
  part bda crop_rn50_pre_post crop_rn50_pre_post_s1 crop_rn50_pre_post_s2 \
    crop_rn50_pre_post_tight crop_rn50_pre_post_tight_s1 crop_rn50_pre_post_tight_s2 \
    crop_rn50_pre_post_pswap_dw5 crop_rn50_pre_post_pswap_dw5_s1 crop_rn50_pre_post_pswap_dw5_s2 changeos_r101
  part bda_tc changeos_swint changestar_changen2
  part bda_baselines mambabda_small
  part bda_dan strong_baseline disasteradaptivenet
  part bda xview2_first_light xview2_first
  # fallback: the four encoder families separately (their mean is the full ensemble)
  if [ ! -f results/turkey/xview2_first/turkey_preds.npz ]; then
    part bda xview2_first_res34 xview2_first_res50 xview2_first_dpn92 xview2_first_se154
  fi
fi
echo "Job finished at $(date)"
