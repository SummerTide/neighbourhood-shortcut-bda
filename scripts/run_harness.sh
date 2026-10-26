#!/bin/bash
# Harness runs, one after another on one GPU:
#   MODEL=changeos_r101 TARGETS=500 [ENV=bda_tc] bash scripts/run_harness.sh
#   MODEL=crop_rn50_post:crop_rn50_pre_post   several models (colon-separated)
#   DATA=data_xfbd_test SPLIT=xfbd_test MAXDIST=64 TARGETS=0 OUT=results/harness_xfbd   another test set
#   BLEND=poisson CFONLY=1   Poisson-blended counterfactual only (harness_counterfactual_poisson.csv)
#   PLACEMENT=vacant CFONLY=1  counterfactual on open ground (harness_counterfactual_vacant.csv)
#   CFGROUPS=concordant_damage:concordant_intact   only these counterfactual groups (others replayed)
set -euo pipefail
export H_HOME=${H_HOME:-$HOME}
PROJ=${PROJ:-$PWD}
NAME=${MODEL//:/+}
if [ ${#NAME} -gt 100 ]; then NAME="${MODEL%%:*}+$(( $(echo "$MODEL" | tr ':' '\n' | wc -l) - 1 ))more"; fi
exec >"$PROJ/logs/harness_${NAME}${DATA:+_$DATA}${SPLIT:+_$SPLIT}${BLEND:+_$BLEND}${PLACEMENT:+_$PLACEMENT}_${JOB_ID:-$$}.log" 2>&1
export OMP_NUM_THREADS=4
source "$H_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate "${ENV:-bda}"
# haste_mc/pylib holds packages built for bda's Python; other envs bring their own.
if [ "${ENV:-bda}" = bda ]; then export PYTHONPATH=${PYLIB:-}; fi
export HF_HOME=$H_HOME/huggingface_cache HF_HUB_OFFLINE=1
# Backbone initialisations (e.g. torchvision Swin-T inside TorChange's ChangeOS) come from the
export TORCH_HOME=$H_HOME/.cache/torch XDG_CACHE_HOME=$H_HOME/.cache
export MPLCONFIGDIR=$PROJ/.mplconfig
cd "$PROJ"
for m in ${MODEL//:/ }; do
    python harness.py --model "$m" --targets "${TARGETS:-500}" --out "${OUT:-results/harness}/$m" \
        ${DATA:+--data $DATA --split ${SPLIT:-test}} ${MAXDIST:+--max-dist-pasted $MAXDIST} \
        ${BLEND:+--blend $BLEND} ${PLACEMENT:+--placement $PLACEMENT} ${CFONLY:+--cf-only} ${CFGROUPS:+--cf-groups $CFGROUPS}
done
echo "Job finished at $(date)"
