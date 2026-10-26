#!/bin/bash
# The best ResNet recipe on DINOv3-SAT ViT-L/16 fine-tuned crops (same settings as dinov3ft_*):
#   tight building crop (post, pre+post) and v1 = Poisson event swap + discordance x5 (post, pre+post),
# to be combined post hoc with the existing dinov3ft_post / dinov3ft_pre_post context models.
# Then each model goes through harness.py (standard + hard and Poisson counterfactuals + xFBD).
set -euo pipefail
export H_HOME=${H_HOME:-$HOME}
PROJ=${PROJ:-$PWD}
exec >"$PROJ/logs/method_dinov3_${JOB_ID:-$$}.log" 2>&1
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
VIT="--backbone dinov3_vitl16_sat --batch 64 --lr 2e-5 --head-lr-mult 10 --weight-decay 0.05"
TIGHT="--context-px 8 --tight-margin 4"
NAMES=(dinov3ft_post_tight dinov3ft_pre_post_tight dinov3ft_post_pswap_dw5 dinov3ft_pre_post_pswap_dw5)
OPTS=("--inputs post $VIT $TIGHT" "--inputs pre+post $VIT $TIGHT" "--inputs post $VIT $SWAP $DW" "--inputs pre+post $VIT $SWAP $DW")
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
