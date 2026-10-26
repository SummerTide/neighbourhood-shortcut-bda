#!/bin/bash
# CPU-only data preparation; the 4 GPUs are requested for their 36 CPU cores.
set -euo pipefail
export H_HOME=${H_HOME:-$HOME}
PROJ=${PROJ:-$PWD}
exec >"$PROJ/logs/prep_${JOB_ID:-$$}.log" 2>&1
export OMP_NUM_THREADS=1
source "$H_HOME/miniconda3/etc/profile.d/conda.sh"
conda activate bda
NCORES=$(python -c "import os; print(len(os.sched_getaffinity(0)))")
echo "Job ${JOB_ID:-$$} on $(hostname), $NCORES cores, $(date)"
cd "$PROJ"
python prep_official.py --root "${XBD:-datasets/xBD}" --out "$PROJ/data" --workers "$NCORES"
python make_conditions.py --data "$PROJ/data" --split test --workers "$NCORES"
echo "Job finished at $(date)"
