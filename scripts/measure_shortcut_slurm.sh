#!/usr/bin/env bash
# STEP 2 shortcut 지수 S 측정.
#   sbatch scripts/measure_shortcut_slurm.sh k5 0     # 반사실 이름, seed

#SBATCH --job-name=xai-S
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-4
#SBATCH --output=logs/shortcut-%A.out

set -euo pipefail
NAME="${1:?사용법: measure_shortcut_slurm.sh <반사실 이름> <seed>}"
SEED="${2:?사용법: measure_shortcut_slurm.sh <반사실 이름> <seed>}"
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"
mkdir -p logs

bash scripts/stage_data.sh
bash scripts/stage_cf.sh "$NAME"

RUN="$PROJ/runs/erm_seed${SEED}"
EXPECT=$(grep '"avg_acc"' "$RUN/final_test.json" | tr -dc '0-9.')

"$PY" measure_shortcut.py \
    --ckpt "$RUN/best.pt" \
    --wb_root "$WB_ROOT" \
    --cf_dir "$LOCAL_ROOT/cf/$NAME" \
    --out "$RUN/shortcut_${NAME}.json" \
    --expect_avg_acc "$EXPECT" \
    --seed "$SEED"
