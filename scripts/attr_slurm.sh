#!/usr/bin/bash
# STEP 4. seed 하나의 모델 목록에 대해 attribution 과 Attr_bg 를 계산한다.
#
#   mkdir -p logs
#   sbatch --exclude=aurora-g2,aurora-g5,aurora-g6 scripts/attr_slurm.sh 0 "erm prune_sp00 prune_sp30 prune_sp50 prune_sp70" gradcam
#   sbatch ... scripts/attr_slurm.sh 0 "erm prune_sp00" ig
#   sbatch ... scripts/attr_slurm.sh 0 "erm" rise            # results/rise_images.csv 가 먼저 있어야 한다
# 점검:  EXTRA="--subset 64 --png_dir data_check/attr_smoke --n_png 8" OUT_SUFFIX=_smoke \
#        srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 1:00:00 \
#        bash scripts/attr_slurm.sh 0 "erm" gradcam
#
# 출력: runs/<run>_seed<k>/attr_<method>.csv.gz, attr_<method>.json

#SBATCH --job-name=xai-attr
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --output=logs/attr-%A.out

set -euo pipefail
SEED="${1:?사용법: attr_slurm.sh <seed> \"<run 목록>\" <gradcam|ig|rise>}"
RUNS="${2:?run 목록이 필요하다. 예: \"erm prune_sp00\"}"
METHOD="${3:?method 가 필요하다 (gradcam|ig|rise)}"
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"
mkdir -p logs

echo "=== job ${SLURM_JOB_ID:-none} | node $(hostname) | seed $SEED | $METHOD | $RUNS ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bash scripts/stage_data.sh

for R in $RUNS; do
  RUN="$PROJ/runs/${R}_seed${SEED}"
  echo; echo "########## ${R}_seed${SEED} / $METHOD ##########"
  EXPECT=$(grep '"avg_acc"' "$RUN/final_test.json" | head -1 | tr -dc '0-9.')
  "$PY" compute_attribution.py \
      --ckpt "$RUN/best.pt" --wb_root "$WB_ROOT" --cub_root "$CUB_ROOT" \
      --method "$METHOD" --out "$RUN/attr_${METHOD}${OUT_SUFFIX:-}" \
      --expect_acc "$EXPECT" --ref_dp "$RUN/shortcut_k5_dp.csv.gz" --num_workers 8 ${EXTRA:-}
done
echo "=== 완료 ==="
