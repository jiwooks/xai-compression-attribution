#!/usr/bin/bash
# STEP 4 sanity check. 가중치 무작위화 (Adebayo et al. 2018).
# 원래 모델과 모든 가중치를 새로 초기화한 모델의 맵 Spearman 을 이미지별로 계산한다 (세 방법).
#
#   sbatch --exclude=aurora-g2,aurora-g5,aurora-g6 scripts/attr_sanity_slurm.sh 0 200 [rise]    # seed, 이미지 수, 방법
#
# 출력: results/attr_sanity_erm_seed<k>.json

#SBATCH --job-name=xai-sanity
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-4
#SBATCH --output=logs/attr-sanity-%A.out

set -euo pipefail
SEED="${1:-0}"
N="${2:-200}"
METHODS="${3:-gradcam,ig,rise}"
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"
bash scripts/stage_data.sh
"$PY" compute_attribution.py --sanity "$N" --ckpt "runs/erm_seed${SEED}/best.pt" \
    --wb_root "$WB_ROOT" --cub_root "$CUB_ROOT" --out "results/attr_sanity_erm_seed${SEED}" --sanity_methods "$METHODS"
