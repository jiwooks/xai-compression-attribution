#!/usr/bin/env bash
# 학습 파이프라인 점검 (1~2분). 본 학습 전에 반드시 먼저 돌린다.
#   srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 0:30:00 \
#        bash scripts/smoke_test.sh
set -euo pipefail
# sbatch 는 스크립트를 /var/spool/slurm 으로 복사해 실행하므로 dirname $0 을 쓸 수 없다.
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
bash scripts/stage_data.sh

"$PY" train_erm.py \
    --wb_root "$WB_ROOT" \
    --out_dir "$PROJ/runs/smoke" \
    --subset 128 --epochs 2 --eval_every 1 --num_workers 4 --amp

echo "=== smoke test 완료: $PROJ/runs/smoke ==="
