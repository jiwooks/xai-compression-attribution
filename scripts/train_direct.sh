#!/usr/bin/env bash
# 대화형 GPU 세션 안에서 직접 돌릴 때 (점검/디버그용).
# 본 학습은 sbatch scripts/train_slurm.sh 를 쓸 것.
#
#   srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 4:00:00 --pty bash
#   cd /data/$USER/xai_project && bash scripts/train_direct.sh 0
#
# 마스터 노드(aurora-master)에서 실행하면 안 된다. GPU 도 없고 규칙 위반이다.
set -euo pipefail
SEED="${1:-0}"
# sbatch 는 스크립트를 /var/spool/slurm 으로 복사해 실행하므로 dirname $0 을 쓸 수 없다.
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"
mkdir -p logs

[ -n "${SLURM_JOB_ID:-}" ] || echo "[경고] SLURM 할당 밖이다. 마스터 노드가 아닌지 확인할 것."
command -v nvidia-smi >/dev/null || { echo "[중단] GPU 노드가 아니다."; exit 1; }
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

bash scripts/stage_data.sh

"$PY" train_erm.py \
    --wb_root "$WB_ROOT" \
    --out_dir "$PROJ/runs/erm_seed${SEED}" \
    --seed "$SEED" \
    --epochs 100 \
    --batch_size 128 \
    --num_workers 8 \
    --amp \
    2>&1 | tee "logs/erm_seed${SEED}.log"
