#!/usr/bin/bash
# STEP 1. dense ERM 학습 SLURM 작업.
#
# 제출:  mkdir -p logs && sbatch scripts/train_slurm.sh 0      # 인자 = seed
#
# 확인된 클러스터 사실 (aurora, 2026-10-28):
#   - 학부 파티션: batch_ugrad(무제한) / debug_ugrad(4시간)
#   - QoS 한도: 동시 8잡, 제출 16잡, GPU 8장, 최대 6일
#   - GPU: RTX 3090 24GB, 드라이버 535.161.07 (CUDA 12.2) -> torch cu121
#   - /home 은 계산 노드와 공유되지 않는다. 코드/출력/캐시 모두 /data 아래.

#SBATCH --job-name=xai-erm
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=1-0
#SBATCH --output=logs/erm-%A.out

set -euo pipefail
SEED="${1:-0}"
# sbatch 는 스크립트를 /var/spool/slurm 으로 복사해 실행하므로 dirname $0 을 쓸 수 없다.
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"
mkdir -p logs

echo "=== job ${SLURM_JOB_ID:-none} | node $(hostname) | seed $SEED ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# 데이터를 계산 노드 로컬 디스크로 펼친다 (NAS 직접 읽기 금지).
bash scripts/stage_data.sh

echo "python: $PY"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# 출력은 NAS 로. 쓰기 빈도가 낮아서(체크포인트 갱신 + eval_every 마다 로그 한 줄) 부담이 작다.
OUT_DIR="$PROJ/runs/erm_seed${SEED}"

"$PY" train_erm.py \
    --wb_root "$WB_ROOT" \
    --out_dir "$OUT_DIR" \
    --seed "$SEED" \
    --epochs 100 \
    --batch_size 128 \
    --num_workers 8 \
    --amp

echo "=== 완료: $OUT_DIR ==="
exit 0
