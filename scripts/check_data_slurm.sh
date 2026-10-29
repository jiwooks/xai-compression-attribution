#!/usr/bin/env bash
# STEP 0. 데이터/마스크 점검. 계산 노드에서 실행 (GPU 불필요).
#   srun -p debug_ugrad --cpus-per-task=8 --mem=16G -t 1:00:00 bash scripts/check_data_slurm.sh
set -euo pipefail
# sbatch 는 스크립트를 /var/spool/slurm 으로 복사해 실행하므로 dirname $0 을 쓸 수 없다.
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"

bash scripts/stage_data.sh

"$PY" check_data.py \
    --wb_root "$WB_ROOT" \
    --cub_root "$CUB_ROOT" \
    --out_dir "$PROJ/data_check" \
    "$@"

echo "=== 결과: $PROJ/data_check (overlay 이미지를 눈으로 확인할 것) ==="
