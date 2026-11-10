#!/usr/bin/bash
# STEP 3. seed 하나에 대해 여러 sparsity 를 차례로 pruning + fine-tuning 하고 S 를 측정한다.
#
#   mkdir -p logs
#   sbatch scripts/prune_slurm.sh 0 "0 30 50 70"     # seed, sparsity(%) 목록
#   sbatch scripts/prune_slurm.sh 0 "80 90 95"
#   EXTRA="--epochs 5 --eval_every 1 --save_oneshot" sbatch scripts/prune_slurm.sh 0 "0 50 70" prune5ep   # P4 ST-A
# 점검:  EXTRA="--subset 128 --epochs 2 --eval_every 1" SKIP_S=1 \
#        srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 0:30:00 \
#        bash scripts/prune_slurm.sh 0 "50" smoke
#
# 출력: runs/prune_sp<XX>_seed<k>/  (summarize_shortcut.py 가 seed 를 묶을 수 있는 이름)

#SBATCH --job-name=xai-prune
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-12
#SBATCH --output=logs/prune-%A.out

set -euo pipefail
SEED="${1:?사용법: prune_slurm.sh <seed> \"<sparsity% 목록>\" [접두어]}"
SPARSITIES="${2:?sparsity 목록이 필요하다. 예: \"0 30 50\"}"
PREFIX="${3:-prune}"
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"
mkdir -p logs

echo "=== job ${SLURM_JOB_ID:-none} | node $(hostname) | seed $SEED | sparsity $SPARSITIES ==="
bash scripts/stage_data.sh
[ -n "${SKIP_S:-}" ] || bash scripts/stage_cf.sh k5

for SP in $SPARSITIES; do
  RUN="${PREFIX}_sp$(printf %02d "$SP")_seed${SEED}"
  echo; echo "########## $RUN ##########"
  "$PY" prune_finetune.py \
      --ckpt "$PROJ/runs/erm_seed${SEED}/best.pt" \
      --wb_root "$WB_ROOT" \
      --out_dir "$PROJ/runs/$RUN" \
      --sparsity "$(awk -v s="$SP" 'BEGIN{print s/100}')" \
      --seed "$SEED" --num_workers 8 --amp ${EXTRA:-}
  [ -n "${SKIP_S:-}" ] && continue
  EXPECT=$(grep '"avg_acc"' "$PROJ/runs/$RUN/final_test.json" | head -1 | tr -dc '0-9.')
  "$PY" measure_shortcut.py \
      --ckpt "$PROJ/runs/$RUN/best.pt" --wb_root "$WB_ROOT" \
      --cf_dir "$LOCAL_ROOT/cf/k5" --out "$PROJ/runs/$RUN/shortcut_k5.json" \
      --expect_avg_acc "$EXPECT" --seed "$SEED" --save_probs
  # one-shot(재학습 전) 모델도 있으면 같은 방식으로 잰다 (EXTRA 에 --save_oneshot, P4 ST-A)
  if [ -f "$PROJ/runs/$RUN/oneshot.pt" ]; then
    EXPECT1=$(grep '"avg_acc"' "$PROJ/runs/$RUN/oneshot_test.json" | head -1 | tr -dc '0-9.')
    "$PY" measure_shortcut.py \
        --ckpt "$PROJ/runs/$RUN/oneshot.pt" --wb_root "$WB_ROOT" \
        --cf_dir "$LOCAL_ROOT/cf/k5" --out "$PROJ/runs/$RUN/shortcut_k5_oneshot.json" \
        --expect_avg_acc "$EXPECT1" --seed "$SEED" --save_probs \
      || echo "[경고] $RUN one-shot 측정 실패 (fine-tuning 결과는 유지)"
  fi
done
echo "=== 완료 ==="
