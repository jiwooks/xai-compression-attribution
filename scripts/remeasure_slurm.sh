#!/usr/bin/bash
# STEP 3+. 이미 학습된 모델들의 반사실 예측 확률을 저장하고 Δp 를 계산한다 (재학습 없음).
# 기존 shortcut_k5.json 과 flip 기반 값(S, BFR, 그룹별, CI)이 같은지 확인한 뒤에만 덮어쓴다.
#
#   mkdir -p logs
#   sbatch --exclude=aurora-g2,aurora-g5,aurora-g6 scripts/remeasure_slurm.sh 0
#   sbatch ... scripts/remeasure_slurm.sh 0 "erm"        # run 이름(seed 접미어 제외) 목록을 줄 수 있다
# 점검:  srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 0:30:00 \
#        bash scripts/remeasure_slurm.sh 0 "erm"
#
# 출력: runs/<run>_seed<k>/shortcut_k5.json (dp 항목 추가), shortcut_k5_rows.csv.gz, shortcut_k5_dp.csv.gz

#SBATCH --job-name=xai-dp
#SBATCH --partition=batch_ugrad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=32G
#SBATCH --time=0-4
#SBATCH --output=logs/remeasure-%A.out

set -euo pipefail
SEED="${1:?사용법: remeasure_slurm.sh <seed> [\"<run 목록>\"]}"
RUNS="${2:-erm prune_sp00 prune_sp30 prune_sp50 prune_sp70 prune_sp80 prune_sp90 prune_sp95}"
TAG="${TAG:-k5}"
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"
mkdir -p logs

echo "=== job ${SLURM_JOB_ID:-none} | node $(hostname) | seed $SEED | runs $RUNS ==="
bash scripts/stage_data.sh
bash scripts/stage_cf.sh "$TAG"
[ -f "$LOCAL_ROOT/cf/$TAG/manifest.csv" ] || { echo "[실패] 반사실 $TAG 가 이 노드에 없다"; exit 1; }

for R in $RUNS; do
  RUN="$PROJ/runs/${R}_seed${SEED}"
  echo; echo "########## ${R}_seed${SEED} ##########"
  OUT="$RUN/shortcut_${TAG}.json"
  [ -f "$OUT" ] || { echo "[실패] 기존 결과 $OUT 가 없다 (회귀 확인 불가)"; exit 1; }
  cp "$OUT" "$RUN/.shortcut_${TAG}.prev.json"
  EXPECT=$(grep '"avg_acc"' "$RUN/final_test.json" | head -1 | tr -dc '0-9.')
  "$PY" measure_shortcut.py \
      --ckpt "$RUN/best.pt" --wb_root "$WB_ROOT" \
      --cf_dir "$LOCAL_ROOT/cf/$TAG" --out "$OUT" \
      --expect_avg_acc "$EXPECT" --seed "$SEED" \
      --save_probs --regress_check "$RUN/.shortcut_${TAG}.prev.json"
done
echo "=== 완료 ==="
