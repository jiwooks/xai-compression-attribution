#!/usr/bin/bash
# seed 3·4 보조 분석 (PREREG_seeds5.md) — dense 학습 잡 뒤에 이어질 잡을 SLURM 의존성으로 한 번에 건다.
# 사람이 중간에 손대지 않아도 앞 단계가 성공하면(afterok) 다음 단계가 자동으로 시작된다.
# sbatch 만 부르므로 마스터 노드에서 실행해도 된다 (python 실행 없음).
#
#   cd /data/$USER/xai_project
#   bash scripts/seeds5_chain.sh 3 <dense 학습 job id>
#
# seed 하나당 잡 7개 (제출 한도 16 안에 들도록 나눔):
#   measure  : dense 의 반사실 확률·Δp (--save_probs)          ← 학습 뒤
#   prune    : sparsity 0 30 50 70, 20 epoch, S·Δp 측정        ← 학습 뒤
#   sanity   : erm 가중치 무작위화 200장 (세 방법)              ← 학습 뒤
#   gradcam  : 5개 모델                                          ← measure, prune 뒤
#   ig ×2    : "erm prune_sp00 prune_sp30", "prune_sp50 prune_sp70" (200 step)  ← measure, prune 뒤
#   rise     : 5개 모델 (rise_images.csv 1,000장, 마스크 2,000개, rise_seed 0)   ← measure, prune 뒤
# 앞 잡이 실패하면 뒤 잡은 DependencyNeverSatisfied 로 남는다. squeue 로 확인하고 scancel 로 지운다.

set -euo pipefail
SEED="${1:?사용법: seeds5_chain.sh <seed> <dense 학습 job id>}"
TRAIN="${2:?dense 학습 job id 가 필요하다}"
mkdir -p logs

sub () {   # $1 = 의존성, 나머지 = sbatch 인자. job id 만 출력
  local dep="$1"; shift
  sbatch --parsable --dependency="$dep" --kill-on-invalid-dep=no "$@"
}

M=$(sub "afterok:$TRAIN" --export="ALL,EXTRA=--save_probs" scripts/measure_shortcut_slurm.sh k5 "$SEED")
P=$(sub "afterok:$TRAIN" scripts/prune_slurm.sh "$SEED" "0 30 50 70")
S=$(sub "afterok:$TRAIN" scripts/attr_sanity_slurm.sh "$SEED" 200)
AFTER="afterok:$M:$P"
G=$(sub "$AFTER" scripts/attr_slurm.sh "$SEED" "erm prune_sp00 prune_sp30 prune_sp50 prune_sp70" gradcam)
I1=$(sub "$AFTER" --export="ALL,EXTRA=--ig_steps 200" scripts/attr_slurm.sh "$SEED" "erm prune_sp00 prune_sp30" ig)
I2=$(sub "$AFTER" --export="ALL,EXTRA=--ig_steps 200" scripts/attr_slurm.sh "$SEED" "prune_sp50 prune_sp70" ig)
R=$(sub "$AFTER" scripts/attr_slurm.sh "$SEED" "erm prune_sp00 prune_sp30 prune_sp50 prune_sp70" rise)

echo "seed $SEED (학습 $TRAIN): measure $M, prune $P, sanity $S, gradcam $G, ig $I1 $I2, rise $R"
