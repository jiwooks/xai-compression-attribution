#!/usr/bin/env bash
# STEP 2 반사실 이미지 생성. 계산 노드 로컬에 만든 뒤 NAS 에는 tar 하나로만 올린다
# (Seraph 규칙: 생성 이미지를 NAS 에 직접 저장 금지).
#
#   srun -p debug_ugrad --cpus-per-task=16 --mem=32G -t 2:00:00 \
#        bash scripts/generate_cf_slurm.sh k5 --n_bg 5
# 점검:  ... bash scripts/generate_cf_slurm.sh smoke --n_bg 2 --subset 64
set -euo pipefail
PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
source "$PROJ/scripts/_common.sh"
cd "$PROJ"

NAME="${1:?사용법: generate_cf_slurm.sh <이름> [generate_counterfactuals.py 인자]}"
shift
bash scripts/stage_data.sh
bash scripts/stage_cf.sh

CF_DIR="$LOCAL_ROOT/cf/$NAME"
rm -rf "$CF_DIR"
"$PY" generate_counterfactuals.py \
    --wb_root "$WB_ROOT" --cub_root "$CUB_ROOT" \
    --bg_root "$LOCAL_ROOT/places_bg" --out_dir "$CF_DIR" \
    --num_workers "${SLURM_CPUS_PER_TASK:-8}" "$@"

OUT="$TAR_DIR/counterfactuals_$NAME.tar"
tar -cf "$OUT.tmp" -C "$CF_DIR" .
mv "$OUT.tmp" "$OUT"
touch "$LOCAL_ROOT/.staged_$(basename "$OUT")"
du -sh "$CF_DIR"; ls -lh "$OUT"
