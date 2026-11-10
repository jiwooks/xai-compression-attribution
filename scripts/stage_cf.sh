#!/usr/bin/env bash
# STEP 2 데이터를 계산 노드 로컬로 펼친다.
#   - places_val_bg.tar           -> $LOCAL_ROOT/places_bg
#   - counterfactuals_<이름>.tar  -> $LOCAL_ROOT/cf/<이름>  (인자로 이름을 줬고 tar 가 있을 때)
set -euo pipefail

TAR_DIR="${TAR_DIR:-/data/$USER/datasets/xai}"
LOCAL_ROOT="${LOCAL_ROOT:-/local_datasets/$USER/xai}"
mkdir -p "$LOCAL_ROOT"

exec 8>"$LOCAL_ROOT/.stage_cf.lock"
flock 8

untar () {   # $1 = tar 경로, $2 = 풀 위치
  local mark="$LOCAL_ROOT/.staged_$(basename "$1")"
  [ -e "$mark" ] && { echo "[skip] $(basename "$1") 이미 펼쳐져 있음"; return; }
  [ -f "$1" ] || { echo "[실패] $1 없음. 먼저 scripts/download_places.sh / generate_cf_slurm.sh 실행할 것"; exit 1; }
  mkdir -p "$2"
  tar -xf "$1" -C "$2"
  touch "$mark"
  echo "==> $(basename "$1") -> $2"
}

# 완료 표식을 먼저 본다. NAS tar 가 없어도 이미 펼쳐진 노드에서는 그대로 쓸 수 있다.
untar "$TAR_DIR/places_val_bg.tar" "$LOCAL_ROOT/places_bg"

if [ -n "${1:-}" ]; then
  if [ -e "$LOCAL_ROOT/.staged_counterfactuals_$1.tar" ] || [ -f "$TAR_DIR/counterfactuals_$1.tar" ]; then
    untar "$TAR_DIR/counterfactuals_$1.tar" "$LOCAL_ROOT/cf/$1"
  fi
fi
