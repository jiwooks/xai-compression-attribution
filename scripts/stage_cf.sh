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
  mkdir -p "$2"
  tar -xf "$1" -C "$2"
  touch "$mark"
  echo "==> $(basename "$1") -> $2"
}

[ -f "$TAR_DIR/places_val_bg.tar" ] || { echo "[실패] 먼저 scripts/download_places.sh 실행할 것"; exit 1; }
untar "$TAR_DIR/places_val_bg.tar" "$LOCAL_ROOT/places_bg"

if [ -n "${1:-}" ] && [ -f "$TAR_DIR/counterfactuals_$1.tar" ]; then
  untar "$TAR_DIR/counterfactuals_$1.tar" "$LOCAL_ROOT/cf/$1"
fi
