#!/usr/bin/env bash
# NAS 에 있는 tar 를 계산 노드의 로컬 디스크(/local_datasets)로 펼친다.
#
# Seraph 규칙: dataloader 가 NAS(/data) 를 직접 읽으면 안 된다.
#              모든 학습/평가 잡은 시작할 때 이 스크립트를 먼저 부른다.
#
# /local_datasets 는 노드마다 따로다. 펼치기가 끝난 tar 는 건너뛰므로 매번 불러도 된다.
# 같은 노드에 잡 여러 개가 동시에 걸려도 flock 으로 한 번에 하나만 펼치고,
# 나머지는 끝날 때까지 기다린다.
set -euo pipefail

TAR_DIR="${TAR_DIR:-/data/$USER/datasets/xai}"
LOCAL_ROOT="${LOCAL_ROOT:-/local_datasets/$USER/xai}"

WB_DIR="waterbird_complete95_forest2water2"
CUB_DIR="CUB_200_2011"

mkdir -p "$LOCAL_ROOT"
echo "==> node=$(hostname)  local=$LOCAL_ROOT"

exec 9>"$LOCAL_ROOT/.stage.lock"
flock 9

untar () {   # $1 = tar, $2 = 펼친 뒤 있어야 하는 경로(LOCAL_ROOT 기준), $3 = 풀 위치
  local tgz="$TAR_DIR/$1" sentinel="$2" dest="$3"
  # 완료 표식은 tar 가 끝까지 풀린 뒤에만 생긴다. 중간에 끊기면 다음 잡이 다시 푼다.
  local done_mark="$LOCAL_ROOT/.staged_$1"
  if [ -e "$done_mark" ]; then
    echo "[skip] $1 이미 펼쳐져 있음"
    return
  fi
  [ -f "$tgz" ] || { echo "[실패] $tgz 없음. 먼저 scripts/download_data.sh 실행할 것"; exit 1; }
  echo "==> $1 펼치는 중 ($(du -h "$tgz" | cut -f1))"
  mkdir -p "$dest"
  tar -xzf "$tgz" -C "$dest"
  [ -e "$LOCAL_ROOT/$sentinel" ] || { echo "[실패] $sentinel 이 생기지 않음"; exit 1; }
  touch "$done_mark"
}

untar "${WB_DIR}.tar.gz"  "$WB_DIR/metadata.csv"      "$LOCAL_ROOT"
untar "${CUB_DIR}.tgz"    "$CUB_DIR/images.txt"       "$LOCAL_ROOT"
# segmentations.tgz 의 최상위는 segmentations/ 이므로 CUB_200_2011 안에 푼다.
untar "segmentations.tgz" "$CUB_DIR/segmentations"    "$LOCAL_ROOT/$CUB_DIR"

echo
echo "==> 스테이징 완료"
du -sh "$LOCAL_ROOT/$WB_DIR" "$LOCAL_ROOT/$CUB_DIR" 2>/dev/null || true
echo "LOCAL_ROOT=$LOCAL_ROOT"
