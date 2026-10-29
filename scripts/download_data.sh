#!/usr/bin/env bash
# 데이터 3종을 NAS 에 tar 형태로 내려받는다. 여기서 압축을 풀지 않는다.
#
#   압축 해제는 scripts/stage_data.sh 가 계산 노드의 /local_datasets 에서 수행한다.
#   (Seraph 규칙: 마스터 노드에서 압축 해제 금지, dataloader 가 NAS 직접 읽기 금지)
#
# 사용:  srun -p debug_ugrad --cpus-per-task=4 --mem=8G -t 2:00:00 bash scripts/download_data.sh
set -euo pipefail

TAR_DIR="${TAR_DIR:-/data/$USER/datasets/xai}"
mkdir -p "$TAR_DIR"
cd "$TAR_DIR"
echo "==> 저장 위치(NAS): $(pwd)"
echo

WGET_OPTS=(-c --timeout=30 --tries=3 --progress=dot:giga)

get () {   # $1 = 파일명, $2.. = 시도할 URL 들
  local f="$1"; shift
  if [ -f "$f" ] && gzip -t "$f" 2>/dev/null; then
    echo "[skip] $f 이미 존재 (무결성 OK)"
    return
  fi
  local url
  for url in "$@"; do
    echo "==> $f  <-  $url"
    if wget "${WGET_OPTS[@]}" -O "$f" "$url"; then
      gzip -t "$f" 2>/dev/null && { echo "    무결성 OK"; return; }
      echo "    [경고] 받은 파일이 손상됨. 다음 URL 시도"
    fi
    rm -f "$f"
  done
  echo "[실패] $f 를 받지 못했다."
  return 1
}

# ---------------------------------------------------------------- Waterbirds
# 압축 468MB / 해제 후 약 494MB. 학습·평가용 공식 배포본.
# nlp.stanford.edu 는 현재 응답이 없어 downloads.cs.stanford.edu 를 먼저 쓴다.
get "waterbird_complete95_forest2water2.tar.gz" \
    "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz" \
    "https://nlp.stanford.edu/data/dro/waterbird_complete95_forest2water2.tar.gz"

# ------------------------------------------------------------------- CUB-200
# 1.1GB. 원본 새 이미지와 메타데이터.
get "CUB_200_2011.tgz" \
    "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz"

# ------------------------------------------------------- CUB segmentations
# ★ 38MB. **CUB_200_2011.tgz 에는 segmentations/ 가 들어있지 않다.**
#   Caltech 의 별도 레코드(w9d68-gec53, "CUB-200-2011 Segmentations")에 있다.
#   Attr_bg 계산과 STEP 2 반사실 합성에 반드시 필요하다.
get "segmentations.tgz" \
    "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"

echo
echo "==> 완료"
ls -lh "$TAR_DIR"
echo
echo "다음 단계: 잡 안에서  bash scripts/stage_data.sh  (계산 노드로 펼치기)"
