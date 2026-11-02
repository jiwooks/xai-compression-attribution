#!/usr/bin/env bash
# STEP 2 배경 이미지 준비. Places365 평가셋에서 4개 카테고리만 뽑아 NAS 에 tar 하나로 둔다.
#
# 서버가 연결당 약 0.3~0.5MB/s 로 느리고 중간에 끊는 일이 있다 (val_large.tar 2.27GB, 약 2시간).
# 그래서 로컬 디스크에 이어받기로 받은 뒤 필요한 파일만 꺼내고, 원본 tar 는 지운다.
#   mkdir -p logs && sbatch scripts/download_places.sh

#SBATCH --job-name=xai-places
#SBATCH --partition=batch_ugrad
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=0-12
#SBATCH --output=logs/places-%A.out

set -euo pipefail

TAR_DIR="${TAR_DIR:-/data/$USER/datasets/xai}"
OUT="$TAR_DIR/places_val_bg.tar"
WORK="${WORK:-/local_datasets/$USER/xai/places_tmp}"
BASE="https://data.csail.mit.edu/places/places365"

if [ -f "$OUT" ]; then
  echo "[skip] $OUT 이미 있음"
  exit 0
fi

mkdir -p "$WORK" "$TAR_DIR"
cd "$WORK"
echo "==> node=$(hostname) work=$WORK"

fetch () {   # $1 = 파일명. 끊기면 이어서 받는다.
  local expected
  expected=$(curl -fsSIL --max-time 60 "$BASE/$1" | tr -d '\r' | awk 'tolower($1)=="content-length:"{n=$2} END{print n}')
  [ -n "$expected" ] || { echo "[실패] $1 크기 확인 불가"; exit 1; }
  local try
  for try in $(seq 1 200); do
    wget -c -q --timeout=60 --tries=1 "$BASE/$1" || true
    if [ "$(stat -c %s "$1" 2>/dev/null || echo 0)" = "$expected" ]; then
      echo "    $1 완료 ($expected bytes, 시도 $try회)"
      return
    fi
    echo "    $1 $(stat -c %s "$1" 2>/dev/null || echo 0)/$expected bytes, 재시도 $try"
    sleep 20
  done
  echo "[실패] $1 를 다 받지 못했다."
  exit 1
}

fetch filelist_places365-standard.tar
tar -xf filelist_places365-standard.tar categories_places365.txt places365_val.txt

# bg_manifest.csv: file,category,place  (place: 0=land, 1=water)
awk '
  NR==FNR { id[$1]=$2; next }
  FNR==1 {
    want["/b/bamboo_forest"]=0; want["/f/forest/broadleaf"]=0
    want["/l/lake/natural"]=1;  want["/o/ocean"]=1
    for (c in want) {
      if (!(c in id)) { print "카테고리 없음: " c > "/dev/stderr"; exit 1 }
      label[id[c]]=c; place[id[c]]=want[c]
    }
    print "file,category,place" > "bg_manifest.csv"
  }
  ($2 in label) { print $1 "," label[$2] "," place[$2] >> "bg_manifest.csv"; print "val_large/" $1 > "wanted.txt" }
' categories_places365.txt places365_val.txt
echo "==> 대상 $(wc -l < wanted.txt)장"

fetch val_large.tar

rm -rf images
mkdir images
tar -xf val_large.tar -C images --strip-components=1 -T wanted.txt

N=$(ls images | wc -l)
[ "$N" -eq "$(wc -l < wanted.txt)" ] || { echo "[실패] 추출 $N장, 기대 $(wc -l < wanted.txt)장"; exit 1; }

tar -cf "$OUT.tmp" bg_manifest.csv images
mv "$OUT.tmp" "$OUT"
cut -d, -f2,3 bg_manifest.csv | tail -n +2 | sort | uniq -c
ls -lh "$OUT"
cd /
rm -rf "$WORK"
echo "=== 완료 ==="
