#!/usr/bin/bash
# NAS 의 데이터 tar 를 계산 노드 로컬 사본(/local_datasets)에서 다시 만든다 (2026-11-10, CLAUDE.md §5-17).
# 원래 tar 와 같은 내부 구조로 묶어 stage_data.sh / stage_cf.sh 가 그대로 쓸 수 있게 한다.
# 이미 NAS 에 있는 tar 는 건드리지 않는다. 만든 뒤 임시 폴더에 다시 풀어 원본과 파일 목록·크기를 비교한다.
#
#   srun -p debug_ugrad -w aurora-g1 --ntasks=1 --cpus-per-task=1 --mem=8G -t 2:00:00 \
#        bash scripts/restore_nas_tars.sh </dev/null
#
# 주의: 공식 배포 tarball 과 바이트 단위로 같지는 않다 (같은 파일을 다시 묶은 것).

set -euo pipefail
TAR_DIR="${TAR_DIR:-/data/$USER/datasets/xai}"
LOCAL_ROOT="${LOCAL_ROOT:-/local_datasets/$USER/xai}"
CHECK="${LOCAL_ROOT}_restore_check"
mkdir -p "$TAR_DIR"
echo "=== node $(hostname) | from $LOCAL_ROOT -> $TAR_DIR ==="

for m in waterbird_complete95_forest2water2.tar.gz CUB_200_2011.tgz segmentations.tgz \
         places_val_bg.tar counterfactuals_k5.tar; do
  [ -e "$LOCAL_ROOT/.staged_$m" ] || { echo "[중단] 이 노드에 $m 완료 표식이 없다"; exit 1; }
done

listing () {   # $1 = 폴더. 상대경로와 크기 목록
  (cd "$1" && find . -type f -printf '%P %s\n' | LC_ALL=C sort)
}

make_tar () {  # $1 = tar 이름, $2 = tar 옵션(z 또는 빈 문자열), $3 = -C 위치, 나머지 = 대상
  local name="$1" z="$2" dir="$3"; shift 3
  if [ -e "$TAR_DIR/$name" ]; then echo "[skip] $name 이미 NAS 에 있다"; return; fi
  echo "==> $name 만드는 중"
  tar -c${z}f "$TAR_DIR/$name.tmp" -C "$dir" "$@"
  mv "$TAR_DIR/$name.tmp" "$TAR_DIR/$name"
  ls -lh "$TAR_DIR/$name"
}

verify () {    # $1 = tar 이름, $2 = tar 옵션, $3 = 비교할 원본 폴더, $4 = 풀린 뒤 비교할 하위 경로
  local name="$1" z="$2" src="$3" sub="$4"
  rm -rf "$CHECK" && mkdir -p "$CHECK"
  tar -x${z}f "$TAR_DIR/$name" -C "$CHECK"
  if diff <(listing "$src") <(listing "$CHECK/$sub") >/dev/null; then
    echo "[검증 통과] $name : $(listing "$src" | wc -l) 파일, 목록·크기 일치"
  else
    echo "[검증 실패] $name"; diff <(listing "$src") <(listing "$CHECK/$sub") | head; exit 1
  fi
  rm -rf "$CHECK"
}

WB=waterbird_complete95_forest2water2
CUB=CUB_200_2011

# 1) Waterbirds: 최상위 waterbird_complete95_forest2water2/
make_tar "$WB.tar.gz" z "$LOCAL_ROOT" "$WB"
verify   "$WB.tar.gz" z "$LOCAL_ROOT/$WB" "$WB"

# 2) CUB: 최상위 CUB_200_2011/ + attributes.txt. segmentations/ 는 따로 받는 것이라 뺀다.
make_tar "$CUB.tgz" z "$LOCAL_ROOT" --exclude="$CUB/segmentations" "$CUB" attributes.txt
rm -rf "$CHECK" && mkdir -p "$CHECK" && tar -xzf "$TAR_DIR/$CUB.tgz" -C "$CHECK"
if diff <(listing "$LOCAL_ROOT/$CUB" | grep -v '^segmentations/') <(listing "$CHECK/$CUB") >/dev/null \
   && [ -f "$CHECK/attributes.txt" ]; then
  echo "[검증 통과] $CUB.tgz : $(listing "$CHECK/$CUB" | wc -l) 파일 + attributes.txt"
else
  echo "[검증 실패] $CUB.tgz"; exit 1
fi
rm -rf "$CHECK"

# 3) segmentations: 최상위 segmentations/ (stage_data.sh 가 CUB_200_2011 안에 푼다)
make_tar segmentations.tgz z "$LOCAL_ROOT/$CUB" segmentations
verify   segmentations.tgz z "$LOCAL_ROOT/$CUB/segmentations" segmentations

# 4) Places 배경: 최상위 bg_manifest.csv, images/ (download_places.sh 와 같은 구조)
make_tar places_val_bg.tar "" "$LOCAL_ROOT/places_bg" bg_manifest.csv images
verify   places_val_bg.tar "" "$LOCAL_ROOT/places_bg" .

# 5) 반사실 k5: 폴더 내용 그대로 (generate_cf_slurm.sh 와 같은 구조)
make_tar counterfactuals_k5.tar "" "$LOCAL_ROOT/cf/k5" .
verify   counterfactuals_k5.tar "" "$LOCAL_ROOT/cf/k5" .

echo; ls -lh "$TAR_DIR"
echo "=== 완료 ==="
