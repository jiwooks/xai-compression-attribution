#!/usr/bin/env bash
# 주 결과 회귀 확인 (PREREG_seeds5.md "주 결과를 지키는 장치").
# 기본값(seed 0 1 2)으로 분석 스크립트를 다시 돌려 임시 폴더에 쓰고, 저장소의 results/*.md 와 그림을 비교한다.
# 그림 링크 줄은 경로만 달라서 빼고 비교하고, 그림은 픽셀 단위로 비교한다.
# 주의: results/figs 의 h2a_*, h2b_* 는 클러스터에서 그렸다. 로컬 PC 에서 돌리면 폰트·렌더링 차이로 픽셀이 다르게 나온다
#       (2026-11-17 확인: 같은 PC 에서 바뀌기 전 analyze_h2.py 로 그린 그림과는 픽셀까지 같았다).
#       그림까지 비교하려면 클러스터에서 srun 으로 돌린다.
#
#   bash scripts/regress_main_results.sh [임시 폴더]          # 저장소 루트에서. 로컬 PC 또는 srun CPU 1개
#   (클러스터) srun -p debug_ugrad --ntasks=1 --cpus-per-task=1 --mem=24G -t 2:00:00 \
#              bash scripts/regress_main_results.sh </dev/null
set -euo pipefail
OUT="${1:-${TMPDIR:-/tmp}/xai_regress}"
PY="${PY:-python}"
rm -rf "$OUT" && mkdir -p "$OUT/figs"
echo "=== 회귀 확인 → $OUT ==="

"$PY" summarize_pruning.py --out "$OUT/pruning_k5.md" > /dev/null
"$PY" analyze_h1.py --out "$OUT/h1_k5.md" > /dev/null
"$PY" analyze_h2.py --out "$OUT/attribution_k5.md" --fig_dir "$OUT/figs" > /dev/null
"$PY" analyze_stress.py --out "$OUT/stress_k5.md" --fig_dir "$OUT/figs" > /dev/null

fail=0
for f in pruning_k5.md h1_k5.md attribution_k5.md stress_k5.md; do
  if diff <(grep -v '](' "results/$f" | tr -d '\r') <(grep -v '](' "$OUT/$f" | tr -d '\r') > "$OUT/$f.diff"; then
    echo "[같음] results/$f"
  else
    echo "[다름] results/$f"; head -20 "$OUT/$f.diff"; fail=1
  fi
done

"$PY" - "$OUT/figs" <<'EOF' || fail=1
import os, sys
import matplotlib.image as mpimg
import numpy as np
bad = 0
for name in sorted(os.listdir(sys.argv[1])):
    a = mpimg.imread(os.path.join("results/figs", name))
    b = mpimg.imread(os.path.join(sys.argv[1], name))
    same = a.shape == b.shape and np.array_equal(a, b)
    print(f"[{'같음' if same else '다름'}] results/figs/{name}")
    bad += not same
sys.exit(1 if bad else 0)
EOF

[ "$fail" = 0 ] && echo "=== 회귀 확인 통과 ===" || { echo "=== 회귀 확인 실패 ==="; exit 1; }
