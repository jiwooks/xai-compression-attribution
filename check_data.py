"""
STEP 0: 데이터 점검 (torch 불필요)

하는 일
  1) Waterbirds metadata.csv 를 읽어 split/group 분포를 출력
  2) Waterbirds 이미지와 CUB segmentation 마스크의 크기가 일치하는지 확인
  3) 마스크를 겹친 overlay 이미지를 저장해 눈으로 확인

왜 중요한가
  Attr_bg(배경에 할당된 attribution 비율)를 계산하려면 "새 영역" 마스크가
  Waterbirds 이미지와 픽셀 단위로 정렬되어야 한다. 공식 생성 스크립트는
  배경을 '새 이미지 크기'에 맞춰 잘라 붙이고 CUB와 같은 상대경로로 저장하므로,
  CUB_200_2011/segmentations 의 같은 경로 파일이 그대로 마스크가 된다.
  --> 이 스크립트로 반드시 검증하고 넘어갈 것.

사용법
  python check_data.py \
      --wb_root  /path/to/waterbird_complete95_forest2water2 \
      --cub_root /path/to/CUB_200_2011 \
      --out_dir  ./data_check
"""

import argparse
import os

import numpy as np
import pandas as pd
from PIL import Image

GROUP_NAMES = {
    0: "landbird on land  (majority)",
    1: "landbird on water (minority)",
    2: "waterbird on land (minority)",
    3: "waterbird on water(majority)",
}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


def load_metadata(wb_root):
    path = os.path.join(wb_root, "metadata.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"metadata.csv 를 찾을 수 없음: {path}")
    df = pd.read_csv(path)
    for col in ["img_filename", "y", "split", "place"]:
        if col not in df.columns:
            raise KeyError(f"metadata.csv 에 '{col}' 컬럼이 없음. 컬럼: {list(df.columns)}")
    # group = y * 2 + place  (0..3)
    df["group"] = df["y"] * 2 + df["place"]
    return df


def print_distribution(df):
    print("=" * 68)
    print("split / group 분포")
    print("=" * 68)
    for split in sorted(df["split"].unique()):
        sub = df[df["split"] == split]
        print(f"\n[{SPLIT_NAMES.get(split, split)}]  총 {len(sub)}장")
        for g in range(4):
            n = int((sub["group"] == g).sum())
            pct = 100.0 * n / len(sub) if len(sub) else 0.0
            print(f"   group {g}  {GROUP_NAMES[g]:32s} {n:6d}  ({pct:5.1f}%)")

    # 학습셋에서 새-배경 상관이 실제로 95% 근처인지 확인
    tr = df[df["split"] == 0]
    if len(tr):
        match = int((tr["y"] == tr["place"]).sum())
        print(f"\n학습셋에서 새 종류와 배경이 '일치'하는 비율: {100.0*match/len(tr):.1f}%")
        print("  -> 95% 내외면 정상 (shortcut 이 존재하는 상태)")


def check_masks(df, wb_root, cub_root, out_dir, n_check=200, n_overlay=6):
    print("\n" + "=" * 68)
    print(f"마스크 정렬 검사 (무작위 {n_check}장)")
    print("=" * 68)

    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.RandomState(0)
    idx = rng.choice(len(df), size=min(n_check, len(df)), replace=False)

    n_ok = n_missing = n_mismatch = 0
    fg_fracs = []
    saved = 0

    for k, i in enumerate(idx):
        rel = df.iloc[i]["img_filename"]
        img_path = os.path.join(wb_root, rel)
        seg_path = os.path.join(cub_root, "segmentations", rel.replace(".jpg", ".png"))

        if not (os.path.exists(img_path) and os.path.exists(seg_path)):
            n_missing += 1
            continue

        img = Image.open(img_path).convert("RGB")
        seg = Image.open(seg_path).convert("L")

        if img.size != seg.size:
            n_mismatch += 1
            if n_mismatch <= 3:
                print(f"  [크기 불일치] {rel}: img={img.size} seg={seg.size}")
            continue

        n_ok += 1
        m = np.asarray(seg) > 127           # True = 새(전경)
        fg_fracs.append(float(m.mean()))

        if saved < n_overlay:
            a = np.asarray(img).astype(np.float32)
            a[~m] *= 0.35                   # 배경을 어둡게 -> 새만 밝게 남음
            name = rel.replace("/", "__").replace(".jpg", "_overlay.jpg")
            Image.fromarray(a.astype(np.uint8)).save(os.path.join(out_dir, name))
            saved += 1

    print(f"\n  정상 정렬 : {n_ok}")
    print(f"  파일 없음 : {n_missing}")
    print(f"  크기 불일치: {n_mismatch}")

    if fg_fracs:
        fg = np.array(fg_fracs)
        print(f"\n  새 영역이 차지하는 평균 면적 : {fg.mean():.3f}  (중앙값 {np.median(fg):.3f})")
        print(f"  -> 균일한 attribution map 의 Attr_bg 기댓값 = {1-fg.mean():.3f}")
        print("     (실제 Attr_bg 가 이 값보다 낮아야 '새를 본다'고 말할 수 있음)")

    print(f"\n  overlay 이미지 {saved}장 저장: {os.path.abspath(out_dir)}")
    print("  -> 새 윤곽이 정확히 밝게 남았는지 직접 눈으로 확인할 것")

    if n_mismatch == 0 and n_ok > 0:
        print("\n  [통과] CUB segmentation 을 그대로 마스크로 사용 가능")
    else:
        print("\n  [주의] 불일치가 있음. 마스크 생성 방식을 다시 확인해야 함")

    return n_ok, n_missing, n_mismatch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wb_root", required=True, help="waterbird_complete95_forest2water2 경로")
    ap.add_argument("--cub_root", default=None, help="CUB_200_2011 경로 (마스크 검사용)")
    ap.add_argument("--out_dir", default="./data_check")
    ap.add_argument("--n_check", type=int, default=200)
    args = ap.parse_args()

    df = load_metadata(args.wb_root)
    print(f"metadata.csv 로드 완료: 총 {len(df)}장\n")
    print_distribution(df)

    if args.cub_root:
        check_masks(df, args.wb_root, args.cub_root, args.out_dir, args.n_check)
    else:
        print("\n[건너뜀] --cub_root 를 주면 마스크 정렬까지 검사함")


if __name__ == "__main__":
    main()
