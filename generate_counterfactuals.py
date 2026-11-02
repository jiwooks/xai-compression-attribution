"""
STEP 2. 반사실 이미지 생성.

Waterbirds 이미지마다 같은 새를 다른 배경에 합성한다.
  x_opp  : 반대 타입 배경 (물 <-> 숲)
  x_same : 같은 타입의 다른 배경 (대조군)

합성은 공식 generate_waterbirds.py / dataset_utils.py 와 같은 방식이다.
  - CUB 원본 이미지와 소프트 마스크(0~1)를 그대로 사용
  - crop_and_resize -> combine_and_mask
  - .jpg 기본 품질로 저장 (원본과 압축 흔적을 맞춘다)
Image.ANTIALIAS 만 Pillow 10 이후 이름인 Image.LANCZOS 로 바꿨다.

배경은 Places365 평가셋에서 가져온다. Waterbirds 는 학습 풀(data_large)만 썼으므로
어떤 원본 이미지에도 쓰이지 않은 held-out 배경이다.
"""

import argparse
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import binary_erosion

from waterbirds import SPLIT_ID

CONDITIONS = ("opp", "same")


def crop_and_resize(source_img, target_img):
    """공식 dataset_utils.crop_and_resize 와 동일 (LANCZOS 이름만 변경)."""
    source_width, source_height = source_img.size
    target_width, target_height = target_img.size

    if (source_width < target_width) or (source_height < target_height):
        width_resize = (target_width, int((target_width / source_width) * source_height))
        if (width_resize[0] >= target_width) and (width_resize[1] >= target_height):
            source_resized = source_img.resize(width_resize, Image.LANCZOS)
        else:
            height_resize = (int((target_height / source_height) * source_width), target_height)
            assert (height_resize[0] >= target_width) and (height_resize[1] >= target_height)
            source_resized = source_img.resize(height_resize, Image.LANCZOS)
        return crop_and_resize(source_resized, target_img)

    source_aspect = source_width / source_height
    target_aspect = target_width / target_height

    if source_aspect > target_aspect:
        new_source_width = int(target_aspect * source_height)
        offset = (source_width - new_source_width) // 2
        box = (offset, 0, source_width - offset, source_height)
    else:
        new_source_height = int(source_width / target_aspect)
        offset = (source_height - new_source_height) // 2
        box = (0, offset, source_width, source_height - offset)

    return source_img.crop(box).resize((target_width, target_height), Image.LANCZOS)


def combine_and_mask(img_new, mask, img_black):
    """공식 dataset_utils.combine_and_mask 와 동일."""
    img_resized_np = np.asarray(crop_and_resize(img_new, img_black))
    img_masked_np = np.around(img_resized_np * (1 - mask)).astype(np.uint8)
    return Image.fromarray(np.asarray(img_black) + img_masked_np)


def load_bird(cub_root, rel):
    """공식 스크립트와 같은 방식으로 새만 남긴 이미지와 소프트 마스크를 만든다."""
    img_np = np.asarray(Image.open(os.path.join(cub_root, "images", rel)).convert("RGB"))
    seg_path = os.path.join(cub_root, "segmentations", rel.replace(".jpg", ".png"))
    seg_np = np.asarray(Image.open(seg_path).convert("RGB")) / 255
    return Image.fromarray(np.around(img_np * seg_np).astype(np.uint8)), seg_np


def sample_rows(df, pools, n_bg, seed):
    """이미지별로 타입당 배경 n_bg 장을 고른다. 시드와 img_id 로 결정되어 재현 가능하다."""
    rows = []
    for rec in df.itertuples(index=False):
        rng = np.random.default_rng([seed, int(rec.img_id)])
        for cond in CONDITIONS:
            bg_type = rec.place if cond == "same" else 1 - rec.place
            pool = pools[bg_type]
            for k, j in enumerate(rng.choice(len(pool), size=n_bg, replace=False)):
                rows.append({
                    "img_id": rec.img_id, "img_filename": rec.img_filename,
                    "y": rec.y, "place": rec.place, "condition": cond, "k": k,
                    "bg_file": pool[j],
                    "out_path": os.path.join(cond, str(k), rec.img_filename),
                })
    return rows


def _render(job):
    row, cub_root, bg_dir, out_dir = job
    dst = os.path.join(out_dir, row["out_path"])
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    bird, mask = load_bird(cub_root, row["img_filename"])
    place = Image.open(os.path.join(bg_dir, row["bg_file"])).convert("RGB")
    combine_and_mask(place, mask, bird).save(dst)


def check_alignment(rows, wb_root, cub_root, out_dir, n_check=50):
    """새 안쪽 픽셀은 공식 이미지와 같고 배경 픽셀은 달라야 한다.

    경계는 JPEG 압축이 새 배경의 영향을 받아 원래 차이가 나므로 8픽셀 안쪽만 비교한다.
    """
    bird_err, bg_err = [], []
    seen = set()
    skipped = 0
    for row in rows:
        rel = row["img_filename"]
        if rel in seen or len(bird_err) >= n_check:
            continue
        seen.add(rel)
        _, mask = load_bird(cub_root, rel)
        # CUB 마스크는 소프트 값이고 최댓값이 0.8 인 이미지나 아주 작은 마스크도 있어,
        # 그런 경우 비교할 안쪽 픽셀이 없다.
        fg = binary_erosion(mask[:, :, 0] > 0.99, iterations=8)
        if not fg.any():
            skipped += 1
            continue
        official = np.asarray(Image.open(os.path.join(wb_root, rel)).convert("RGB"), dtype=float)
        new = np.asarray(Image.open(os.path.join(out_dir, row["out_path"])).convert("RGB"), dtype=float)
        diff = np.abs(official - new).mean(axis=2)
        bird_err.append(diff[fg].mean())
        bg_err.append(diff[mask[:, :, 0] < 0.01].mean())
    if not bird_err:
        raise SystemExit("[실패] 정렬을 검사할 이미지가 없다.")
    print(f"정렬 검사 {len(bird_err)}장 (제외 {skipped}장): 새 안쪽 평균 차이 {np.mean(bird_err):.2f}, "
          f"최대 {np.max(bird_err):.2f} / 배경 영역 평균 차이 {np.mean(bg_err):.2f} (0~255)")
    return float(np.max(bird_err))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wb_root", required=True)
    ap.add_argument("--cub_root", required=True)
    ap.add_argument("--bg_root", required=True, help="bg_manifest.csv 와 images/ 가 있는 경로")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--split", default="test", choices=list(SPLIT_ID))
    ap.add_argument("--n_bg", type=int, default=5, help="이미지당 조건별 배경 수")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=0, help="점검용. 이미지 수 제한")
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--max_bird_diff", type=float, default=8.0,
                    help="정렬 검사에서 허용할 새 안쪽 영역의 최대 평균 차이")
    args = ap.parse_args()

    df = pd.read_csv(os.path.join(args.wb_root, "metadata.csv"))
    df = df[df["split"] == SPLIT_ID[args.split]].reset_index(drop=True)
    if args.subset > 0:
        df = df.sample(n=min(args.subset, len(df)), random_state=args.seed).reset_index(drop=True)

    bg = pd.read_csv(os.path.join(args.bg_root, "bg_manifest.csv"))
    pools = {t: sorted(bg.loc[bg["place"] == t, "file"]) for t in (0, 1)}
    print(f"이미지 {len(df)}장, 배경 land {len(pools[0])} / water {len(pools[1])}, n_bg={args.n_bg}")

    rows = sample_rows(df, pools, args.n_bg, args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    jobs = [(r, args.cub_root, os.path.join(args.bg_root, "images"), args.out_dir) for r in rows]
    with Pool(args.num_workers) as pool:
        for i, _ in enumerate(pool.imap_unordered(_render, jobs, chunksize=32), 1):
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(args.out_dir, "manifest.csv"), index=False)
    print(f"생성 완료: {len(rows)}장 -> {args.out_dir}")

    if check_alignment(rows, args.wb_root, args.cub_root, args.out_dir) > args.max_bird_diff:
        raise SystemExit("[실패] 새 영역이 공식 이미지와 다르다. 합성 또는 경로를 확인할 것.")


if __name__ == "__main__":
    main()
