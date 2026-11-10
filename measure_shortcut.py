"""
STEP 2. 행동 기반 shortcut 지수 측정 (attribution 미사용).

  S = BFR_opp - BFR_same

BFR 는 원본 예측과 반사실 예측이 달라진 비율이다.
이미지별로 배경 k 장의 flip 비율을 평균한 뒤 이미지 전체를 평균한다.
ci95 는 이미지만 재표본한 bootstrap 이고, ci95_two_way 는 배경도 함께 재표본한다.
배경 풀이 작아 같은 배경이 여러 이미지에 반복되므로 ci95_two_way 를 기본으로 보고한다.

이미지별 행동 지표 (CLAUDE.md §2, 2026-11-10)
  Δp = mean p(waterbird | 물 배경 반사실) − mean p(waterbird | 땅 배경 반사실)
원본 사진은 쓰지 않는다. flip 은 지금처럼 argmax 로 센다.

--save_probs 를 주면 <out>_rows.csv.gz (반사실 행별 확률)와 <out>_dp.csv.gz (이미지별 Δp)를 저장한다.
--regress_check 를 주면 그 JSON 의 S, BFR, 그룹별 값과 새 결과를 비교해 다르면 저장하지 않고 멈춘다.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from train_erm import build_model
from waterbirds import GROUP_NAMES, N_GROUPS, SPLIT_ID, build_transform


class ImageList(Dataset):
    def __init__(self, root, rel_paths, transform):
        self.root, self.rel_paths, self.transform = root, list(rel_paths), transform

    def __len__(self):
        return len(self.rel_paths)

    def __getitem__(self, i):
        img = Image.open(os.path.join(self.root, self.rel_paths[i])).convert("RGB")
        return self.transform(img)


@torch.no_grad()
def predict(model, root, rel_paths, device, batch_size, num_workers):
    """argmax 예측과 softmax 의 waterbird(y=1) 확률을 돌려준다."""
    transform = build_transform(train=False)
    loader = DataLoader(ImageList(root, rel_paths, transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    preds, p_w = [], []
    for x in loader:
        logits = model(x.to(device, non_blocking=True))
        preds.append(logits.argmax(1).cpu().numpy())
        p_w.append(logits.float().softmax(1)[:, 1].cpu().numpy())
    return np.concatenate(preds), np.concatenate(p_w)


def regress_check(old, new):
    """기존 JSON 과 flip 기반 값이 같은지 본다. 다른 항목 목록을 돌려준다."""
    bad = []

    def cmp(path, a, b):
        if isinstance(a, dict):
            for k in a:
                if k not in b:
                    bad.append(f"{path}.{k} 없음")
                else:
                    cmp(f"{path}.{k}", a[k], b[k])
        elif isinstance(a, list):
            for i, (x, y) in enumerate(zip(a, b)):
                cmp(f"{path}[{i}]", x, y)
        elif abs(float(a) - float(b)) > 1e-12:
            bad.append(f"{path}: {a} -> {b}")

    for k in ("BFR_opp", "BFR_same", "S", "n_images", "ci95", "ci95_two_way"):
        if k in old["overall"]:
            cmp(f"overall.{k}", old["overall"][k], new["overall"][k])
    cmp("groups", old["groups"], new["groups"])
    cmp("by_group_and_orig_correct", old["by_group_and_orig_correct"], new["by_group_and_orig_correct"])
    cmp("avg_acc_orig", old["avg_acc_orig"], new["avg_acc_orig"])
    return bad


def bootstrap_ci(opp, same, n_boot, seed):
    rng = np.random.default_rng(seed)
    n = len(opp)
    idx = rng.integers(0, n, size=(n_boot, n))
    bo, bs = opp[idx].mean(1), same[idx].mean(1)
    return {name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for name, v in (("BFR_opp", bo), ("BFR_same", bs), ("S", bo - bs))}


def two_way_ci(rows, n_boot, seed):
    """이미지와 배경을 독립적으로 재표본한 S 의 95% 신뢰구간."""
    rng = np.random.default_rng(seed)
    img = pd.factorize(rows["img_id"])[0]
    bg = pd.factorize(rows["bg_file"])[0]
    flip = rows["flip"].to_numpy()
    is_opp = (rows["condition"] == "opp").to_numpy()
    n_img, n_bg = img.max() + 1, bg.max() + 1
    bo, bs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        w = (rng.multinomial(n_img, np.full(n_img, 1 / n_img))[img]
             * rng.multinomial(n_bg, np.full(n_bg, 1 / n_bg))[bg]).astype(float)
        bo[b] = (w[is_opp] * flip[is_opp]).sum() / w[is_opp].sum()
        bs[b] = (w[~is_opp] * flip[~is_opp]).sum() / w[~is_opp].sum()
    return {name: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
            for name, v in (("BFR_opp", bo), ("BFR_same", bs), ("S", bo - bs))}


def summarize(table, n_boot, seed):
    opp, same = table["flip_opp"].to_numpy(), table["flip_same"].to_numpy()
    return {
        "n_images": int(len(table)),
        "BFR_opp": float(opp.mean()),
        "BFR_same": float(same.mean()),
        "S": float(opp.mean() - same.mean()),
        "ci95": bootstrap_ci(opp, same, n_boot, seed),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--wb_root", required=True)
    ap.add_argument("--cf_dir", required=True, help="generate_counterfactuals.py 의 출력 경로")
    ap.add_argument("--out", required=True, help="결과 JSON 경로")
    ap.add_argument("--split", default="test", choices=list(SPLIT_ID))
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--expect_avg_acc", type=float, default=None,
                    help="final_test.json 의 avg_acc. 주면 원본 정확도와 대조한다")
    ap.add_argument("--save_probs", action="store_true",
                    help="반사실 행별 확률(<out>_rows.csv.gz)과 이미지별 Δp(<out>_dp.csv.gz)를 저장")
    ap.add_argument("--regress_check", default=None,
                    help="기존 결과 JSON. flip 기반 값이 하나라도 다르면 저장하지 않고 멈춘다")
    args = ap.parse_args()
    old = None
    if args.regress_check:
        with open(args.regress_check) as f:
            old = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(device, pretrained=False)
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    manifest = pd.read_csv(os.path.join(args.cf_dir, "manifest.csv"))
    images = manifest.drop_duplicates("img_id")[["img_id", "img_filename", "y", "place"]]
    images = images.sort_values("img_id").reset_index(drop=True)

    orig, p_orig = predict(model, args.wb_root, images["img_filename"], device,
                           args.batch_size, args.num_workers)
    images["pred_orig"] = orig
    images["p_w_orig"] = p_orig
    avg_acc = float((orig == images["y"].to_numpy()).mean())
    print(f"원본 정확도 {avg_acc*100:.2f}% ({len(images)}장)")
    if args.expect_avg_acc is not None and abs(avg_acc - args.expect_avg_acc) > 1e-6:
        raise SystemExit(f"[실패] final_test.json 의 avg_acc {args.expect_avg_acc} 와 다르다.")

    manifest["pred_cf"], manifest["p_w"] = predict(model, args.cf_dir, manifest["out_path"], device,
                                                   args.batch_size, args.num_workers)
    manifest = manifest.merge(images[["img_id", "pred_orig"]], on="img_id")
    manifest["flip"] = (manifest["pred_cf"] != manifest["pred_orig"]).astype(float)
    # same 은 원본과 같은 타입, opp 는 반대 타입 배경 (generate_counterfactuals.sample_rows)
    manifest["bg_place"] = np.where(manifest["condition"] == "same",
                                    manifest["place"], 1 - manifest["place"]).astype(int)

    wide = manifest.pivot_table(index="img_id", columns="condition", values="flip", aggfunc="mean")
    table = images.merge(wide.rename(columns={"opp": "flip_opp", "same": "flip_same"}),
                         left_on="img_id", right_index=True)
    table["group"] = table["y"] * 2 + table["place"]
    table["correct_orig"] = table["pred_orig"] == table["y"]

    result = {"ckpt": args.ckpt, "avg_acc_orig": avg_acc,
              "n_bg": int(manifest["k"].nunique()),
              "overall": summarize(table, args.n_boot, args.seed),
              "groups": {GROUP_NAMES[g]: summarize(table[table["group"] == g], args.n_boot, args.seed)
                         for g in range(N_GROUPS) if (table["group"] == g).any()}}
    result["overall"]["ci95_two_way"] = two_way_ci(manifest, args.n_boot, args.seed)
    result["by_group_and_orig_correct"] = {
        f"{GROUP_NAMES[g]}|{'correct' if c else 'wrong'}": {
            "n_images": int(len(sub)),
            "BFR_opp": float(sub["flip_opp"].mean()),
            "BFR_same": float(sub["flip_same"].mean()),
        }
        for (g, c), sub in table.groupby(["group", "correct_orig"])
    }

    if old is not None:
        bad = regress_check(old, result)
        if bad:
            print("[회귀 실패] 기존 결과와 다르다. 저장하지 않는다.")
            for b in bad[:30]:
                print("  ", b)
            raise SystemExit(1)
        print(f"[회귀 확인 통과] {args.regress_check} 와 S·BFR·그룹별·CI 값이 모두 같다.")

    # 이미지별 Δp: 물 배경 반사실 평균 확률 − 땅 배경 반사실 평균 확률
    pw = manifest.pivot_table(index="img_id", columns="bg_place", values="p_w", aggfunc="mean")
    table = table.merge(pw[1].sub(pw[0]).rename("dp"), left_on="img_id", right_index=True)
    assert table["dp"].notna().all() and len(table) == len(images), "Δp 계산에 빠진 이미지가 있다"
    result["dp"] = {
        "definition": "mean p(waterbird|water bg cf) - mean p(waterbird|land bg cf), per image",
        "overall": float(table["dp"].mean()),
        "groups": {GROUP_NAMES[g]: float(sub["dp"].mean()) for g, sub in table.groupby("group")},
        "by_group_and_orig_correct": {
            f"{GROUP_NAMES[g]}|{'correct' if c else 'wrong'}": float(sub["dp"].mean())
            for (g, c), sub in table.groupby(["group", "correct_orig"])},
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)
    stem = os.path.splitext(args.out)[0]
    table.to_csv(stem + "_per_image.csv", index=False)
    if args.save_probs:
        rows = manifest.sort_values(["img_id", "condition", "k"])[
            ["img_id", "condition", "k", "bg_file", "bg_place", "pred_cf", "p_w", "flip"]]
        rows = rows.astype({"flip": int})
        rows.to_csv(stem + "_rows.csv.gz", index=False, float_format="%.6g")
        table[["img_id", "group", "y", "correct_orig", "dp", "p_w_orig"]].astype(
            {"correct_orig": int}).to_csv(stem + "_dp.csv.gz", index=False, float_format="%.6g")
        print(f"저장: {stem}_rows.csv.gz, {stem}_dp.csv.gz")

    o = result["overall"]
    print(f"BFR_opp {o['BFR_opp']:.4f}  BFR_same {o['BFR_same']:.4f}  S {o['S']:.4f} "
          f"(95% CI 이미지 {o['ci95']['S'][0]:.4f} ~ {o['ci95']['S'][1]:.4f}, "
          f"이미지+배경 {o['ci95_two_way']['S'][0]:.4f} ~ {o['ci95_two_way']['S'][1]:.4f})")
    for name, g in result["groups"].items():
        print(f"  {name:16s} n={g['n_images']:5d}  opp {g['BFR_opp']:.4f}  "
              f"same {g['BFR_same']:.4f}  S {g['S']:.4f}")
    for name, g in result["by_group_and_orig_correct"].items():
        print(f"  {name:24s} n={g['n_images']:5d}  opp {g['BFR_opp']:.4f}  same {g['BFR_same']:.4f}  "
              f"Δp {result['dp']['by_group_and_orig_correct'][name]:+.4f}")
    print(f"평균 Δp {result['dp']['overall']:+.4f}")


if __name__ == "__main__":
    main()
