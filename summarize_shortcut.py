"""
measure_shortcut.py 가 남긴 shortcut_*.json 들을 모아 분석용 표를 만든다 (재실행 불필요).

같은 설정의 seed 들(runs/<이름>_seed<k>/)은 묶어서 평균과 범위를 낸다.
torch 가 필요 없으므로 로컬 PC 에서 돌려도 된다 (결과 JSON 은 git 에 있다).
마스터 노드에서는 python 을 실행하지 말 것.

출력 표
  1. 전체      : BFR_opp, BFR_same, S
  2. 분해      : S = S_CW + S_WC (이진 분류라 정확히 성립)
                 d_correct, d_wrong (정확도와 무관한 이미지당 배경 민감도)
  3. 그룹별    : BFR_opp, BFR_same, S
  4. 그룹 x 원본 정오답 : 반대/같은 타입 배경에서 뒤집힌 비율과 방향

사용법
  python summarize_shortcut.py                        # runs/*/shortcut_k5.json 전부
  python summarize_shortcut.py --tag k5 --out results/shortcut_k5.md
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

GROUPS = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
GROUP_KO = {"landbird/land": "g0 땅새/땅", "landbird/water": "g1 땅새/물",
            "waterbird/land": "g2 물새/땅", "waterbird/water": "g3 물새/물"}
# 반대 배경이 미는 방향: 땅 배경 -> landbird, 물 배경 -> waterbird
OPP_PUSHES_TO_CORRECT = {"landbird/land": False, "landbird/water": True,
                         "waterbird/land": True, "waterbird/water": False}
SMALL_N = 50


def decompose(d):
    """S 를 정답 붕괴(C->W)와 오답 교정(W->C)으로 나누고, 조건부 민감도를 계산한다."""
    cells = d["by_group_and_orig_correct"]
    n = {"correct": 0, "wrong": 0}
    opp = {"correct": 0.0, "wrong": 0.0}
    same = {"correct": 0.0, "wrong": 0.0}
    for key, v in cells.items():
        c = key.split("|")[1]
        n[c] += v["n_images"]
        opp[c] += v["n_images"] * v["BFR_opp"]
        same[c] += v["n_images"] * v["BFR_same"]
    N = n["correct"] + n["wrong"]
    out = {
        "acc": n["correct"] / N,
        "S_CW": (opp["correct"] - same["correct"]) / N,
        "S_WC": (opp["wrong"] - same["wrong"]) / N,
        "d_correct": (opp["correct"] - same["correct"]) / n["correct"],
        "d_wrong": (opp["wrong"] - same["wrong"]) / n["wrong"] if n["wrong"] else float("nan"),
    }
    assert abs(out["S_CW"] + out["S_WC"] - d["overall"]["S"]) < 1e-9, "S 분해 항등식 불일치"
    return out


def agg(vals, pct=False, digits=4):
    v = np.asarray(vals, dtype=float)
    k = 100.0 if pct else 1.0
    if len(v) == 1:
        return f"{v[0]*k:.{digits}f}"
    return f"{v.mean()*k:.{digits}f} ({v.min()*k:.{digits}f}~{v.max()*k:.{digits}f})"


def load_runs(tag):
    runs = defaultdict(list)
    for path in sorted(glob.glob(os.path.join("runs", "*", f"shortcut_{tag}.json"))):
        run = os.path.basename(os.path.dirname(path))
        name = re.sub(r"_seed\d+$", "", run)
        with open(path) as f:
            runs[name].append((run, json.load(f)))
    return runs


def report(name, items):
    L = []
    seeds = ", ".join(r for r, _ in items)
    L.append(f"## {name}\n")
    L.append(f"seed {len(items)}개 ({seeds}). 값은 평균, 괄호는 seed 간 범위.\n")

    # 1. 전체
    D = [d for _, d in items]
    L.append("### 1. 전체\n")
    L.append("| 원본 avg | BFR_opp | BFR_same | S |")
    L.append("| --- | --- | --- | --- |")
    L.append(f"| {agg([d['avg_acc_orig'] for d in D], pct=True, digits=2)} "
             f"| {agg([d['overall']['BFR_opp'] for d in D])} "
             f"| {agg([d['overall']['BFR_same'] for d in D])} "
             f"| {agg([d['overall']['S'] for d in D])} |\n")
    if all("ci95_two_way" in d["overall"] for d in D):
        cis = " / ".join(f"{d['overall']['ci95_two_way']['S'][0]:.4f}~{d['overall']['ci95_two_way']['S'][1]:.4f}"
                         for d in D)
        L.append(f"S 의 95% CI (이미지+배경 재표본, seed 순): {cis}\n")

    # 2. 분해
    P = [decompose(d) for d in D]
    L.append("### 2. 분해 (정확도와 분리)\n")
    L.append("| S_C→W (정답 붕괴) | S_W→C (오답 교정) | d_correct | d_wrong |")
    L.append("| --- | --- | --- | --- |")
    L.append(f"| {agg([p['S_CW'] for p in P])} | {agg([p['S_WC'] for p in P])} "
             f"| {agg([p['d_correct'] for p in P])} | {agg([p['d_wrong'] for p in P])} |\n")
    L.append("S = S_C→W + S_W→C. d_correct / d_wrong 는 원래 맞은/틀린 이미지 한 장당 "
             "(opp 뒤집힘 − same 뒤집힘) 으로, 정확도가 변해도 기계적으로 움직이지 않는다.\n")

    # 3. 그룹별
    L.append("### 3. 그룹별\n")
    L.append("| 그룹 | 장수 | BFR_opp | BFR_same | S |")
    L.append("| --- | --- | --- | --- | --- |")
    for g in GROUPS:
        rows = [d["groups"][g] for d in D if g in d["groups"]]
        if not rows:
            continue
        L.append(f"| {GROUP_KO[g]} | {rows[0]['n_images']:,} "
                 f"| {agg([r['BFR_opp'] for r in rows], pct=True, digits=1)}% "
                 f"| {agg([r['BFR_same'] for r in rows], pct=True, digits=1)}% "
                 f"| {agg([r['S'] for r in rows], digits=3)} |")
    L.append("")

    # 4. 그룹 x 원본 정오답
    L.append("### 4. 그룹 × 원본 정오답\n")
    L.append("| 그룹 | 원본 | 장수 | 반대 배경 방향 | 반대 배경에서 뒤집힘 | 같은 타입 배경에서 뒤집힘 |")
    L.append("| --- | --- | --- | --- | --- | --- |")
    for g in GROUPS:
        for c in ("correct", "wrong"):
            key = f"{g}|{c}"
            rows = [d["by_group_and_orig_correct"].get(key) for d in D]
            rows = [r for r in rows if r is not None]
            if not rows:
                continue
            ns = [r["n_images"] for r in rows]
            n_txt = f"{min(ns):,}" if min(ns) == max(ns) else f"{min(ns):,}~{max(ns):,}"
            warn = " ⚠️" if min(ns) < SMALL_N else ""
            to = "오답으로" if c == "correct" else "정답으로"
            push = "정답 쪽" if OPP_PUSHES_TO_CORRECT[g] else "오답 쪽"
            L.append(f"| {GROUP_KO[g]} | {'맞음' if c == 'correct' else '틀림'} | {n_txt}{warn} | {push} "
                     f"| {agg([r['BFR_opp'] for r in rows], pct=True, digits=1)}% {to} "
                     f"| {agg([r['BFR_same'] for r in rows], pct=True, digits=1)}% {to} |")
    L.append(f"\n⚠️ = 이미지 {SMALL_N}장 미만. 비율이 크게 흔들리므로 해석 비중을 낮춘다.\n")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="k5", help="shortcut_<tag>.json 의 tag")
    ap.add_argument("--out", default=None, help="마크다운으로 저장할 경로 (생략하면 화면 출력만)")
    args = ap.parse_args()

    runs = load_runs(args.tag)
    if not runs:
        raise SystemExit(f"runs/*/shortcut_{args.tag}.json 을 찾지 못했다. 저장소 루트에서 실행할 것.")
    text = f"# shortcut 측정 요약 (tag={args.tag})\n\n" + "\n\n".join(
        report(name, items) for name, items in sorted(runs.items()))
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w") as f:
            f.write(text + "\n")
        print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
