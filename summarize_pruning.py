"""
STEP 3 결과 요약. runs/ 의 final_test.json 과 shortcut_<tag>.json 을 모아 sparsity 별 표를 만든다.

torch 없이 numpy 만 있으면 된다. 마스터 노드에서는 python 을 실행하지 말 것.

비교 기준
  dense : runs/erm_seed<k>      (pruning, fine-tuning 모두 없음)
  ft0   : runs/prune_sp00_seed<k> (pruning 없이 같은 횟수만큼 fine-tuning 한 대조군)
seed 별로 짝지어 차이를 낸 뒤 seed 평균을 쓴다.

판정
  - 평균 정확도가 dense 대비 2pt 넘게 떨어진 seed 는 메인 분석에서 제외한다 (CLAUDE.md §4).
  - H1: S 변화가 dense seed 표준편차의 2배를 넘는지 표시한다 (CLAUDE.md §1 기각 조건).

S 분해(S_CW, S_WC, d_correct, d_wrong)는 summarize_shortcut.py 와 같은 정의다.

사용법
  python summarize_pruning.py --out results/pruning_k5.md
"""

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

MAX_DROP = 0.02


def decompose(d):
    """S 를 정답 붕괴(C->W)와 오답 교정(W->C)으로 나눈다. summarize_shortcut.py 와 같은 식."""
    n = {"correct": 0, "wrong": 0}
    opp = {"correct": 0.0, "wrong": 0.0}
    same = {"correct": 0.0, "wrong": 0.0}
    for key, v in d["by_group_and_orig_correct"].items():
        c = key.split("|")[1]
        n[c] += v["n_images"]
        opp[c] += v["n_images"] * v["BFR_opp"]
        same[c] += v["n_images"] * v["BFR_same"]
    N = n["correct"] + n["wrong"]
    out = {
        "S_CW": (opp["correct"] - same["correct"]) / N,
        "S_WC": (opp["wrong"] - same["wrong"]) / N,
        "d_correct": (opp["correct"] - same["correct"]) / n["correct"],
        "d_wrong": (opp["wrong"] - same["wrong"]) / n["wrong"] if n["wrong"] else float("nan"),
    }
    assert abs(out["S_CW"] + out["S_WC"] - d["overall"]["S"]) < 1e-9, "S 분해 항등식 불일치"
    return out


def load(tag):
    runs = defaultdict(dict)   # name -> seed -> record
    for path in sorted(glob.glob(os.path.join("runs", "*", "final_test.json"))):
        run = os.path.basename(os.path.dirname(path))
        m = re.fullmatch(r"(erm|prune_sp\d+)_seed(\d+)", run)
        sc = os.path.join(os.path.dirname(path), f"shortcut_{tag}.json")
        if not m or not os.path.exists(sc):
            continue
        with open(path) as f:
            ft = json.load(f)
        with open(sc) as f:
            s = json.load(f)
        rec = {"avg": ft["avg_acc"], "wga": ft["worst_group_acc"],
               "sparsity": ft.get("sparsity_actual", 0.0),
               "oneshot_avg": ft.get("oneshot", {}).get("avg_acc"),
               "oneshot_wga": ft.get("oneshot", {}).get("worst_group_acc"),
               "S": s["overall"]["S"], "ci": s["overall"].get("ci95_two_way", {}).get("S")}
        rec.update(decompose(s))
        runs[m.group(1)][int(m.group(2))] = rec
    return runs


def fmt(v, k=1.0, digits=4):
    v = np.asarray([x for x in v if x is not None], dtype=float)
    if len(v) == 0:
        return "-"
    if len(v) == 1:
        return f"{v[0]*k:.{digits}f}"
    return f"{v.mean()*k:.{digits}f} ± {v.std(ddof=1)*k:.{digits}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="k5")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = load(args.tag)
    dense = runs.get("erm", {})
    if not dense:
        raise SystemExit("runs/erm_seed*/ 의 결과가 없다. 저장소 루트에서 실행할 것.")
    s_sd = float(np.std([r["S"] for r in dense.values()], ddof=1)) if len(dense) > 1 else float("nan")
    ft0 = runs.get("prune_sp00", {})
    names = ["erm"] + sorted((n for n in runs if n != "erm"), key=lambda n: int(n.split("sp")[1]))

    L = [f"# STEP 3 pruning 요약 (tag={args.tag})\n",
         "값은 seed 평균 ± seed 표준편차. Δ 는 seed 별로 짝지은 차이의 평균이다.",
         f"dense S 의 seed 표준편차 {s_sd:.4f}, H1 기준(2배) {2*s_sd:.4f}.",
         f"평균 정확도가 dense 대비 {MAX_DROP*100:.0f}pt 넘게 떨어진 seed 는 '제외' 에 센다.\n",
         "## 1. 정확도와 S\n",
         "| 모델 | sparsity | seed | avg | WGA | Δavg vs dense | 제외 | S | ΔS vs dense | ΔS vs ft0 | H1 기준 초과 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name in names:
        R = runs[name]
        seeds = sorted(R)
        d_avg = [R[k]["avg"] - dense[k]["avg"] for k in seeds if k in dense]
        excluded = sum(x < -MAX_DROP for x in d_avg)
        dS = [R[k]["S"] - dense[k]["S"] for k in seeds if k in dense]
        dS0 = [R[k]["S"] - ft0[k]["S"] for k in seeds if k in ft0]
        over = []
        for v in (dS, dS0):
            over.append("예" if v and abs(np.mean(v)) > 2 * s_sd else ("아니오" if v else "-"))
        label = "dense" if name == "erm" else ("ft0 (fine-tuning 만)" if name == "prune_sp00" else name)
        L.append(f"| {label} | {fmt([R[k]['sparsity'] for k in seeds], 100, 1)}% | {len(seeds)} "
                 f"| {fmt([R[k]['avg'] for k in seeds], 100, 2)} | {fmt([R[k]['wga'] for k in seeds], 100, 2)} "
                 f"| {fmt([np.mean(d_avg)] if name != 'erm' else [], 100, 2)} | {excluded}/{len(seeds)} "
                 f"| {fmt([R[k]['S'] for k in seeds])} "
                 f"| {fmt([np.mean(dS)] if name != 'erm' else [])} "
                 f"| {fmt([np.mean(dS0)] if name not in ('erm', 'prune_sp00') and dS0 else [])} "
                 f"| {'-' if name == 'erm' else ' / '.join(over)} |")
    L.append("\nH1 기준 초과 열은 'dense 대비 / ft0 대비' 순이다.\n")

    L += ["## 2. S 분해\n",
          "| 모델 | S_C→W (정답 붕괴) | S_W→C (오답 교정) | d_correct | d_wrong |",
          "| --- | --- | --- | --- | --- |"]
    for name in names:
        R = runs[name]
        L.append(f"| {name} | " + " | ".join(fmt([R[k][c] for k in sorted(R)])
                                             for c in ("S_CW", "S_WC", "d_correct", "d_wrong")) + " |")
    L.append("\nd_correct / d_wrong 는 원래 맞은/틀린 이미지 한 장당 (반대 배경 뒤집힘 − 같은 타입 배경 뒤집힘) 이다.\n")

    L += ["## 3. fine-tuning 전 (one-shot) 정확도\n",
          "| 모델 | avg | WGA |", "| --- | --- | --- |"]
    for name in names[1:]:
        R = runs[name]
        L.append(f"| {name} | {fmt([R[k]['oneshot_avg'] for k in sorted(R)], 100, 2)} "
                 f"| {fmt([R[k]['oneshot_wga'] for k in sorted(R)], 100, 2)} |")

    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
