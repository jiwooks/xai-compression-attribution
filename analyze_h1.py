"""
H1 재판정 (CLAUDE.md §4 "2026-11-10 결정 사항" 1~3).

입력: runs/<run>_seed<k>/shortcut_<tag>_rows.csv.gz (반사실 행별 flip, p_w)
      runs/<run>_seed<k>/shortcut_<tag>_dp.csv.gz   (이미지별 원본 정오답)
      runs/<run>_seed<k>/final_test.json            (정확도 제외 판정)

지표 (모델마다)
  S         = BFR_opp − BFR_same
  d_correct = 원래 맞은 이미지의 (opp flip − same flip)
  d_wrong   = 원래 틀린 이미지의 (opp flip − same flip)
  mean_dp   = 평균 Δp = mean p_w(물 배경 반사실) − mean p_w(땅 배경 반사실)
이미지마다 배경이 조건·타입별로 같은 수(5장)라 행 단위 비율 = 이미지 평균의 평균이다.

비교
  ft0 대비(주): 같은 seed 의 prune_sp00 와 짝지은 차이. dense 대비(참고): 같은 seed 의 erm.
  seed 합산 = 포함된 seed 의 짝지은 차이 평균. 정확도가 dense 대비 2pt 넘게 떨어진 seed 는 합산에서 뺀다.

신뢰구간: 짝지은 two-way bootstrap (기본 2,000회)
  - 이미지를 복원추출 (다항 가중치)
  - 배경을 땅/물 타입별로 따로 복원추출 (층화)
  - 모든 모델·seed 에 같은 가중치를 써서 짝을 유지한다 (모든 모델이 같은 반사실 사진을 쓴다).

판정: 95% CI 가 0 을 포함하지 않고 |Δ| ≥ 실질 크기 기준(dense seed 표준편차 × 2, 지표별)일 때만 "변함".

numpy/pandas 만 쓴다. 마스터 노드에서 실행하지 말 것 (srun).
  python analyze_h1.py --out results/h1_k5.md
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd

METRICS = ["S", "d_correct", "d_wrong", "mean_dp"]
LABEL = {"S": "S", "d_correct": "d_correct", "d_wrong": "d_wrong", "mean_dp": "평균 Δp"}
MAX_DROP = 0.02
KEY = ["img_id", "condition", "k"]


def find_runs(tag):
    out = {}
    for name in sorted(os.listdir("runs")):
        m = re.fullmatch(r"(erm|prune_sp\d+)_seed(\d+)", name)
        if m and os.path.exists(os.path.join("runs", name, f"shortcut_{tag}_rows.csv.gz")):
            out[(m.group(1), int(m.group(2)))] = os.path.join("runs", name)
    return out


def load(runs, tag):
    """모든 모델의 행을 같은 순서로 맞추고 지표 계산용 열 행렬을 만든다."""
    ref = None
    cols, avg = {}, {}
    for key, d in runs.items():
        rows = pd.read_csv(os.path.join(d, f"shortcut_{tag}_rows.csv.gz"))
        rows = rows.sort_values(KEY).reset_index(drop=True)
        dp = pd.read_csv(os.path.join(d, f"shortcut_{tag}_dp.csv.gz"))
        if ref is None:
            ref = rows[KEY + ["bg_file", "bg_place"]].copy()
        elif not rows[KEY + ["bg_file", "bg_place"]].equals(ref):
            raise SystemExit(f"[실패] {d} 의 반사실 행이 다른 모델과 다르다")
        corr = rows["img_id"].map(dp.set_index("img_id")["correct_orig"]).to_numpy().astype(bool)
        cols[key] = (rows["flip"].to_numpy(float), rows["p_w"].to_numpy(float), corr)
        with open(os.path.join(d, "final_test.json")) as f:
            avg[key] = json.load(f)["avg_acc"]
    return ref, cols, avg


def design(ref, cols):
    """모델마다 16개 열. W @ X 의 비율로 네 지표가 나온다."""
    opp = (ref["condition"] == "opp").to_numpy(float)
    same = 1.0 - opp
    water = (ref["bg_place"] == 1).to_numpy(float)
    land = 1.0 - water
    keys = list(cols)
    blocks = []
    for k in keys:
        flip, p, c = cols[k]
        c = c.astype(float)
        w = 1.0 - c
        blocks.append(np.stack([
            flip * opp, opp, flip * same, same,
            flip * opp * c, opp * c, flip * same * c, same * c,
            flip * opp * w, opp * w, flip * same * w, same * w,
            p * water, water, p * land, land], 1))
    return keys, np.concatenate(blocks, 1)


def metrics_from(T, n_models):
    """T: (B, 16*M) 가중 합 -> (B, M, 4) 지표."""
    T = T.reshape(T.shape[0], n_models, 16)
    r = lambda a, b: T[..., a] / T[..., b]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.stack([r(0, 1) - r(2, 3), r(4, 5) - r(6, 7),
                         r(8, 9) - r(10, 11), r(12, 13) - r(14, 15)], -1)


def bootstrap(ref, X, n_models, n_boot, seed, chunk=100):
    rng = np.random.default_rng(seed)
    img = pd.factorize(ref["img_id"])[0]
    bg = pd.factorize(ref["bg_file"])[0]
    n_img = img.max() + 1
    bg_type = pd.Series(ref["bg_place"].to_numpy()).groupby(bg).first().to_numpy()
    pools = [np.flatnonzero(bg_type == t) for t in (0, 1)]
    out = []
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        wi = rng.multinomial(n_img, np.full(n_img, 1 / n_img), size=b).astype(float)
        wb = np.zeros((b, len(bg_type)))
        for pool in pools:   # 배경은 타입별로 따로 재표본
            wb[:, pool] = rng.multinomial(len(pool), np.full(len(pool), 1 / len(pool)), size=b)
        W = wi[:, img] * wb[:, bg]
        out.append(metrics_from(W @ X, n_models))
    return np.concatenate(out)


def fmt_ci(v, lo, hi):
    return f"{v:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", default="k5")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    runs = find_runs(args.tag)
    if not runs:
        raise SystemExit("runs/*/shortcut_<tag>_rows.csv.gz 가 없다. 저장소 루트에서 실행하고 remeasure 를 먼저 돌릴 것.")
    ref, cols, avg = load(runs, args.tag)
    keys, X = design(ref, cols)
    M = len(keys)
    idx = {k: i for i, k in enumerate(keys)}
    point = metrics_from(X.sum(0, keepdims=True), M)[0]          # (M, 4)
    boot = bootstrap(ref, X, M, args.n_boot, args.seed)         # (B, M, 4)

    # 기존 JSON 과 대조 (S 는 같은 정의라 일치해야 한다)
    for k, d in runs.items():
        with open(os.path.join(d, f"shortcut_{args.tag}.json")) as f:
            js = json.load(f)
        assert abs(point[idx[k], 0] - js["overall"]["S"]) < 1e-9, f"{k} S 불일치"
        assert abs(point[idx[k], 3] - js["dp"]["overall"]) < 1e-5, f"{k} 평균 Δp 불일치"

    models = sorted({k[0] for k in keys}, key=lambda n: -1 if n == "erm" else int(n.split("sp")[1]))
    seeds = sorted({k[1] for k in keys})
    dense = [idx[("erm", s)] for s in seeds if ("erm", s) in idx]
    thr = 2 * point[dense].std(0, ddof=1)                       # (4,)

    def included(name, s):
        return (name, s) in idx and ("erm", s) in idx and avg[(name, s)] - avg[("erm", s)] >= -MAX_DROP

    def compare(base):
        """base 대비 짝지은 차이. 반환: {(name, seed|'all'): (Δ(4), lo(4), hi(4), 판정(4))}"""
        res = {}
        for name in models:
            if name in ("erm", base):
                continue
            per = {}
            for s in seeds:
                if (name, s) in idx and (base, s) in idx:
                    per[s] = (point[idx[(name, s)]] - point[idx[(base, s)]],
                              boot[:, idx[(name, s)]] - boot[:, idx[(base, s)]])
            inc = [s for s in per if included(name, s)]
            if inc:
                per["all"] = (np.mean([per[s][0] for s in inc], 0), np.mean([per[s][1] for s in inc], 0))
            for s, (d, b) in per.items():
                lo, hi = np.nanpercentile(b, 2.5, 0), np.nanpercentile(b, 97.5, 0)
                changed = ((lo > 0) | (hi < 0)) & (np.abs(d) >= thr)
                res[(name, s)] = (d, lo, hi, changed, inc if s == "all" else None)
        return res

    L = [f"# H1 재판정 (tag={args.tag})\n",
         "CLAUDE.md §4 \"2026-11-10 결정 사항\" 1~3 의 방법. 생성: `analyze_h1.py`.\n",
         f"- 신뢰구간: 짝지은 two-way bootstrap {args.n_boot}회 (이미지 + 배경을 땅/물 타입별로 층화 재표본, "
         "모든 모델에 같은 가중치)",
         "- **판정 \"변함\"** = 95% CI 가 0 을 포함하지 않음 **AND** |Δ| ≥ 실질 크기 기준",
         f"- seed 합산은 정확도가 dense 대비 {MAX_DROP*100:.0f}pt 넘게 떨어진 seed 를 뺀 짝지은 차이 평균",
         "- 값 표기: Δ [95% CI]. ★ = 변함\n",
         "## 실질 크기 기준 (dense seed 표준편차 × 2)\n",
         "| 지표 | dense seed 값 | 표준편차 | 기준 |", "| --- | --- | --- | --- |"]
    for j, m in enumerate(METRICS):
        vals = ", ".join(f"{point[i, j]:.4f}" for i in dense)
        L.append(f"| {LABEL[m]} | {vals} | {thr[j]/2:.4f} | {thr[j]:.4f} |")

    L += ["\n## 모델별 값 (seed 평균 ± 표준편차)\n",
          "| 모델 | avg | " + " | ".join(LABEL[m] for m in METRICS) + " |",
          "| --- | --- | " + " | ".join("---" for _ in METRICS) + " |"]
    for name in models:
        ii = [idx[(name, s)] for s in seeds if (name, s) in idx]
        a = np.array([avg[(name, s)] for s in seeds if (name, s) in idx]) * 100
        cells = [f"{point[ii, j].mean():.4f} ± {point[ii, j].std(ddof=1):.4f}" if len(ii) > 1
                 else f"{point[ii, j].mean():.4f}" for j in range(4)]
        L.append(f"| {name} | {a.mean():.2f} | " + " | ".join(cells) + " |")

    for base, title in (("prune_sp00", "ft0 대비 (주 판정)"), ("erm", "dense 대비 (참고)")):
        res = compare(base)
        L += [f"\n## {title} — seed 합산\n",
              "| 모델 | 포함 seed | " + " | ".join(f"Δ{LABEL[m]}" for m in METRICS) + " | 판정 |",
              "| --- | --- | " + " | ".join("---" for _ in METRICS) + " | --- |"]
        for name in models:
            if (name, "all") not in res:
                if name not in ("erm", base) and any((name, s) in res for s in seeds):
                    L.append(f"| {name} | 없음 | " + " | ".join("-" for _ in METRICS) + " | 전 seed 정확도 제외 |")
                continue
            d, lo, hi, ch, inc = res[(name, "all")]
            cells = [fmt_ci(d[j], lo[j], hi[j]) + (" ★" if ch[j] else "") for j in range(4)]
            verdict = "변함: " + ", ".join(LABEL[m] for j, m in enumerate(METRICS) if ch[j]) if ch.any() else "변함 없음"
            L.append(f"| {name} | {','.join(map(str, inc))} | " + " | ".join(cells) + f" | {verdict} |")
        L += [f"\n### {title} — seed 별\n",
              "| 모델 | seed | 정확도 제외 | " + " | ".join(f"Δ{LABEL[m]}" for m in METRICS) + " |",
              "| --- | --- | --- | " + " | ".join("---" for _ in METRICS) + " |"]
        for name in models:
            for s in seeds:
                if (name, s) not in res:
                    continue
                d, lo, hi, ch, _ = res[(name, s)]
                cells = [fmt_ci(d[j], lo[j], hi[j]) + (" ★" if ch[j] else "") for j in range(4)]
                L.append(f"| {name} | {s} | {'예' if not included(name, s) else ''} | " + " | ".join(cells) + " |")

    text = "\n".join(L) + "\n"
    print(text)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
