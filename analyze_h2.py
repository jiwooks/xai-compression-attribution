"""
STEP 5. 히트맵 채점 — H2a, H2b (CLAUDE.md §4 "2026-11-10 결정 사항" 4, 추가 결정 11).

입력: runs/<run>_seed<k>/shortcut_k5_dp.csv.gz (이미지별 Δp), runs/<run>_seed<k>/attr_<method>.csv.gz
      results/h1_k5.md (H1 판정, H2b 해석 제외용)
모델: dense(erm), ft0(prune_sp00), prune_sp30/50/70 × seed 0,1,2.  방법: gradcam, ig (5,794장), rise (1,000장).
Attr_bg 열: gradcam·ig 는 attr_bg, rise 는 attr_bg_centered (추가 결정 11). C = (1 − Attr_bg) / area.

H2a — 모델 안에서 히트맵이 행동을 추적하는가
  주 지표: 부분 Spearman ρ(Attr_bg, Δp | area)
           = rank(Attr_bg), rank(Δp) 를 각각 rank(area) 에 선형회귀한 잔차끼리의 Pearson
           (= 순위 Pearson 들로 쓴 1차 부분상관 공식과 같다)
  보조: 원시 Spearman, AUROC(정답 = Δp ≥ 0.10, 점수 = Attr_bg), C 로 계산한 같은 지표
  CI: 이미지 bootstrap 2,000회. 모든 모델이 같은 재표본 인덱스를 써서 차이가 짝지어진다.
  판정 (결정 4): 70% 의 주 지표가 같은 seed ft0 의 95% CI 밖이고, seed 3개 중 2개 이상이 같은 방향이면 "지지".
                 방향이 아래(추적 약화)면 H2a 지지, 위(추적 강화)면 반대 방향으로 표시한다.
  참고: ft0 대비 짝지은 차이의 CI (seed 별, seed 평균) 도 함께 보고한다.

H2b — 행동이 같은 모델끼리 히트맵이 같은가
  변화량: 이미지별 |Attr_bg(sp, seed s) − Attr_bg(ft0, seed s)| 의 seed 평균
  잡음 기준: 이미지별 |Attr_bg(ft0, a) − Attr_bg(ft0, b)| 의 세 쌍 평균
  비교: 중앙값(변화량) − 중앙값(잡음), 이미지 bootstrap CI. CI 가 0 보다 크면 "행동은 같은데 설명이 바뀌었다" 후보.
  함께: 평균 Attr_bg 의 방향 변화 mean(Attr_bg(sp) − Attr_bg(ft0)) 과 CI. H1 이 "변함" 이면 해석에서 뺀다.

numpy/pandas/scipy/matplotlib 만 쓴다. 마스터 노드에서 실행하지 말 것 (srun, CPU 1개).
  python analyze_h2.py --out results/attribution_k5.md
"""

import argparse
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import rankdata

MODELS = ["erm", "prune_sp00", "prune_sp30", "prune_sp50", "prune_sp70"]
LAB = {"erm": "dense", "prune_sp00": "ft0", "prune_sp30": "30%", "prune_sp50": "50%", "prune_sp70": "70%"}
SPARSE = ["prune_sp30", "prune_sp50", "prune_sp70"]
SEEDS = [0, 1, 2]
METHODS = ["gradcam", "ig", "rise"]
MLAB = {"gradcam": "Grad-CAM", "ig": "IG", "rise": "RISE (centered)"}   # 그림은 영어 (클러스터에 한글 폰트 없음)
GROUPS = ["landbird/land", "landbird/water", "waterbird/land", "waterbird/water"]
DP_THR = 0.10

# 차트 색 (dataviz 기본 팔레트, validate_palette.js 통과: blue/orange/aqua)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


# ---------------------------------------------------------------- 데이터

def load(method):
    """반환: DataFrame (img_id 정렬) 과 {(model, seed): (attr, C, dp)}"""
    col = "attr_bg_centered" if method == "rise" else "attr_bg"
    base, per = None, {}
    for m in MODELS:
        for s in SEEDS:
            d = f"runs/{m}_seed{s}"
            a = pd.read_csv(f"{d}/attr_{method}.csv.gz")
            p = pd.read_csv(f"{d}/shortcut_k5_dp.csv.gz")[["img_id", "dp"]]
            t = a.merge(p, on="img_id", how="inner").sort_values("img_id").reset_index(drop=True)
            assert len(t) == len(a), f"{d}: Δp 가 없는 이미지가 있다"
            assert t[col].notna().all(), f"{d}/{method}: Attr_bg 에 NaN"
            if base is None:
                base = t[["img_id", "group", "y", "area"]].copy()
            else:
                assert (t["img_id"].to_numpy() == base["img_id"].to_numpy()).all(), "이미지 집합이 모델마다 다르다"
                assert np.allclose(t["area"], base["area"]), "area 가 모델마다 다르다"
            attr = t[col].to_numpy(float)
            per[(m, s)] = (attr, (1 - attr) / t["area"].to_numpy(float), t["dp"].to_numpy(float))
    return base, per


# ---------------------------------------------------------------- 통계

def _rowcorr(x, y):
    x = x - x.mean(1, keepdims=True)
    y = y - y.mean(1, keepdims=True)
    return (x * y).sum(1) / np.sqrt((x * x).sum(1) * (y * y).sum(1))


def track_stats(score, dp, rz, idx):
    """idx: (B, N) 재표본 인덱스 (0 행 = 원표본). rz: rank(area) 를 같은 idx 로 다시 순위화한 것.
    반환: dict(partial, spearman, auroc) 각 (B,)"""
    ra = rankdata(score[idx], axis=1)
    rd = rankdata(dp[idx], axis=1)
    r_ad, r_az, r_dz = _rowcorr(ra, rd), _rowcorr(ra, rz), _rowcorr(rd, rz)
    partial = (r_ad - r_az * r_dz) / np.sqrt((1 - r_az ** 2) * (1 - r_dz ** 2))
    pos = dp[idx] >= DP_THR
    n1 = pos.sum(1)
    n0 = pos.shape[1] - n1
    auroc = ((ra * pos).sum(1) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return {"partial": partial, "spearman": r_ad, "auroc": auroc}


def boot_index(n, n_boot, seed):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot + 1, n))
    idx[0] = np.arange(n)
    return idx


def ci(v):
    return np.nanpercentile(v[1:], 2.5), np.nanpercentile(v[1:], 97.5)


def compute_h2a(base, per, sel, idx):
    """sel: 이미지 부분집합 (bool). 반환: {(model, seed, var): stats dict}, var ∈ {attr, C}"""
    sub = np.flatnonzero(sel)
    assert idx.shape[1] == len(sub), "재표본 인덱스 크기가 부분집합과 다르다"
    area = base["area"].to_numpy(float)[sub]
    rz = rankdata(area[idx], axis=1)
    out = {}
    for (m, s), (attr, C, dp) in per.items():
        out[(m, s, "attr")] = track_stats(attr[sub], dp[sub], rz, idx)
        out[(m, s, "C")] = track_stats(C[sub], dp[sub], rz, idx)
    return out


def verdict_h2a(st, model, var="attr", key="partial"):
    """결정 4 의 규칙. 반환: (판정 문자열, 바깥으로 나간 seed 별 방향 목록)"""
    dirs = []
    for s in SEEDS:
        lo, hi = ci(st[("prune_sp00", s, var)][key])
        v = st[(model, s, var)][key][0]
        dirs.append("↓" if v < lo else ("↑" if v > hi else "·"))
    down, up = dirs.count("↓"), dirs.count("↑")
    if down >= 2:
        return "지지 (추적 약화)", dirs
    if up >= 2:
        return "반대 (추적 강화)", dirs
    return "지지 안 됨", dirs


def paired_diff(st, model, var="attr", key="partial"):
    """ft0 대비 짝지은 차이. 반환: seed 별 (Δ, lo, hi) 와 seed 평균 (Δ, lo, hi)"""
    per = {}
    for s in SEEDS:
        d = st[(model, s, var)][key] - st[("prune_sp00", s, var)][key]
        per[s] = (d[0], *ci(d))
    pooled = np.mean([st[(model, s, var)][key] - st[("prune_sp00", s, var)][key] for s in SEEDS], 0)
    return per, (pooled[0], *ci(pooled))


def compute_h2b(per, idx, sp):
    """반환: dict — 변화량·잡음 중앙값, 중앙값 차이 (점, CI), 평균 방향 변화 (점, CI), seed 별 중앙값"""
    change = np.mean([np.abs(per[(sp, s)][0] - per[("prune_sp00", s)][0]) for s in SEEDS], 0)
    pairs = [(0, 1), (0, 2), (1, 2)]
    noise = np.mean([np.abs(per[("prune_sp00", a)][0] - per[("prune_sp00", b)][0]) for a, b in pairs], 0)
    signed = np.mean([per[(sp, s)][0] - per[("prune_sp00", s)][0] for s in SEEDS], 0)
    dmed = np.median(change[idx], 1) - np.median(noise[idx], 1)
    dmean = signed[idx].mean(1)
    return {"change_med": float(np.median(change)), "noise_med": float(np.median(noise)),
            "dmed": (dmed[0], *ci(dmed)), "dmean": (dmean[0], *ci(dmean)),
            "change_med_seed": [float(np.median(np.abs(per[(sp, s)][0] - per[("prune_sp00", s)][0])))
                                for s in SEEDS],
            "noise_med_pair": [float(np.median(np.abs(per[("prune_sp00", a)][0] - per[("prune_sp00", b)][0])))
                               for a, b in pairs],
            "change": change, "noise": noise}


def read_h1(path):
    """results/h1_k5.md 의 ft0 대비 seed 합산 판정. 반환: {model: 판정}"""
    out = {}
    if not os.path.exists(path):
        return out
    text = open(path, encoding="utf-8").read()
    sec = text.split("## ft0 대비 (주 판정) — seed 합산")[1].split("###")[0]
    for line in sec.splitlines():
        m = re.match(r"\| (prune_sp\d+) \|.*\| ([^|]+) \|\s*$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


# ---------------------------------------------------------------- 그림

def _style(ax):
    ax.set_facecolor(SURF)
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color(MUTED)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def fig_h2a(results, out_dir):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=False)
    fig.patch.set_facecolor(SURF)
    for ax, method in zip(axes, METHODS):
        st = results[method]["all"]
        _style(ax)
        for i, m in enumerate(MODELS):
            vals = [st[(m, s, "attr")]["partial"] for s in SEEDS]
            for j, v in enumerate(vals):                       # seed 별 점 + 95% CI
                lo, hi = ci(v)
                x = i + (j - 1) * 0.12
                ax.plot([x, x], [lo, hi], color=MUTED, linewidth=1.2)
                ax.plot(x, v[0], "o", ms=4, color=MUTED, mec=SURF, mew=1)
            ax.plot(i, np.mean([v[0] for v in vals]), "_", ms=16, mew=2.5, color=SERIES[0])
        ax.set_xticks(range(len(MODELS)), [LAB[m] for m in MODELS])
        ax.set_title(MLAB[method], fontsize=10, color=INK, loc="left")
        if method == "gradcam":
            ax.set_ylabel("partial Spearman ρ(Attr_bg, Δp | area)", fontsize=8, color=INK2)
    fig.suptitle("H2a — how well Attr_bg tracks behaviour (Δp). Grey = per-seed estimate and 95% CI, blue bar = seed mean",
                 fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(out_dir, "h2a_partial_by_sparsity.png")
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    return path


def fig_scatter(data, results, out_dir):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.4), sharey=True)
    fig.patch.set_facecolor(SURF)
    for c, method in enumerate(METHODS):
        base, per = data[method]
        for r, m in enumerate(["prune_sp00", "prune_sp70"]):
            ax = axes[r, c]
            _style(ax)
            ax.grid(axis="x", color=GRID, linewidth=0.6)
            attr, _, dp = per[(m, 0)]
            ax.scatter(attr, dp, s=3, alpha=0.18, color=SERIES[0], linewidths=0)
            ax.axhline(DP_THR, color=MUTED, linewidth=0.8, linestyle="--")
            rho = results[method]["all"][(m, 0, "attr")]["partial"][0]
            ax.set_title(f"{MLAB[method]} · {LAB[m]} (seed 0)   partial ρ = {rho:+.3f}", fontsize=9,
                         color=INK, loc="left")
            if r == 1:
                ax.set_xlabel("Attr_bg", fontsize=8, color=INK2)
            if c == 0:
                ax.set_ylabel("Δp", fontsize=8, color=INK2)
    fig.suptitle(f"Attr_bg vs Δp (dashed = AUROC label threshold Δp {DP_THR})", fontsize=9, color=INK2, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(out_dir, "h2a_scatter_seed0.png")
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    return path


def fig_h2b(h2b, out_dir):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    fig.patch.set_facecolor(SURF)
    for ax, method in zip(axes, METHODS):
        _style(ax)
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        noise = np.sort(h2b[method]["prune_sp70"]["noise"])
        y = np.arange(1, len(noise) + 1) / len(noise)
        ax.plot(noise, y, color=MUTED, linewidth=2, linestyle="--")
        ax.annotate("noise", (np.percentile(noise, 92), 0.92), xytext=(6, -8),
                    textcoords="offset points", fontsize=7, color=INK2)
        for k, sp in enumerate(SPARSE):
            ch = np.sort(h2b[method][sp]["change"])
            ax.plot(ch, y[:len(ch)] if len(ch) == len(y) else np.arange(1, len(ch) + 1) / len(ch),
                    color=SERIES[k], linewidth=2)
            yq = (0.3, 0.55, 0.8)[k]                              # 라벨이 겹치지 않게 높이를 나눈다
            ax.annotate(LAB[sp], (np.percentile(ch, yq * 100), yq), xytext=(5, -2), textcoords="offset points",
                        fontsize=7, color=INK2)
        hi = max(np.percentile(h2b[method][sp]["change"], 98) for sp in SPARSE)
        ax.set_xlim(0, max(hi, np.percentile(noise, 98)))
        ax.set_title(MLAB[method], fontsize=10, color=INK, loc="left")
        ax.set_xlabel("per-image |Attr_bg difference|", fontsize=8, color=INK2)
        if method == "gradcam":
            ax.set_ylabel("cumulative fraction", fontsize=8, color=INK2)
    handles = [plt.Line2D([], [], color=SERIES[k], linewidth=2, label=f"{LAB[sp]} - ft0 (same seed)")
               for k, sp in enumerate(SPARSE)]
    handles.append(plt.Line2D([], [], color=MUTED, linewidth=2, linestyle="--", label="ft0 between seeds (noise)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.01, 0.93), fontsize=7, frameon=False, ncol=4)
    fig.suptitle("H2b — per-image heatmap change from pruning vs between-seed noise (ECDF; further right = larger)",
                 fontsize=9, color=INK2, x=0.01, ha="left", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path = os.path.join(out_dir, "h2b_change_vs_noise.png")
    fig.savefig(path, dpi=150, facecolor=SURF)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 출력

def f3(v):
    return f"{v:+.3f}"


def fci(t):
    return f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/attribution_k5.md")
    ap.add_argument("--fig_dir", default="results/figs")
    ap.add_argument("--h1", default="results/h1_k5.md")
    ap.add_argument("--no_fig", action="store_true")
    args = ap.parse_args()

    h1 = read_h1(args.h1)
    data, results, h2b = {}, {}, {}
    for method in METHODS:
        base, per = load(method)
        data[method] = (base, per)
        n = len(base)
        idx = boot_index(n, args.n_boot, args.seed)
        res = {"all": compute_h2a(base, per, np.ones(n, bool), idx)}
        g = base["group"].to_numpy()
        for k in range(4):
            sel = g == k
            res[GROUPS[k]] = compute_h2a(base, per, sel, boot_index(int(sel.sum()), args.n_boot, args.seed + 1 + k))
        results[method] = res
        h2b[method] = {sp: compute_h2b(per, idx, sp) for sp in SPARSE}
        print(f"{method}: {n}장 완료", flush=True)

    L = ["# STEP 5 — 히트맵 채점 (H2a, H2b)\n",
         "생성: `analyze_h2.py`. 방법·판정 규칙은 CLAUDE.md §4 \"2026-11-10 결정 사항\" 4, 추가 결정 11.\n",
         f"- 이미지: Grad-CAM·IG 5,794장, RISE 1,000장 (`results/rise_images.csv`). RISE 의 Attr_bg 는 centered 맵.",
         f"- 행동 지표 Δp 는 이미지별 (물 배경 반사실 평균 p(waterbird) − 땅 배경 평균). AUROC 정답 = Δp ≥ {DP_THR}.",
         f"- CI: 이미지 bootstrap {args.n_boot}회 (모든 모델이 같은 재표본 인덱스 → ft0 대비 차이가 짝지어진다).",
         "- 값은 Attr_bg 기준. **양수 ρ = 배경 attribution 이 큰 이미지일수록 배경 의존(Δp)도 크다 (추적).**\n"]

    # ---- 판정 요약
    L += ["## 판정 요약\n",
          "| 방법 | H2a 70% (seed 별 방향) | ft0 ρ | 70% ρ | Δρ 70%−ft0 [CI] | H2b 70% 중앙값 차이 [CI] | 평균 Attr_bg 변화 [CI] | H1 70% |",
          "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for method in METHODS:
        st = results[method]["all"]
        v, dirs = verdict_h2a(st, "prune_sp70")
        _, pooled = paired_diff(st, "prune_sp70")
        b = h2b[method]["prune_sp70"]
        r0 = np.mean([st[("prune_sp00", s, "attr")]["partial"][0] for s in SEEDS])
        r7 = np.mean([st[("prune_sp70", s, "attr")]["partial"][0] for s in SEEDS])
        L.append(f"| {MLAB[method]} | {v} ({' '.join(dirs)}) | {r0:+.3f} | {r7:+.3f} | {fci(pooled)} | "
                 f"{fci(b['dmed'])} | {fci(b['dmean'])} | {h1.get('prune_sp70', '-')} |")
    L += ["\n- H2a 방향 기호: ↓ = 70% 가 같은 seed ft0 의 95% CI 아래, ↑ = 위, · = 안. 2개 이상 ↓ 면 지지.",
          "- H2b 중앙값 차이 > 0 이고 CI 가 0 을 넘으면 압축에 따른 히트맵 변화가 seed 간 잡음보다 크다.",
          "- 평균 Attr_bg 변화 < 0 = 히트맵이 새 쪽으로 이동.\n"]

    # ---- H2a 상세
    L.append("## H2a — 모델별 추적 지표\n")
    for method in METHODS:
        st = results[method]["all"]
        L += [f"### {MLAB[method]}\n",
              "| 모델 | 부분 ρ seed 0 [CI] | seed 1 [CI] | seed 2 [CI] | 평균 | 원시 Spearman | AUROC | 부분 ρ (C) | ft0 대비 Δ (seed 평균) [CI] | 판정 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m in MODELS:
            cells = [fci((st[(m, s, "attr")]["partial"][0], *ci(st[(m, s, "attr")]["partial"]))) for s in SEEDS]
            mean = lambda var, key: np.mean([st[(m, s, var)][key][0] for s in SEEDS])
            if m in SPARSE:
                _, pooled = paired_diff(st, m)
                v, dirs = verdict_h2a(st, m)
                extra = [fci(pooled), f"{v} ({' '.join(dirs)})"]
            else:
                extra = ["-", "-"]
            L.append(f"| {LAB[m]} | " + " | ".join(cells) + f" | {mean('attr', 'partial'):+.3f} | "
                     f"{mean('attr', 'spearman'):+.3f} | {mean('attr', 'auroc'):.3f} | {mean('C', 'partial'):+.3f} | "
                     + " | ".join(extra) + " |")
        L.append("")
        L += ["보조 지표의 ft0 대비 판정 (같은 규칙)\n",
              "| 모델 | 원시 Spearman | AUROC | 부분 ρ (C) |", "| --- | --- | --- | --- |"]
        for m in SPARSE:
            cells = []
            for var, key in (("attr", "spearman"), ("attr", "auroc"), ("C", "partial")):
                v, dirs = verdict_h2a(st, m, var, key)
                cells.append(f"{v} ({' '.join(dirs)})")
            L.append(f"| {LAB[m]} | " + " | ".join(cells) + " |")
        L.append("")
        L.append("C 는 높을수록 새에 집중이므로 추적이면 ρ 가 음수다. 판정 문구의 '약화/강화' 는 ρ 값의 증감 기준이다.\n")

    # ---- 그룹별
    L.append("## H2a — 그룹별 (부분 ρ, seed 평균)\n")
    for method in METHODS:
        L += [f"### {MLAB[method]}\n",
              "| 그룹 | n | dense | ft0 | 30% | 50% | 70% | 70% 판정 (seed 별) | Δ 70%−ft0 [CI] |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        base = data[method][0]
        for k, gname in enumerate(GROUPS):
            st = results[method][gname]
            means = [np.mean([st[(m, s, "attr")]["partial"][0] for s in SEEDS]) for m in MODELS]
            v, dirs = verdict_h2a(st, "prune_sp70")
            _, pooled = paired_diff(st, "prune_sp70")
            L.append(f"| {gname} | {int((base['group'] == k).sum())} | " + " | ".join(f"{x:+.3f}" for x in means)
                     + f" | {v} ({' '.join(dirs)}) | {fci(pooled)} |")
        L.append("")

    # ---- H2b
    L.append("## H2b — 행동이 같은 모델끼리 히트맵이 같은가\n")
    L.append("변화량 = 이미지별 |Attr_bg(sp) − Attr_bg(ft0)| (같은 seed) 의 seed 평균. "
             "잡음 = 이미지별 |Attr_bg(ft0, a) − Attr_bg(ft0, b)| 의 세 쌍 평균.\n")
    for method in METHODS:
        L += [f"### {MLAB[method]}\n",
              "| sparsity | H1 (ft0 대비) | 변화량 중앙값 | 잡음 중앙값 | 차이 [CI] | 평균 Attr_bg 변화 [CI] | seed 별 변화량 중앙값 | 쌍별 잡음 중앙값 | 해석 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for sp in SPARSE:
            b = h2b[method][sp]
            h = h1.get(sp, "-")
            bigger = b["dmed"][1] > 0
            if h.startswith("변함:"):
                interp = "H1 변함 → 해석 제외"
            elif bigger:
                direction = "새 쪽" if b["dmean"][2] < 0 else ("배경 쪽" if b["dmean"][1] > 0 else "방향 불분명")
                interp = f"행동은 같은데 설명이 바뀌었다 ({direction})"
            else:
                interp = "잡음 수준"
            L.append(f"| {LAB[sp]} | {h} | {b['change_med']:.4f} | {b['noise_med']:.4f} | {fci(b['dmed'])} | "
                     f"{fci(b['dmean'])} | {', '.join(f'{x:.4f}' for x in b['change_med_seed'])} | "
                     f"{', '.join(f'{x:.4f}' for x in b['noise_med_pair'])} | {interp} |")
        L.append("")
    L.append("H1 열은 `results/h1_k5.md` 의 ft0 대비 seed 합산 판정이다 (S, d_correct, d_wrong, 평균 Δp).\n")

    # ---- 모델 단위 참고
    L += ["## 참고 — 모델 단위 (15개 모델)\n",
          "| 방법 | Spearman(평균 Attr_bg, 평균 Δp) |", "| --- | --- |"]
    from scipy.stats import spearmanr
    for method in METHODS:
        per = data[method][1]
        a = [per[k][0].mean() for k in per]
        d = [per[k][2].mean() for k in per]
        L.append(f"| {MLAB[method]} | {spearmanr(a, d)[0]:+.3f} |")
    L.append("\n모델 단위 상관은 CLAUDE.md §2 에 따라 참고용이다.\n")

    if not args.no_fig:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import font_manager
        for name in ("NanumGothic", "Noto Sans CJK KR", "Malgun Gothic", "UnDotum"):
            if any(name in f.name for f in font_manager.fontManager.ttflist):
                plt.rcParams["font.family"] = name
                break
        plt.rcParams["axes.unicode_minus"] = False
        os.makedirs(args.fig_dir, exist_ok=True)
        figs = [fig_h2a(results, args.fig_dir), fig_scatter(data, results, args.fig_dir), fig_h2b(h2b, args.fig_dir)]
        rel = [os.path.relpath(p, os.path.dirname(os.path.abspath(args.out))).replace("\\", "/") for p in figs]
        L += ["## 그림\n",
              f"- H2a sparsity 별 주 지표: ![]({rel[0]})",
              f"- ft0·70% 산점도 (seed 0): ![]({rel[1]})",
              f"- H2b 변화량 vs 잡음: ![]({rel[2]})\n"]

    text = "\n".join(L) + "\n"
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
