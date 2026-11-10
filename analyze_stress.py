"""
P4 stress test (CLAUDE.md §4 "2026-11-10 결정 사항" 10). 출력: results/stress_k5.md (+ 그림 1장).

기본 결론 (STEP 3·5, results/h1_k5.md, results/attribution_k5.md)
  H1  : ≤70% 에서 ft0 대비 행동 지표(S, d_correct, d_wrong, 평균 Δp) 변함 없음
  H2a : 70% 에서 Grad-CAM·IG 지지 안 됨, RISE(centered) 반대(추적 강화)
  H2b : 70% 에서 세 방법 모두 히트맵 변화 > seed 간 잡음, 새 쪽으로 이동

ST-A  fine-tuning 길이. prune5ep_sp{00,50,70} (5 epoch) 을 one-shot(재학습 전)·20 epoch(prune_sp*) 와 한 표로.
      one-shot 은 prune5ep 실행 때 저장한 oneshot.pt 를 measure_shortcut 으로 잰 shortcut_k5_oneshot_* 를 쓴다.
      sparsity 0 의 one-shot = dense(erm). 행동 지표 CI 는 analyze_h1 과 같은 짝지은 two-way bootstrap.
      5 epoch 모델 중 정확도 기준(dense 대비 2pt)을 통과한 것의 Grad-CAM·IG 로 H2a·H2b 를 다시 낸다
      (기준 = prune5ep_sp00).
ST-B  측정 방식. 같은 STEP 4 결과의 attr_bg_band (경계 ±3px 띠 제외), attr_bg_7x7 (Grad-CAM 7×7) 로 H2a·H2b.
      RISE 의 band 는 raw 맵에서만 저장돼 있어 raw 와 짝지어 참고로 싣는다.
ST-C  IG baseline. ft0·70% 의 IG 를 blur baseline (가우시안 σ=10px) 으로 다시 계산한 attr_ig_blur.* 로 H2a·H2b.

규칙·지표는 analyze_h1.py, analyze_h2.py 의 함수를 그대로 쓴다. 이미지 bootstrap 은 STEP 5 와 같은 시드라
같은 이미지 수면 같은 재표본 인덱스다.
마스터 노드에서 실행하지 말 것 (srun, CPU 1개).
  python analyze_stress.py --out results/stress_k5.md
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

import analyze_h1 as H1
import analyze_h2 as H2

SEEDS = [0, 1, 2]
SPS = ["00", "50", "70"]
REGIMES = ["one-shot", "5ep", "20ep"]
MAX_DROP = 0.02
M_LAB = ["S", "d_correct", "d_wrong", "평균 Δp"]
BASE_H2A = {"gradcam": "지지 안 됨", "ig": "지지 안 됨", "rise": "반대 (추적 강화)"}


def fci(t, nd=3):
    return f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]"


# ---------------------------------------------------------------- H2 공통

def h2_eval(method, col, models, ref, targets, n_boot, seed, suffix="", seeds=SEEDS):
    """반환: {target: dict(verdict, dirs, r_ref, r_t, dpartial, h2b)} 와 모델별 평균 Attr_bg"""
    base, per = H2.load(method, col=col, models=models, seeds=seeds, suffix=suffix)
    n = len(base)
    idx = H2.boot_index(n, n_boot, seed)
    st = H2.compute_h2a(base, per, np.ones(n, bool), idx)
    out = {}
    for t in targets:
        v, dirs = H2.verdict_h2a(st, t, ref=ref, seeds=seeds)
        _, pooled = H2.paired_diff(st, t, ref=ref, seeds=seeds)
        out[t] = {"verdict": v, "dirs": dirs, "dpartial": pooled,
                  "r_ref": float(np.mean([st[(ref, s, "attr")]["partial"][0] for s in seeds])),
                  "r_t": float(np.mean([st[(t, s, "attr")]["partial"][0] for s in seeds])),
                  "h2b": H2.compute_h2b(per, idx, t, ref=ref, seeds=seeds)}
    mean_attr = {m: float(np.mean([per[(m, s)][0].mean() for s in seeds])) for m in models}
    return out, mean_attr


def h2b_label(b):
    if b["dmed"][1] <= 0:
        return "잡음 수준"
    d = "새 쪽" if b["dmean"][2] < 0 else ("배경 쪽" if b["dmean"][1] > 0 else "방향 불분명")
    return f"변화 > 잡음 ({d})"


def h2_rows(name, r):
    b = r["h2b"]
    return (f"| {name} | {r['r_ref']:+.3f} | {r['r_t']:+.3f} | {fci(r['dpartial'])} | "
            f"{r['verdict']} ({' '.join(r['dirs'])}) | {b['change_med']:.4f} / {b['noise_med']:.4f} | "
            f"{fci(b['dmed'], 4)} | {fci(b['dmean'], 4)} | {h2b_label(b)} |")


H2_HEAD = ["| 설정 | 기준 ρ | 70% ρ | Δρ [CI] | H2a 판정 (seed 별) | H2b 변화량 / 잡음 중앙값 | 중앙값 차이 [CI] | "
           "평균 Attr_bg 변화 [CI] | H2b |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]


def weakened(r):
    return r["verdict"].startswith("지지 (")


def same_conclusion(method, r, base_r):
    """기본 결론과 비교. 핵심 = H2a '추적 약화 여부' 와 H2b 결론이 같은가.
    세부 = H2a 판정 문자열까지 같은가 ('지지 안 됨' 과 '반대 (추적 강화)' 는 둘 다 약화 없음).
    반환: (핵심 같음, H2a 핵심 같음, H2b 같음, H2a 세부 같음)"""
    a = weakened(r) == weakened(base_r)
    b = h2b_label(r["h2b"]) == h2b_label(base_r["h2b"])
    return a and b, a, b, r["verdict"] == base_r["verdict"]


def diff_text(name, res, r, base_r):
    ok, a, b, ax = res
    t = f"{name}: H2a 약화 {'있음' if weakened(r) else '없음'} ({'같음' if a else '다름'})"
    if not ax:
        t += f", 세부 판정 '{base_r['verdict']}' → '{r['verdict']}'"
    return t + f"; H2b {h2b_label(r['h2b'])} ({'같음' if b else '다름'})"


# ---------------------------------------------------------------- ST-A 행동 지표

def st_a_spec():
    """반환: {(regime, sp, seed): (dir, tag, acc_json)} — 파일이 있는 것만"""
    spec = {}
    for s in SEEDS:
        spec[("dense", "00", s)] = (f"runs/erm_seed{s}", "k5", f"runs/erm_seed{s}/final_test.json")
        for sp in SPS:
            d5 = f"runs/prune5ep_sp{sp}_seed{s}"
            spec[("5ep", sp, s)] = (d5, "k5", f"{d5}/final_test.json")
            spec[("20ep", sp, s)] = (f"runs/prune_sp{sp}_seed{s}", "k5", f"runs/prune_sp{sp}_seed{s}/final_test.json")
            if sp != "00":
                spec[("one-shot", sp, s)] = (d5, "k5_oneshot", f"{d5}/oneshot_test.json")
    return {k: v for k, v in spec.items()
            if os.path.exists(f"{v[0]}/shortcut_{v[1]}_rows.csv.gz") and os.path.exists(v[2])}


def st_a_load(spec):
    ref, cols, acc = None, {}, {}
    for key, (d, tag, aj) in spec.items():
        rows = pd.read_csv(f"{d}/shortcut_{tag}_rows.csv.gz").sort_values(H1.KEY).reset_index(drop=True)
        dp = pd.read_csv(f"{d}/shortcut_{tag}_dp.csv.gz")
        if ref is None:
            ref = rows[H1.KEY + ["bg_file", "bg_place"]].copy()
        elif not rows[H1.KEY + ["bg_file", "bg_place"]].equals(ref):
            raise SystemExit(f"[실패] {d} ({tag}) 의 반사실 행이 다른 모델과 다르다")
        corr = rows["img_id"].map(dp.set_index("img_id")["correct_orig"]).to_numpy().astype(bool)
        cols[key] = (rows["flip"].to_numpy(float), rows["p_w"].to_numpy(float), corr)
        with open(aj) as f:
            js = json.load(f)
        acc[key] = (js["avg_acc"], js["worst_group_acc"])
    return ref, cols, acc


def st_a_behaviour(n_boot, seed):
    spec = st_a_spec()
    if not any(k[0] == "5ep" for k in spec):
        return None
    ref, cols, acc = st_a_load(spec)
    keys, X = H1.design(ref, cols)
    M = len(keys)
    ix = {k: i for i, k in enumerate(keys)}
    point = H1.metrics_from(X.sum(0, keepdims=True), M)[0]
    boot = H1.bootstrap(ref, X, M, n_boot, seed)
    for k in keys:                                               # 저장된 JSON 과 대조
        d, tag, _ = spec[k]
        with open(f"{d}/shortcut_{tag}.json") as f:
            js = json.load(f)
        assert abs(point[ix[k], 0] - js["overall"]["S"]) < 1e-9, f"{k} S 불일치"
    dense = [ix[("dense", "00", s)] for s in SEEDS]
    thr = 2 * point[dense].std(0, ddof=1)

    def passed(k):
        return acc[k][0] - acc[("dense", "00", k[2])][0] >= -MAX_DROP

    def compare(a, b):
        """a − b (regime, sp) 짝지은 차이. 반환: (Δ, lo, hi, 변함, 포함 seed, 정확도 통과 seed)"""
        ss = [s for s in SEEDS if (*a, s) in ix and (*b, s) in ix]
        if not ss:
            return None
        inc = [s for s in ss if passed((*a, s)) and passed((*b, s))]
        use = inc or ss
        d = np.mean([point[ix[(*a, s)]] - point[ix[(*b, s)]] for s in use], 0)
        bb = np.mean([boot[:, ix[(*a, s)]] - boot[:, ix[(*b, s)]] for s in use], 0)
        lo, hi = np.nanpercentile(bb, 2.5, 0), np.nanpercentile(bb, 97.5, 0)
        ch = ((lo > 0) | (hi < 0)) & (np.abs(d) >= thr)
        return d, lo, hi, ch, use, inc

    table = {}
    for k in keys:
        table.setdefault(k[:2], []).append((acc[k][0], acc[k][1], *point[ix[k]], passed(k)))
    return {"table": table, "compare": compare, "thr": thr, "passed": passed, "acc": acc}


def fig_st_a(beh, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
    fig.patch.set_facecolor(H2.SURF)
    xs = {"00": 0, "50": 1, "70": 2}
    panels = [(2, "S = BFR_opp - BFR_same"), (5, "mean Δp"), (0, "test avg accuracy")]
    for ax, (j, title) in zip(axes, panels):
        H2._style(ax)
        for c, reg in enumerate(REGIMES):
            pts = []
            for sp in SPS:
                key = ("dense", "00") if (reg == "one-shot" and sp == "00") else (reg, sp)
                rows = beh["table"].get(key, [])
                if not rows:
                    continue
                vals = [r[j] for r in rows]
                x = xs[sp] + (c - 1) * 0.1
                ax.plot([x] * len(vals), vals, "o", ms=3.5, color=H2.SERIES[c], alpha=0.45, mec="none")
                pts.append((x, np.mean(vals)))
            if pts:
                ax.plot(*zip(*pts), "-", color=H2.SERIES[c], linewidth=2, label=reg)
        ax.set_xticks([0, 1, 2], ["0%", "50%", "70%"])
        ax.set_title(title, fontsize=10, color=H2.INK, loc="left")
    axes[0].legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("ST-A — behaviour after pruning by fine-tuning length (dots = seeds, line = seed mean; "
                 "0% one-shot = dense)", fontsize=9, color=H2.INK2, x=0.01, ha="left")
    fig.tight_layout()
    path = os.path.join(out_dir, "stress_a_finetune_length.png")
    fig.savefig(path, dpi=150, facecolor=H2.SURF)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/stress_k5.md")
    ap.add_argument("--fig_dir", default="results/figs")
    args = ap.parse_args()
    nb, sd = args.n_boot, args.seed
    std_models = ["prune_sp00", "prune_sp70"]

    # 기본 설정 (STEP 5 와 같은 값이 나와야 한다)
    base = {}
    for method in H2.METHODS:
        r, ma = h2_eval(method, None, std_models, "prune_sp00", ["prune_sp70"], nb, sd)
        base[method] = (r["prune_sp70"], ma)
        print(f"기본 {method}: {r['prune_sp70']['verdict']} / {h2b_label(r['prune_sp70']['h2b'])}", flush=True)

    L = ["# P4 — stress test (ST-A, ST-B, ST-C)\n",
         "생성: `analyze_stress.py`. CLAUDE.md §4 \"2026-11-10 결정 사항\" 10. "
         "판정 규칙은 STEP 3·5 와 같다 (`analyze_h1.py`, `analyze_h2.py`).\n",
         "**기본 결론**: H1 ≤70% ft0 대비 변함 없음 / H2a 70% Grad-CAM·IG 지지 안 됨, RISE 반대(추적 강화) / "
         "H2b 70% 세 방법 모두 변화 > 잡음, 새 쪽.\n",
         "ρ = 부분 Spearman ρ(Attr_bg, Δp | area) 의 seed 평균. H2a 방향 기호 ↓ = 기준 모델 CI 아래, ↑ = 위, · = 안.\n"]
    summary = []

    # ---------------- ST-B
    L.append("## ST-B — 측정 방식 (마스크 경계 띠, Grad-CAM 7×7)\n")
    # (방법, 열, 표시 이름, 비교 기준). 비교 기준 None = 기본 설정 자체, "raw" = RISE raw 맵 (판정에서 뺀다)
    variants = [("gradcam", "attr_bg", "Grad-CAM · 기본 (224 업샘플)", None),
                ("gradcam", "attr_bg_band", "Grad-CAM · 경계 ±3px 띠 제외", "base"),
                ("gradcam", "attr_bg_7x7", "Grad-CAM · 7×7 해상도", "base"),
                ("ig", "attr_bg", "IG · 기본", None),
                ("ig", "attr_bg_band", "IG · 경계 ±3px 띠 제외", "base"),
                ("rise", "attr_bg_centered", "RISE · 기본 (centered)", None),
                ("rise", "attr_bg", "RISE · raw 맵 (참고, 결정 11)", "raw"),
                ("rise", "attr_bg_band", "RISE · raw 맵 + 띠 제외", "raw_band")]
    L += H2_HEAD
    held = []
    raw_rise = None
    for method, col, name, cmp in variants:
        if cmp is None:
            r = base[method][0]
        else:
            r = h2_eval(method, col, std_models, "prune_sp00", ["prune_sp70"], nb, sd)[0]["prune_sp70"]
            if cmp == "raw":
                raw_rise = r
            else:
                ref_r = raw_rise if cmp == "raw_band" else base[method][0]
                held.append((name + (" (raw 맵과 비교)" if cmp == "raw_band" else ""),
                             same_conclusion(method, r, ref_r)))
        L.append(h2_rows(name, r))
    L.append("\n- RISE 의 띠 제외 값은 raw 맵에서만 저장돼 있어 raw 맵과 비교한다. "
             "raw 맵 자체는 sanity 를 통과하지 못해(결정 11) 판정에 넣지 않고 참고로만 싣는다.\n")
    for n_, res in held:
        ok, a, b, ax = res
        L.append(f"- {n_}: H2a 핵심 {'같음' if a else '다름'}" + ("" if ax else " (세부 판정 다름)")
                 + f", H2b {'같음' if b else '다름'}")
    vb = all(res[0] and res[3] for _, res in held)
    line = ("**ST-B 판정: 기본 결론 유지** — 경계 띠 제외·7×7 에서도 H2a 판정과 H2b 결론이 같다."
            if vb else "**ST-B 판정: 일부 달라짐** — 위 목록의 '다름' 항목 참조.")
    L += ["", line, ""]
    summary.append(("ST-B 측정 방식", line))

    # ---------------- ST-C
    L.append("## ST-C — IG baseline (검정 → blur σ=10px)\n")
    blur_ok = all(os.path.exists(f"runs/{m}_seed{s}/attr_ig_blur.csv.gz") for m in std_models for s in SEEDS)
    if blur_ok:
        rb, mab = h2_eval("ig", None, std_models, "prune_sp00", ["prune_sp70"], nb, sd, suffix="_blur")
        rb = rb["prune_sp70"]
        L += H2_HEAD
        L.append(h2_rows("IG · 검정 baseline (기본)", base["ig"][0]))
        L.append(h2_rows("IG · blur baseline", rb))
        mak = base["ig"][1]
        L += ["", "| 모델 | 평균 Attr_bg 검정 | 평균 Attr_bg blur |", "| --- | --- | --- |"]
        for m in std_models:
            L.append(f"| {H2.LAB[m]} | {mak[m]:.4f} | {mab[m]:.4f} |")
        L.append("")
        res = same_conclusion("ig", rb, base["ig"][0])
        line = (f"**ST-C 판정: 핵심 결론 {'유지' if res[0] else '달라짐'}** — "
                + diff_text("IG blur", res, rb, base["ig"][0])
                + f". 다만 ρ 수준이 baseline 에 따라 바뀐다 (ft0 {base['ig'][0]['r_ref']:+.3f} → {rb['r_ref']:+.3f}).")
        L += [line, ""]
        summary.append(("ST-C IG baseline", line))
    else:
        L.append("attr_ig_blur 결과가 아직 없다.\n")
        summary.append(("ST-C IG baseline", "미완료"))

    # ---------------- ST-A
    L.append("## ST-A — fine-tuning 길이 (one-shot / 5 epoch / 20 epoch)\n")
    beh = st_a_behaviour(nb, sd)
    if beh is None:
        L.append("prune5ep 결과가 아직 없다.\n")
        summary.append(("ST-A fine-tuning 길이", "미완료"))
    else:
        T = beh["table"]
        L += ["### 행동 지표 (seed 평균, 괄호 = 정확도 기준 통과 seed 수)\n",
              "| sparsity | 재학습 | n | avg | WGA | S | d_correct | d_wrong | 평균 Δp |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for sp in SPS:
            for reg in REGIMES:
                key = ("dense", "00") if (reg == "one-shot" and sp == "00") else (reg, sp)
                rows = T.get(key)
                lab = f"{reg}" + (" (= dense)" if key[0] == "dense" else "")
                if not rows:
                    L.append(f"| {int(sp)}% | {lab} | 0 | 미측정 | | | | | |")
                    continue
                a = np.array([r[:6] for r in rows], float)
                npass = sum(r[6] for r in rows)
                mu = a.mean(0)
                L.append(f"| {int(sp)}% | {lab} | {len(rows)} ({npass}) | {mu[0]*100:.2f} | {mu[1]*100:.2f} | "
                         + " | ".join(f"{v:.4f}" for v in mu[2:]) + " |")
        L.append(f"\n실질 크기 기준 (dense seed 표준편차 × 2): "
                 + ", ".join(f"{M_LAB[j]} {beh['thr'][j]:.4f}" for j in range(4)) + "\n")

        L += ["### 짝지은 비교 (같은 seed, two-way bootstrap 95% CI, ★ = H1 규칙상 변함)\n",
              "| 비교 | seed (정확도 통과) | ΔS | Δd_correct | Δd_wrong | Δ평균 Δp | 판정 |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
        comps = [(("one-shot", "50"), ("dense", "00"), "one-shot 50% − dense"),
                 (("one-shot", "70"), ("dense", "00"), "one-shot 70% − dense"),
                 (("5ep", "00"), ("dense", "00"), "5ep ft0 − dense"),
                 (("5ep", "50"), ("5ep", "00"), "5ep 50% − 5ep ft0 (H1 기준)"),
                 (("5ep", "70"), ("5ep", "00"), "5ep 70% − 5ep ft0 (H1 기준)"),
                 (("20ep", "50"), ("20ep", "00"), "20ep 50% − ft0 (STEP 3 재현)"),
                 (("20ep", "70"), ("20ep", "00"), "20ep 70% − ft0 (STEP 3 재현)"),
                 (("5ep", "50"), ("20ep", "50"), "5ep 50% − 20ep 50%"),
                 (("5ep", "70"), ("20ep", "70"), "5ep 70% − 20ep 70%"),
                 (("5ep", "70"), ("one-shot", "70"), "5ep 70% − one-shot 70%")]
        verdict_5ep = {}
        for a, b, name in comps:
            res = beh["compare"](a, b)
            if res is None:
                L.append(f"| {name} | - | | | | | 미측정 |")
                continue
            d, lo, hi, ch, use, inc = res
            cells = [f"{d[j]:+.4f} [{lo[j]:+.4f}, {hi[j]:+.4f}]" + (" ★" if ch[j] else "") for j in range(4)]
            v = ("변함: " + ", ".join(M_LAB[j] for j in range(4) if ch[j])) if ch.any() else "변함 없음"
            note = "" if inc else " (정확도 미달 seed 포함, 참고)"
            L.append(f"| {name} | {','.join(map(str, use))} ({len(inc)}) | " + " | ".join(cells) + f" | {v}{note} |")
            if "H1 기준" in name:
                verdict_5ep[a[1]] = v
        L.append("")

        # 5 epoch 히트맵
        passing = {sp: [s for s in SEEDS if ("5ep", sp, s) in beh["acc"] and beh["passed"](("5ep", sp, s))]
                   for sp in SPS}
        L.append(f"정확도 기준 통과 5 epoch 모델: " + ", ".join(f"{int(sp)}% seed {passing[sp]}" for sp in SPS) + "\n")
        st_a_h2 = {}
        models5 = [f"prune5ep_sp{sp}" for sp in SPS]
        seeds5 = [s for s in SEEDS if all(s in passing[sp] for sp in SPS)]
        for method in ("gradcam", "ig"):
            have = seeds5 and all(os.path.exists(f"runs/{m}_seed{s}/attr_{method}.csv.gz")
                                  for m in models5 for s in seeds5)
            if have and len(seeds5) == 3:
                r, ma = h2_eval(method, None, models5, "prune5ep_sp00", ["prune5ep_sp50", "prune5ep_sp70"], nb, sd)
                st_a_h2[method] = (r, ma)
        if st_a_h2:
            L += ["### 5 epoch 모델의 히트맵 (기준 = 5 epoch ft0)\n"] + H2_HEAD
            for method, (r, ma) in st_a_h2.items():
                L.append(h2_rows(f"{H2.MLAB[method]} · 20 epoch (기본)", base[method][0]))
                L.append(h2_rows(f"{H2.MLAB[method]} · 5 epoch", r["prune5ep_sp70"]))
                L.append(h2_rows(f"{H2.MLAB[method]} · 5 epoch, 50% (참고)", r["prune5ep_sp50"]))
            L.append("\n'50% (참고)' 행의 '70% ρ' 열과 H2b 는 50% 모델 값이다.\n")
            L += ["| 방법 | 평균 Attr_bg 5ep ft0 | 5ep 50% | 5ep 70% | 20ep ft0 | 20ep 70% |", "| --- | --- | --- | --- | --- | --- |"]
            for method, (r, ma) in st_a_h2.items():
                mb = base[method][1]
                L.append(f"| {H2.MLAB[method]} | {ma['prune5ep_sp00']:.4f} | {ma['prune5ep_sp50']:.4f} | "
                         f"{ma['prune5ep_sp70']:.4f} | {mb['prune_sp00']:.4f} | {mb['prune_sp70']:.4f} |")
            L.append("")
        else:
            L.append("5 epoch 모델의 Grad-CAM·IG 결과가 아직 없거나, 세 seed 가 모두 정확도 기준을 통과하지 못했다.\n")

        def core_changed(v):
            names = v.split(":", 1)[1].split(", ") if ":" in v else []
            return any(n.strip() in ("S", "평균 Δp") for n in names)
        core_h1 = not any(core_changed(v) for v in verdict_5ep.values())
        h1_txt = "; ".join(f"{int(k)}% {v}" for k, v in verdict_5ep.items())
        parts = [f"H1 (5ep ft0 대비) {h1_txt} → S·평균 Δp {'변함 없음' if core_h1 else '변함'}"]
        all_ok = core_h1
        for method, (r, _) in st_a_h2.items():
            res = same_conclusion(method, r["prune5ep_sp70"], base[method][0])
            parts.append(diff_text(H2.MLAB[method], res, r["prune5ep_sp70"], base[method][0]))
            all_ok &= res[0]
        if len(st_a_h2) < 2:
            line = "**ST-A 판정: 히트맵 미완료** — " + " / ".join(parts)
        else:
            line = (f"**ST-A 판정: 핵심 결론 {'유지' if all_ok else '달라짐'}** — " + " / ".join(parts))
        L += [line, ""]
        summary.append(("ST-A fine-tuning 길이", line))
        os.makedirs(args.fig_dir, exist_ok=True)
        p = fig_st_a(beh, args.fig_dir)
        rel = os.path.relpath(p, os.path.dirname(os.path.abspath(args.out))).replace("\\", "/")
        L.append(f"![ST-A]({rel})\n")

    # ---- 요약을 맨 위에
    head = ["## 판정 요약\n",
            "핵심 결론 = H1 S·평균 Δp 변함 없음 / H2a 추적 약화 없음 / H2b 70% 변화 > 잡음 (새 쪽). "
            "H2a 의 '지지 안 됨' 과 '반대 (추적 강화)' 는 둘 다 '약화 없음' 이므로 핵심은 같고, "
            "세부 판정이 다르면 따로 적는다.\n",
            "| stress test | 기본 결론이 유지되는가 |", "| --- | --- |"]
    head += [f"| {n} | {t} |" for n, t in summary]
    L[4:4] = head + [""]
    text = "\n".join(L) + "\n"
    print(text)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
