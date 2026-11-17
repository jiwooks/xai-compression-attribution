# 사전 등록 — seed 3·4 보조 분석

작성 2026-11-15, 결과를 보기 전에 정함. 주 분석(seed 0~2)과 따로 확인하는 보조 분석이라 이 파일에 따로 둔다.
결정 번호는 CLAUDE.md §4 의 1~13 에 이어 14~21 로 붙인다.

## 결정 사항

seed 를 5개로 늘려도 결론이 같은지 보는 **보조 분석**이다. 새 seed 번호(3, 4)와 bootstrap seed(0)는
결과를 본 뒤에 바꾸지 않는다. 결과는 결론이 같든 다르든 주 결과와 나란히 보고한다.

14. **seed 3, 4 는 보조 분석이다.** 주 판정(H1, H2a, H2b)은 seed 0~2 로만 낸다.
    기존 결과 파일(`results/h1_k5.md`, `attribution_k5.md`, `pruning_k5.md`, `stress_k5.md`)과 `results/figs/` 는 덮어쓰지 않는다.
    보조 결과는 `results/h1_k5_seeds5.md`, `results/attribution_k5_seeds5.md`, 그림은 `results/figs_seeds5/`.
15. **대상: erm, prune_sp00/30/50/70 × seed 3, 4 (10개).** 80% 이상과 stress test(5 epoch, one-shot, blur IG)는 하지 않는다.
    설정은 seed 0~2 와 같다.
    - 학습 100 epoch, pruning 20 epoch.
    - Grad-CAM 기본, IG 200 step·검정 baseline.
    - RISE 마스크 2,000개·rise_seed 0·기존 `results/rise_images.csv` 1,000장 (목록을 다시 만들지 않는다).
    - dense seed 3·4 도 `measure_shortcut --save_probs` 로 반사실 확률·Δp 를 저장한다 (H1 기준과 H2 의 Δp 에 필요).
16. **dense seed 3 또는 4 가 정확도 기준(avg 90%, 격차 15pt)에 못 미치면 seed 2 와 같이 재학습 없이 쓰고 명시한다.**
    다른 seed 로 바꾸지 않는다.
17. **pruning 모델이 dense 대비 2pt 넘게 떨어진 경우.**
    - H1: 기존 규칙대로 그 sparsity 합산에서 뺀다.
    - H2a·H2b: seed 3 또는 4 의 sp30/50/70 중 하나라도 해당되면 그 seed 를 보조 H2 분석 전체에서 뺀다 (짝 유지).
18. **H1 실질 크기 기준(보조).** 네 지표(S, d_correct, d_wrong, 평균 Δp) 모두 dense seed 5개 표준편차(ddof=1) × 2 로 정한다.
    5개 기준 판정과 3개 기준(기존 값) 판정을 나란히 쓴다.
    나머지 규칙(짝지은 two-way bootstrap 2,000회, bootstrap seed 0, "CI 0 미포함 AND |Δ| ≥ 기준")은 그대로다.
19. **H2a(보조).** 70% 주 지표가 같은 seed ft0 의 95% CI 밖인 seed 가 5개 중 3개 이상이고 같은 방향이면 판정한다.
    - 아래 = 지지, 위 = 반대 "추적 강화".
    - seed 가 빠져 4개면 3개 이상이다. 일반식은 포함 seed 수의 과반.
    - 보조 지표·그룹별·30%·50% 도 같은 규칙. 이미지 bootstrap 인덱스는 기존과 같다 (seed 0, 2,000회).
20. **H2b(보조).** 변화량 = 포함 seed 평균, 잡음 = ft0 의 모든 seed 쌍(5개면 10쌍) 평균.
    해석 제외에 쓰는 H1 판정은 `results/h1_k5_seeds5.md` 의 5개 기준 판정이다.
21. **attribution 점검(seed 3·4).** 하나라도 실패하면 멈추고 상의한다.
    - 원본 예측 이미지별 대조 5장 이하 (결정 13).
    - IG completeness 상대오차 중앙값 ≤ 5%.
    - 가중치 무작위화 sanity 를 erm seed 3·4 에 200장으로 다시 해, Grad-CAM·IG·RISE centered Spearman 중앙값 < 0.5.

## 주 결과를 지키는 장치

- `analyze_h1.py`, `analyze_h2.py`, `summarize_pruning.py` 에 `--seeds` 를 넣는다. 기본값은 0 1 2 다.
  이 세 스크립트는 `runs/` 의 seed 를 자동으로 모으거나 seed 를 하드코딩하고 있어서, 필터가 없으면 seed 3·4 가 주 결과에 섞인다.
- 기본값으로 다시 돌린 `h1_k5.md`, `attribution_k5.md`, `pruning_k5.md`, `stress_k5.md` 를 임시 경로에 쓰고
  기존 파일과 diff 한다. 다르면 멈춘다.
- 보조 실행은 `--out`, `--fig_dir`, `--h1` 을 따로 줘서 `results/*_seeds5.md`, `results/figs_seeds5/` 에만 쓴다.
