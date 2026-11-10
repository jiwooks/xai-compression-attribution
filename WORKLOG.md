# 작업 기록

진행 상황과 그 과정에서 알아낸 것을 날짜순으로 남긴다.
최신 항목이 위로 온다. 새 항목은 `## YYYY-MM-DD` 로 시작한다.

---

## 2026-11-10 (5) — P4 stress test (ST-A fine-tuning 길이, ST-B 측정 방식, ST-C IG baseline)

### 한 줄 요약

- **세 stress test 모두 핵심 결론이 유지된다.** 핵심 결론은 세 가지다.
  - H2a: 압축해도 추적이 약해지지 않는다.
  - H2b: 70% 에서 히트맵이 seed 간 잡음보다 크게 새 쪽으로 이동한다.
  - H1: 재학습 후 S·평균 Δp 가 변하지 않는다.
- **H2b 가 가장 단단하다.** 경계 띠를 빼도, 7×7 로 봐도, blur baseline 이어도, 5 epoch 재학습이어도 변화량과 이동 방향이 거의 같다.
  - Grad-CAM 평균 Attr_bg 변화: 20ep −0.031, 5ep −0.031.
- **새로 알게 된 것: 재학습 전(one-shot)에는 압축이 S 를 크게 올린다.**
  - 50% 에서 +0.040, 70% 에서 +0.057 이다. 실질 크기 기준 0.0069 의 6~8배다.
  - 5 epoch 만 재학습해도 이 증가가 사라진다.
  - 그래서 "≤70% 에서 행동이 변하지 않는다" 는 정확히는 "재학습이 압축의 효과를 되돌린다" 이다.
  - one-shot 은 정확도가 3~8pt 떨어져 정확도 기준 미달이다. 참고 결과로만 쓴다.
- **세부 판정은 달라진 곳이 있다.** 5 epoch 의 Grad-CAM·IG 와 blur IG 에서 H2a 가 "지지 안 됨" 대신 "반대 (추적 강화)" 로 나온다.
  - 어느 쪽이든 추적이 약해지지 않는다는 점은 같다.
  - IG 는 baseline 을 바꾸면 ρ 의 부호가 바뀐다 (ft0 +0.010 → −0.095). IG 의 이미지별 Attr_bg 로 추적을 말하면 안 된다.

### 실행 방법

```bash
cd /data/$USER/xai_project
# ST-A: 5 epoch 재학습 + one-shot 저장·측정 (seed 당 약 15분)
for s in 0 1 2; do EXTRA="--epochs 5 --eval_every 1 --save_oneshot" sbatch scripts/prune_slurm.sh $s "0 50 70" prune5ep; done
for s in 0 1 2; do sbatch scripts/attr_slurm.sh $s "prune5ep_sp00 prune5ep_sp50 prune5ep_sp70" gradcam; done
for s in 0 1 2; do EXTRA="--ig_steps 200" sbatch scripts/attr_slurm.sh $s "prune5ep_sp00 prune5ep_sp50 prune5ep_sp70" ig; done
# ST-C: IG blur baseline (모델당 약 28분)
for s in 0 1 2; do EXTRA="--ig_steps 200 --ig_baseline blur" OUT_SUFFIX=_blur sbatch scripts/attr_slurm.sh $s "prune_sp00 prune_sp70" ig; done
# 분석 (ST-B 포함). 로컬 PC 에서 돌렸다 (약 11분). 클러스터에서는 srun CPU 1개로.
python analyze_stress.py --out results/stress_k5.md
```

### 코드 변경

- `prune_finetune.py --save_oneshot`: 재학습 전 모델을 `oneshot.pt` 로, 성능을 `oneshot_test.json` 으로 저장한다.
  전역 magnitude pruning 은 결정적이라 같은 dense 에서 같은 one-shot 모델이 나온다.
- `prune_slurm.sh`: `oneshot.pt` 가 있으면 `shortcut_k5_oneshot*` 로 측정한다.
  이 측정이 실패해도 잡을 멈추지 않는다.
- `compute_attribution.py --ig_baseline blur`: 정규화된 입력을 가우시안 σ=10px 로 흐려 baseline 으로 쓴다
  (커널 61, reflect 패딩).
- `analyze_h2.py`: `load`, `verdict_h2a`, `paired_diff`, `compute_h2b` 에 열·모델·기준 모델 인자를 추가했다.
  기본값은 그대로라 STEP 5 결과가 같게 나온다. analyze_stress 의 기본 행에서 확인했다.
- `analyze_stress.py` (새 파일): ST-A/B/C 를 analyze_h1·h2 의 함수와 판정 규칙으로 다시 판정한다.
  판정은 두 단계다.
  - 핵심: H2a 의 약화 여부와 H2b 결론.
  - 세부: H2a 판정 문자열.

### 결과 (전체는 `results/stress_k5.md`, 그림 `results/figs/stress_a_finetune_length.png`)

**ST-A 행동 (seed 평균)**

| sparsity | one-shot | 5 epoch | 20 epoch |
|---|---|---|---|
| 0% | S 0.081 / Δp 0.185 / avg 90.0 (= dense) | 0.088 / 0.197 / 89.5 | 0.086 / 0.195 / 89.6 |
| 50% | **0.122 / 0.253** / 86.8 (WGA 59.7) | 0.081 / 0.191 / 89.9 | 0.083 / 0.195 / 89.8 |
| 70% | **0.138 / 0.263** / 82.6 (WGA 14.2) | 0.081 / 0.197 / 89.3 | 0.085 / 0.206 / 89.3 |

- one-shot − dense: 50% 와 70% 모두 S, d_correct, 평균 Δp 가 판정 기준을 크게 넘어 "변함" 이다.
  - 70% 는 d_wrong 만 CI 가 0 을 포함한다.
  - 원래 맞은 사진만 보는 d_correct 도 +0.025 / +0.039 올랐다. 그래서 단순히 오답이 늘어난 효과만은 아니다.
- 5 epoch 모델 9개는 모두 정확도 기준을 통과했다 (dense 대비 최대 −1.1pt).
  - best epoch 는 1~5 이고 1 epoch 째인 경우가 많다. 재학습 첫 epoch 에 대부분 회복된다.
- 5ep 50% − 5ep ft0: 변함 없음.
- 5ep 70% − 5ep ft0: **d_wrong 만 −0.046 [−0.078, −0.014] 로 "변함"** 이다 (배경 의존 감소 쪽).
  - S, d_correct, 평균 Δp 는 변하지 않았다.
  - d_wrong 은 원래 틀린 약 10% 사진의 지표라 잡음이 크다. 20ep 70% 에서도 −0.023 (CI 0 포함) 이었다.
- 5ep − 20ep (같은 sparsity): S·평균 Δp 차이는 판정 기준 안이다. 70% 의 d_wrong 은 −0.018 로 "변함".

**ST-A 히트맵 (5 epoch, 기준 = 5 epoch ft0, 70%)**

| 방법 | ρ 5ep ft0 → 70% | H2a | H2b 변화량 / 잡음 | 평균 Attr_bg 변화 | 20 epoch 값 (참고) |
|---|---|---|---|---|---|
| Grad-CAM | +0.335 → +0.359 | 반대 (↑ · ↑) | 0.038 / 0.011 | −0.031 | ρ +0.333 → +0.332, 변화 −0.031 |
| IG | +0.014 → +0.074 | 반대 (↑ ↑ ↑) | 0.025 / 0.011 | −0.016 | ρ +0.010 → +0.039, 변화 −0.017 |

50% 는 두 방법 모두 H2a "지지 안 됨" 이다. H2b 변화량은 잡음보다 0.0004~0.0008 커서 무시할 수준이다.

**ST-B 측정 방식 (70% vs ft0)**: 모든 변형에서 H2a 판정 문자열까지 기본과 같다.

| 변형 | ρ ft0 → 70% | H2b 변화량 / 잡음 | 평균 Attr_bg 변화 |
|---|---|---|---|
| Grad-CAM 기본 / 띠 제외 / 7×7 | +0.333→+0.332 / +0.334→+0.333 / +0.332→+0.331 | 0.036/0.010, 0.038/0.011, 0.038/0.011 | −0.031 / −0.034 / −0.033 |
| IG 기본 / 띠 제외 | +0.010→+0.039 / +0.011→+0.041 | 0.022/0.011, 0.024/0.011 | −0.017 / −0.019 |
| RISE raw / raw + 띠 제외 (참고) | +0.109→+0.112 / +0.088→+0.096 | 0.0010/0.0005 둘 다 | −0.0005 둘 다 |

RISE centered 맵은 이미지별 값만 저장돼 있어 띠 제외를 계산할 수 없다. 그래서 raw 맵끼리 비교했다.
raw 맵은 sanity 를 통과하지 못해 판정에서는 뺐다.

**ST-C IG baseline (70% vs ft0)**

| baseline | ρ ft0 → 70% | H2a | H2b 변화량 / 잡음 | 평균 Attr_bg ft0 → 70% |
|---|---|---|---|---|
| 검정 (기본) | +0.010 → +0.039 | 지지 안 됨 (↑ · ·) | 0.022 / 0.011 | 0.633 → 0.616 |
| blur σ=10 | **−0.095 → −0.049** | 반대 (↑ ↑ ↑) | 0.021 / 0.011 | 0.578 → 0.566 |

blur 의 completeness 상대오차 중앙값은 1.0~1.2% 다.

### 해석

1. **"행동은 같은데 히트맵은 새 쪽으로 이동한다" (H2b) 는 측정 방식·baseline·재학습 길이에 흔들리지 않는다.**
   이 프로젝트에서 가장 단단한 결과다.
2. **이미지 단위 추적은 압축으로 약해지지 않는다 (H2a 기각 쪽).**
   재학습을 짧게 하면 Grad-CAM·IG 의 추적이 오히려 조금 강해진다.
   IG 는 baseline 에 따라 ρ 부호가 바뀐다. IG 는 추적 도구로 쓸 수 없고 "IG 는 추적하지 않는다" 로 정리한다.
3. **압축 자체는 shortcut 의존을 키우고, 재학습이 그것을 되돌린다 (one-shot).**
   - STEP 3 의 "≤70% 변함 없음" 은 재학습을 거친 모델에 대한 말이다.
   - RQ 의 "압축으로 spurious 의존이 변하는 상황" 이 실제로 일어나는 곳은 one-shot 모델이다.
   - 그러나 one-shot 은 정확도 기준(2pt) 미달이라 계획상 히트맵을 계산하지 않았다.
4. 5ep 70% 의 d_wrong 감소는 히트맵이 새 쪽으로 이동한 방향과 같다. 원래 틀린 사진에 한정된, 잡음이 큰 지표라
   "행동이 같다" 는 전제를 뒤집을 정도는 아니다. 보고서에는 적는다.

### 한계

- 재현 전이다. seed 는 3개다.
- one-shot 결과는 정확도 기준 미달이라 참고로만 쓴다. S 증가에는 성능 저하 효과가 섞여 있을 수 있다.
  다만 원래 맞은 사진만 보는 d_correct 도 올랐다.
- ST-B 의 RISE 는 centered 맵의 띠 제외 값이 없어 raw 맵끼리만 비교했다.
- ST-C 는 IG 만, ft0·70% 만 봤다.
- ST-A 의 5 epoch H1 판정 문턱은 dense seed 표준편차로 정했다 (STEP 3 과 같다).

### 다음에 할 일

- [ ] (결정 필요) one-shot 50%·70% 모델의 히트맵을 계산할지 정한다.
  - 행동이 실제로 크게 변한 유일한 조건이라, RQ ("변화를 추적하는가") 를 직접 시험할 수 있다.
  - 정확도 기준 미달이라 계획 밖이다.
- [ ] STEP 2~6 재현 (다른 계정)
- [ ] 결과를 팀원과 공유하고 보고서 서술 방향(RQ 답) 논의

---

## 2026-11-10 (4) — STEP 5 히트맵 채점 (H2a, H2b) + NAS 데이터 복구

### 한 줄 요약

- **H2a (모델 안 추적이 압축으로 약해지는가) — 지지되지 않는다.** 주 지표(부분 Spearman ρ(Attr_bg, Δp | area))는
  70% 에서 Grad-CAM 은 그대로, IG 는 1/3 seed 만 상승, RISE 는 세 seed 모두 **오히려 상승**(추적 강화)했다.
- **H2b (행동이 같은데 설명이 바뀌었는가) — 70% 에서 세 방법 모두 "그렇다".** 행동(H1)은 ft0 대비 "변함 없음" 인데
  이미지별 히트맵 변화가 seed 간 잡음보다 1.7~3.5배 크고, 방향은 모두 **새 쪽**(배경 attribution 감소)이다.
- 같은 70% 에서 행동 평균 Δp 는 +0.011 (배경 의존 쪽, 판정 기준 미달) 이었다. **히트맵의 수준은 행동과 반대로 움직이고,
  이미지 간 순위(추적)는 유지된다.** 재현 전 수치다.

### 실행 방법

```bash
cd /data/$USER/xai_project
srun -p debug_ugrad --ntasks=1 --cpus-per-task=1 --mem=24G -t 1:30:00 \
     /data/$USER/miniconda3/envs/xai/bin/python analyze_h2.py --out results/attribution_k5.md   # 약 12분
```

### 방법 (CLAUDE.md §4 결정 4, 추가 결정 11)

- 입력: `shortcut_k5_dp.csv.gz` (STEP 3+) 와 `attr_<method>.csv.gz` (STEP 4) 를 img_id 로 결합. 15개 모델 × 3방법.
  RISE 는 1,000장, `attr_bg_centered`. `C` 는 각 방법의 주 Attr_bg 로 다시 계산.
- H2a 주 지표: rank(Attr_bg), rank(Δp) 를 rank(area) 에 회귀한 잔차의 Pearson (= 순위 상관의 1차 부분상관 공식).
  보조: 원시 Spearman, AUROC(Δp ≥ 0.10), `C` 의 부분 ρ. 이미지 bootstrap 2,000회, 재표본마다 순위를 다시 매긴다.
  모든 모델이 같은 재표본 인덱스를 써서 ft0 대비 차이가 짝지어진다. 그룹별은 그룹마다 따로 재표본.
- H2a 판정: 70% 가 같은 seed ft0 의 95% CI 밖이고 seed 2개 이상 같은 방향 (결정 4). 짝지은 차이 CI 도 함께 보고.
- H2b: 변화량 = 이미지별 |Attr_bg(sp) − Attr_bg(ft0)| (같은 seed) 의 seed 평균, 잡음 = |Attr_bg(ft0,a) − Attr_bg(ft0,b)| 세 쌍 평균.
  중앙값 차이와 평균 방향 변화의 이미지 bootstrap CI. H1 판정은 `results/h1_k5.md` 에서 읽는다.

### 결과 (전체는 `results/attribution_k5.md`, 그림은 `results/figs/`)

**판정 요약 (70% vs ft0)**

| 방법 | H2a 부분 ρ ft0 → 70% | Δρ [CI] | H2a 판정 (seed 별) | H2b 변화량 / 잡음 중앙값 | 평균 Attr_bg 변화 [CI] | H1 |
|---|---|---|---|---|---|---|
| Grad-CAM | +0.333 → +0.332 | −0.001 [−0.019, +0.016] | 지지 안 됨 (· · ·) | 0.036 / 0.010 | −0.031 [−0.033, −0.029] | 변함 없음 |
| IG | +0.010 → +0.039 | +0.029 [+0.017, +0.043] | 지지 안 됨 (↑ · ·) | 0.022 / 0.011 | −0.017 [−0.017, −0.016] | 변함 없음 |
| RISE (centered) | +0.215 → +0.305 | +0.090 [+0.050, +0.128] | 반대: 추적 강화 (↑ ↑ ↑) | 0.026 / 0.015 | −0.010 [−0.012, −0.007] | 변함 없음 |

30%, 50% 는 H2a 에서 세 방법 모두 "지지 안 됨". H2b 는 30% 에서 세 방법 모두 잡음보다 작다.
50% 는 Grad-CAM·IG 에서 변화량이 잡음보다 0.001 크고 CI 가 0 을 벗어나지만 크기가 무시할 수준이다 (RISE 는 잡음 수준).

**H2a 세부에서 짚을 것**

- **Grad-CAM 은 원시 Spearman(0.441 → 0.411)과 AUROC(0.714 → 0.697)가 70% 에서 떨어진다** (각각 3/3, 2/3 seed 가 CI 아래).
  그러나 새 면적을 통제한 부분 ρ 는 그대로다. 원시 지표의 하락은 면적 경로를 통한 것으로, 결정 7 에서 부분 ρ 를 주 지표로
  고른 이유가 실제로 작동한 사례다. 원시 지표만 봤다면 "H2a 지지" 로 잘못 판정했을 것이다.
- **IG 는 면적을 통제하면 거의 추적하지 않는다** (부분 ρ 0.01, 원시 0.22). IG 의 Attr_bg 는 새 면적을 반영할 뿐
  배경 의존과 거의 무관하다. 70% 에서 0.04 로 조금 오른다.
- RISE 는 70% 에서 추적이 강해진다 (0.215 → 0.305). 그룹별로는 네 그룹 모두 +0.06~+0.08 이지만 그룹당 250장이라 CI 가 넓다.
- 그룹별 Grad-CAM: landbird/land 는 70% 에서 추적이 오르고(+0.039), waterbird/land 는 내려간다(−0.035, CI 0 포함).

**모델 단위 (참고)**: 15개 모델의 Spearman(평균 Attr_bg, 평균 Δp) 는 Grad-CAM −0.16, IG −0.43, RISE +0.23.
모델 평균 히트맵으로 모델 간 배경 의존 순위를 매기면 Grad-CAM·IG 는 반대로 매긴다.

### 해석

1. **이미지 단위 추적(H2a)은 압축으로 나빠지지 않았다.** "높은 sparsity 에서 attribution 이 행동과 어긋난다" 는 원래 H2 의
   이미지 단위 버전은 이 범위(≤70%)에서 기각 쪽이다. RISE 는 오히려 좋아졌다.
2. **히트맵의 수준은 행동과 다른 이야기를 한다(H2b).** 70% 모델은 원본 모델과 행동 지표(S, d_correct, d_wrong, 평균 Δp)가
   판정상 같은데, 세 방법 모두 "배경을 덜 본다" 쪽으로 이동했다. 실제 행동은 배경 의존이 약간 늘어나는 쪽(+0.011)이다.
   따라서 압축 전후의 평균 히트맵을 비교해 "압축 모델이 shortcut 을 덜 쓴다" 고 결론 내리면 틀린다.
3. 둘을 합치면: **압축은 설명의 전역 수준(얼마나 배경을 가리키는가)을 바꾸지만, 같은 모델 안에서 이미지 간 상대 순위는 유지한다.**
   RQ 에 대한 답은 "이미지 순위로는 추적하지만, 모델 간 비교(압축 전후)에는 쓸 수 없다" 쪽이다.

### 한계

- 재현 전. 70% 의 평균 Δp 는 CI 가 0 을 벗어나는 경계선 결과라 "행동이 같다" 는 전제가 완전히 단단하지는 않다.
  dense 대비로는 "변함" 이다 (STEP 3+). H2b 해석은 ft0 대비 기준에 기대고 있다.
- H2b 잡음 기준(ft0 seed 간)은 서로 다른 dense 모델에서 출발해, 같은 seed 안 비교보다 잡음이 크다(보수적).
  그런데도 70% 변화량이 1.7~3.5배 크다.
- RISE 는 1,000장(그룹당 250장)이고 주 지표가 사전 계획과 다르다(centered). IG 는 200 step (계획 50/100).
- IG 는 sanity 0.39 로 입력 윤곽 성분이 있고, 면적 통제 후 추적이 거의 없다.
- 80% 이상, 구조적 pruning, 다른 데이터셋은 보지 않았다. stress test(P4) 전이다.

### NAS 데이터 복구

`/data/$USER/datasets/xai/` 가 11-06 경 통째로 없어진 것을 복구했다 (CLAUDE.md §5-17).

```bash
sbatch -w aurora-g1 --job-name=xai-restore --partition=batch_ugrad --ntasks=1 --cpus-per-task=1 --mem=8G \
       --time=0-3 --output=logs/restore-%A.out scripts/restore_nas_tars.sh       # 약 20분
```

- g1 의 `/local_datasets` 사본(완료 표식 확인)에서 tar 5개를 원래 구조대로 다시 묶었다:
  Waterbirds 467M, CUB 1.1G (segmentations 제외 + attributes.txt), segmentations 38M, Places 배경 29M, 반사실 k5 2.2G.
- 각 tar 를 같은 노드의 임시 폴더에 다시 풀어 원본과 파일 목록·크기가 일치함을 확인했다 (11,789 / 11,800 / 11,788 / 401 / 57,941 파일).
- 데이터가 한 번도 펼쳐진 적 없는 g2 에서 `stage_data.sh` + `stage_cf.sh k5` 로 처음부터 스테이징해 반사실 57,940장,
  segmentation 11,788장, 배경 400장이 펼쳐지는 것을 확인했다. 이제 `--exclude` 없이 잡을 걸 수 있다.
- 공식 배포 tarball 과 바이트 단위로 같지는 않다 (같은 파일을 다시 묶은 것).

### 다음에 할 일

- [x] stress test (P4) — 2026-11-10 (5) 항목
- [ ] STEP 2~5 재현 (다른 계정)
- [ ] 결과를 팀원과 공유하고 보고서 서술 방향(RQ 답) 논의

---

## 2026-11-10 (3) — STEP 4 attribution (Grad-CAM / IG / RISE)

### 한 줄 요약

모델 15개 × 방법 3개의 이미지별 `Attr_bg` 를 계산했다. sanity check 를 통과했고(RISE 는 centered 맵으로),
**세 방법 모두 70% 에서 평균 `Attr_bg` 가 낮아진다**(히트맵이 새 쪽으로 이동). 같은 70% 에서 행동 지표 평균 `Δp` 는
약간 올랐으므로 방향이 반대다. 판정은 STEP 5 (H2a·H2b) 에서 한다. 재현 전 수치다.

### 실행 방법

```bash
cd /data/$USER/xai_project && mkdir -p logs
X="--exclude=aurora-g2,aurora-g5,aurora-g6"          # NAS tar 유실로 데이터가 남은 노드만 (CLAUDE.md §5-17)
# RISE 이미지 목록 (CPU 1개)
srun -p debug_ugrad $X --ntasks=1 --cpus-per-task=1 --mem=4G -t 0:10:00 \
     bash -c 'source scripts/_common.sh && $PY compute_attribution.py --make_rise_list --wb_root $WB_ROOT'
# 점검: dense seed 0, 64장, png 저장
srun -p debug_ugrad $X --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 1:00:00 bash -c \
  'for m in gradcam ig rise; do EXTRA="--subset 64 --png_dir data_check/attr_smoke --n_png 6" OUT_SUFFIX=_smoke bash scripts/attr_slurm.sh 0 erm $m; done'
# 본 실행
M5="erm prune_sp00 prune_sp30 prune_sp50 prune_sp70"
for s in 0 1 2; do sbatch $X scripts/attr_slurm.sh $s "$M5" gradcam; done                 # 모델당 약 10초
for s in 0 1 2; do for R in "erm prune_sp00" "prune_sp30 prune_sp50" "prune_sp70"; do
  EXTRA="--ig_steps 200" sbatch $X scripts/attr_slurm.sh $s "$R" ig; done; done           # 모델당 약 28분
for s in 0 1 2; do for R in "erm prune_sp00 prune_sp30" "prune_sp50 prune_sp70"; do
  sbatch $X scripts/attr_slurm.sh $s "$R" rise; done; done                                 # 모델당 약 12분
sbatch $X scripts/attr_sanity_slurm.sh 0 200                                              # 가중치 무작위화
```

### 설정 (CLAUDE.md §4 2026-11-10 결정 6~9, 추가 결정 11~13)

- 설명 대상: 원본 사진의 예측 클래스 margin `z_pred − z_other`. RISE 는 예측 클래스 확률.
- Grad-CAM: `layer4` 출력, 공간 평균 gradient 가중, ReLU, 224 bilinear 업샘플(주). 7×7 은 마스크를 면적 평균으로 줄여 가중.
- IG: 검정 baseline(픽셀 0 을 정규화), 중점 Riemann **200 step** (아래 경위), 채널별 |attr| 합.
- RISE: 7×7 격자, p=0.5, 마스크 2,000개(시드 0, 모든 모델 공통), 업샘플 후 랜덤 시프트, 가린 곳은 검정, fp16 forward.
  **주 지표는 centered 맵** Σ(p_i − p̄)M_i/(N·p) 의 `Attr_bg` (`attr_bg_centered`). raw 는 보조.
- 마스크: CUB segmentation 을 이미지와 같은 기하(Resize 256 → CenterCrop 224, NEAREST)로 바꾼 뒤 >0.5.
  이미지와 마스크 전처리의 기하 변환이 같은 것을 코드에서 확인했다 (보간만 bilinear / nearest 로 다름).
- 경계 띠: 마스크 팽창 − 침식 (7×7 max-pool, ±3px). 분자·분모에서 모두 뺀다.
- RISE 이미지: `results/rise_images.csv`, 그룹당 250장 (pandas `groupby.sample`, random_state=0).
- 새 면적이 0 인 이미지, `Attr_bg` 가 NaN 인 이미지는 45개 결과 모두 0장이다.

### 점검

**스모크 테스트** (dense seed 0, 64장): `data_check/attr_smoke/` 의 png 18장. 세 방법 모두 새 몸통에 집중하고
마스크 윤곽이 새와 맞는다. RISE raw 맵은 전체에 약 0.9 의 바닥값이 깔려 있다.

**가중치 무작위화 sanity** (Adebayo et al. 2018, dense seed 0, 200장, `results/attr_sanity_erm_seed0.json`)
— 원래 모델 맵과 모든 가중치를 새로 초기화한 모델 맵의 이미지별 Spearman (동순위는 평균 순위)

| 방법 | 중앙값 | IQR | 판정 (기준 0.5) |
|---|---|---|---|
| Grad-CAM | −0.012 | −0.160 ~ 0.124 | 통과 |
| IG | 0.391 | 0.326 ~ 0.471 | 통과. 검정 baseline IG 는 `(x − x′)` 가 곱해져 사진 윤곽이 남는다 (알려진 성질, 한계에 기재) |
| RISE raw | **0.805** | 0.295 ~ 0.999 | **실패** |
| RISE centered | 0.004 | −0.058 ~ 0.098 | 통과 |

RISE raw 실패 원인: 모든 모델이 같은 마스크를 쓰므로 맵에 `p̄ × Σ M_i / (N p)` 패턴이 공유된다. 무작위 모델은
출력이 거의 상수라 맵이 이 패턴 자체가 되고, 원래 모델 맵에도 같은 패턴이 바닥값으로 깔린다.
패턴을 정확히 뺀 centered 맵에서 0.004 가 된 것으로 확인했다. → 사용자 결정으로 centered 를 주 지표로 (추가 결정 11).

**IG completeness**: 처음 규칙(50 step, 중앙값 > 5% 면 100 step)대로 돌리니 50 step 중앙값 3.8~9.6% 로 step 수가 섞였고
(13개 100 step, sp70 seed1·2 는 50 step), 100 step 에서도 erm_seed1 6.09%, prune_sp00_seed1 6.13% 가 기준을 넘었다.
→ 사용자 결정으로 **전부 200 step** (추가 결정 12). 200 step 중앙값 **0.9~2.2%** 로 15개 모두 통과.
100 step 과 200 step 의 평균 `Attr_bg` 차이는 0.0002 이하였다. 100 step 결과는 클러스터에 `attr_ig_s100.*` 로 보관.

**원본 예측 대조**: 이미지별로 `measure_shortcut` 의 원본 예측(p_w_orig > 0.5)과 맞췄다. 45개 중 5개 결과에서 1장씩 다르고,
모두 p_w_orig 가 0.4993~0.5005 인 결정 경계 사진이다. 배치 크기 차이(64·4 vs 256)에 따른 수치 차이로 본다.
처음에는 정확도 완전 일치를 요구해 Grad-CAM 3개가 멈췄다 → 이미지별 대조, 5장 허용으로 바꿨다 (추가 결정 13).

### 처리 시간 (RTX 3090 1장)

| 방법 | 이미지 | 모델당 | 15개 합계 |
|---|---|---|---|
| Grad-CAM | 5,794 | 약 9초 | 약 2분 |
| IG 200 step | 5,794 | 약 27.5분 | 약 6.9 GPU-시간 (잡 9개 병렬, 실제 약 56분) |
| RISE 2,000 마스크 | 1,000 | 약 11.7분 | 약 2.9 GPU-시간 (잡 6개 병렬, 실제 약 40분) |

계획(RISE 약 9 GPU-시간)보다 RISE 가 3배 빨랐다 (fp16, 마스크 배치 500).

### 결과 — 평균 Attr_bg (seed 평균 ± 표준편차)

균일한 맵이면 `1 − area` = 0.833 (전체), 0.830 (RISE 1,000장). 이보다 낮을수록 새를 본다.

| 모델 | Grad-CAM | Grad-CAM band | Grad-CAM 7×7 | IG | IG band | RISE centered (주) | RISE raw |
|---|---|---|---|---|---|---|---|
| dense | 0.6554 ± 0.0029 | 0.6752 ± 0.0031 | 0.6481 ± 0.0031 | 0.6323 ± 0.0033 | 0.6564 ± 0.0034 | 0.6531 ± 0.0033 | 0.8275 ± 0.0002 |
| ft0 | 0.6572 ± 0.0023 | 0.6769 ± 0.0025 | 0.6499 ± 0.0025 | 0.6325 ± 0.0023 | 0.6567 ± 0.0023 | 0.6538 ± 0.0026 | 0.8275 ± 0.0001 |
| 30% | 0.6562 ± 0.0023 | 0.6760 ± 0.0024 | 0.6489 ± 0.0025 | 0.6314 ± 0.0022 | 0.6555 ± 0.0022 | 0.6536 ± 0.0026 | 0.8275 ± 0.0000 |
| 50% | 0.6505 ± 0.0026 | 0.6701 ± 0.0028 | 0.6431 ± 0.0028 | 0.6268 ± 0.0018 | 0.6502 ± 0.0018 | 0.6523 ± 0.0024 | 0.8273 ± 0.0001 |
| **70%** | **0.6257 ± 0.0015** | 0.6427 ± 0.0015 | 0.6169 ± 0.0016 | **0.6158 ± 0.0034** | 0.6377 ± 0.0035 | **0.6440 ± 0.0028** | 0.8269 ± 0.0000 |

그룹별 (ft0 → 70%, seed 평균; landbird/land, landbird/water, waterbird/land, waterbird/water 순)

- Grad-CAM: 0.690 / 0.602 / 0.657 / 0.737 → 0.648 / 0.575 / 0.634 / 0.716 (네 그룹 모두 감소)
- IG: 0.602 / 0.680 / 0.554 / 0.652 → 0.574 / 0.672 / 0.530 / 0.651 (land 배경 그룹에서 감소)
- RISE centered: 0.711 / 0.635 / 0.602 / 0.666 → 0.682 / 0.614 / 0.606 / 0.674 (landbird 쪽 감소)

### 해석 (예비)

- 세 방법 모두 ft0 → 70% 에서 `Attr_bg` 가 0.01~0.03 낮아진다. seed 표준편차(0.002~0.003)의 약 4~15배이고 세 seed 가 같은 방향이다.
  50% 까지는 거의 그대로다. band·7×7 변형에서도 같은 경향이다.
- STEP 3+ 에서 같은 70% 의 행동 지표 평균 `Δp` 는 ft0 대비 +0.011 (배경 의존 쪽, 판정은 "변함 없음") 이었다.
  **히트맵은 "배경을 덜 본다" 쪽으로, 행동은 "배경을 조금 더 쓴다" 쪽으로 움직인다.** H2b 의 핵심 비교 대상이다.
- RISE raw 는 모델 간 차이가 0.0006 으로 거의 없다. 공유 마스크 패턴이 지배하기 때문이며 주 지표로 쓰지 않는다.
- 이 표는 모델 평균이다. 이미지 단위 추적(H2a)과 seed 간 잡음 대비 변화(H2b)는 STEP 5 에서 판정한다.

### 한계

- 재현 전. sanity 는 dense seed 0 한 모델, 200장만 봤다. 무작위화는 전체 가중치 한 번(층별 점진 무작위화는 안 함).
- IG sanity 0.39 는 기준 안이지만 0 보다 높다. IG 맵의 일부는 모델이 아니라 입력 윤곽에서 온다.
- RISE 는 1,000장이고 주 지표가 사전 계획과 다르다 (centered). IG step 수도 사전 계획(50/100)과 다르다 (200).
- `C` 열은 raw `Attr_bg` 기준이다. RISE centered 의 `C` 는 STEP 5 에서 `attr_bg_centered` 와 `area` 로 다시 계산한다.

### 규칙 위반 기록

STEP 4 준비 중 패키지 확인을 위해 **마스터 노드에서 `python -c "import sys"` 를 한 번 실행했다** (아무것도 import 하지 않음).
이후 python 은 계산 노드(srun/sbatch)에서만 실행했다.

### 다음에 할 일

- [ ] STEP 5 (P3) — H2a, H2b
- [ ] NAS 데이터 tar 복구
- [ ] STEP 4 재현 (다른 계정, 최소 dense seed 0 의 Grad-CAM)

---

## 2026-11-10 (2) — 반사실 확률 저장 + Δp + H1 재판정

### 한 줄 요약

ft0 대비, 정확도 유지 구간(≤70%)에서는 `S`, `d_correct`, `d_wrong`, 평균 `Δp` **네 지표 모두 "변함 없음"** 이다.
다만 새 지표 평균 `Δp` 는 sparsity 에 따라 단조에 가깝게 오르고, 70% 에서 CI 가 0 을 벗어나며(+0.0114) 실질 크기
기준(0.0135)에 조금 못 미친다. 80%(seed 1, 2)는 평균 `Δp` 가 +0.0226 으로 "변함" 이다. 재현 전 수치다.

### 실행 방법

```bash
cd /data/$USER/xai_project && mkdir -p logs
# 회귀 확인 (srun, seed 0 dense 하나)
srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 0:30:00 \
     bash scripts/remeasure_slurm.sh 0 "erm" </dev/null
# 본 실행 — 잡 3개, 각 약 16분 (모델 8개 × 2분)
for s in 0 1 2; do sbatch --exclude=aurora-g2,aurora-g5,aurora-g6 scripts/remeasure_slurm.sh $s; done
# 분석 — GPU 없는 잡은 CPU 1개만 허용된다
srun -p debug_ugrad --ntasks=1 --cpus-per-task=1 --mem=16G -t 0:30:00 \
     /data/$USER/miniconda3/envs/xai/bin/python analyze_h1.py --out results/h1_k5.md
```

### 바뀐 코드

- `measure_shortcut.py`: `predict()` 가 waterbird softmax 확률도 돌려준다 (flip 은 그대로 argmax).
  결과 JSON 에 `dp` (전체·그룹별·그룹×정오답별 평균 Δp) 추가.
  `--save_probs` → `shortcut_k5_rows.csv.gz` (반사실 행별 p_w, flip), `shortcut_k5_dp.csv.gz` (이미지별 Δp, 원본 p_w).
  `--regress_check <기존 JSON>` → S·BFR·그룹별·CI 값이 하나라도 다르면 저장하지 않고 멈춘다.
- `scripts/remeasure_slurm.sh`: seed 하나의 erm + prune_sp00~95 를 재측정 (회귀 확인 포함).
- `scripts/prune_slurm.sh`: 측정에 `--save_probs` 추가 (P4 ST-A 에서 쓴다).
- `scripts/stage_cf.sh`: 완료 표식을 NAS tar 확인보다 먼저 본다 (아래 데이터 문제 참고).
- `analyze_h1.py`: 짝지은 two-way bootstrap (이미지 + 배경 타입별 층화, 모든 모델 같은 가중치, 2,000회).

### 회귀 확인

**24개 모델 전부 통과.** 기존 `shortcut_k5.json` 의 S, BFR_opp, BFR_same, 그룹별 값, 이미지/two-way CI,
`by_group_and_orig_correct`, 원본 정확도가 1e-12 안에서 같다. git diff 도 `dp` 항목 추가(삽입만)뿐이다.
`analyze_h1.py` 의 점추정은 JSON 의 S·평균 Δp, `summarize_pruning.decompose` 의 d_correct·d_wrong 와 일치한다.

### ★ 데이터 문제: NAS 의 `/data/$USER/datasets/xai/` 가 없어졌다

11-03 까지 있던 tar(Waterbirds, CUB, segmentations, Places 배경, 반사실 k5)가 폴더째 사라졌다. 상위 폴더 수정 시각은
11-06 12:46. 계산 노드 g1, g3, g4, g7, g8 의 `/local_datasets/$USER/xai` 에는 완료 표식과 함께 전부 남아 있어
(반사실 57,940장, segmentations 200 폴더) 이번 실행은 그 노드에만 걸었다(`--exclude=aurora-g2,aurora-g5,aurora-g6`).
g5 는 Waterbirds·CUB 만 있고 반사실·배경이 없다. **로컬 디스크는 언제 지워질지 모르므로 NAS tar 복구가 필요하다.**

### 결과 (seed 3개, 전체는 `results/h1_k5.md`)

실질 크기 기준 (dense seed 표준편차 × 2): S 0.0069, d_correct 0.0045, d_wrong 0.0138, 평균 Δp 0.0135.

모델별 평균 Δp: dense 0.1846 / ft0 0.1946 / 30% 0.1966 / 50% 0.1949 / 70% 0.2060 / 80% 0.2236 / 90% 0.2438 / 95% 0.3736

**ft0 대비 (주 판정, seed 합산, Δ [95% CI])**

| 모델 | 포함 seed | ΔS | Δd_correct | Δd_wrong | Δ평균 Δp | 판정 |
|---|---|---|---|---|---|---|
| 30% | 0,1,2 | +0.0007 [−0.0022, +0.0034] | +0.0004 [−0.0017, +0.0025] | +0.0064 [−0.0045, +0.0177] | +0.0020 [+0.0007, +0.0032] | 변함 없음 |
| 50% | 0,1,2 | −0.0035 [−0.0085, +0.0014] | −0.0016 [−0.0054, +0.0019] | −0.0092 [−0.0293, +0.0112] | +0.0003 [−0.0029, +0.0037] | 변함 없음 |
| 70% | 0,1,2 | −0.0006 [−0.0081, +0.0070] | +0.0009 [−0.0047, +0.0067] | −0.0233 [−0.0525, +0.0076] | +0.0114 [+0.0056, +0.0174] | 변함 없음 |
| 80% | 1,2 | +0.0024 [−0.0073, +0.0119] | +0.0022 [−0.0055, +0.0098] | −0.0339 [−0.0703, +0.0025] | **+0.0226 [+0.0137, +0.0311]** | 변함: 평균 Δp |

90%, 95% 는 세 seed 모두 정확도 기준으로 제외된다 (seed 별로는 S·d_correct·Δp 모두 "변함").

### 해석

- **H1 (ft0 대비) 은 정확도 유지 구간에서 기각 쪽이다.** 네 지표 모두 판정 규칙을 넘지 않는다. STEP 3 의 결론과 같다.
- **평균 Δp 가 S 보다 민감하다.** S 는 70% 까지 평평하지만 Δp 는 50% → 70% → 80% 로 오른다.
  70% 는 CI 가 0 을 벗어나지만 크기가 기준의 84% 이고, seed 별로 +0.026 / +0.011 / −0.003 으로 갈린다.
  경계선 결과로 보고하고, "변함" 으로 쓰지 않는다.
- 70% 의 Δp 상승은 원래 틀린 다수 그룹(landbird/land 오답 0.38 → 0.53, 4~8장)과 원래 맞은 landbird 이미지
  (land 0.167 → 0.183, water 0.127 → 0.143)에서 온다. waterbird 쪽은 거의 그대로다.
- **dense 대비(참고)는 70% 의 평균 Δp 가 "변함"(+0.0214)** 이다. ft0 와 dense 의 차이(+0.0100) 만큼 fine-tuning 효과가 섞인다.
  기준 선택이 70% Δp 판정을 가른다는 점을 보고서에 적는다 (11-10 결정 1 의 "결론이 같다" 는 S 기준에서만 맞다).
- Δp 는 flip 이 못 보는 곳도 잡는다. 원래 맞은 landbird/water 사진은 opp flip 이 0.0005 인데 Δp 는 +0.11~+0.13 이다.

### 한계

- 재현 전. 실질 크기 기준은 seed 3개로 추정한 표준편차라 불안정하다.
- 80% 합산은 seed 1, 2 둘뿐이다 (seed 0 은 정확도 기준 제외).
- d_wrong 은 원래 틀린 이미지(약 10%)만 쓰므로 CI 가 넓다.
- bootstrap 은 이미지·배경 재표본만 반영한다. 학습 seed 의 변동은 실질 크기 기준으로만 들어간다.

### 다음에 할 일

- [ ] NAS 의 데이터 tar 복구 (노드 로컬 사본에서 다시 묶거나 다시 받기)
- [ ] STEP 4 attribution (P2)
- [ ] STEP 2·3+ 재현

---

## 2026-11-10 — 분석 계획 확정

### 한 줄 요약

STEP 4 전에 H1 판정 방법과 H2 분석 계획을 확정했다. 실험은 하지 않았다. 상세는 `CLAUDE.md` §4 "2026-11-10 결정 사항".

### 왜 바꾸는가

- STEP 3 에서 정확도 유지 구간(≤70%)의 `S` 가 ft0 대비 변하지 않았다. 원래 H2 ("낮은 sparsity 에서는 `S` 를 따라가다
  높은 sparsity 에서 어긋난다") 는 `S` 가 움직인다는 전제라 이 구간에서 검증할 대상이 없다.
- 원래 이미지 단위 정답("반대 배경에서 뒤집히는가")은 원래 맞은 소수 그룹 사진(테스트의 약 41%)에서
  구조적으로 거의 0 이다. 반대 배경이 라벨과 일치하는 배경이라 배경에 의존해도 뒤집힐 수 없다.
- H1 의 ΔS 에 신뢰구간이 없었다.

### 결정 요약

| # | 항목 | 결정 |
|---|---|---|
| 1 | H1 비교 기준 | ft0 대비 (seed 별 짝지음). dense 대비는 함께 보고 |
| 2 | H1 판정 | 짝지은 bootstrap 95% CI 가 0 미포함 **AND** \|Δ\| ≥ dense seed 표준편차 2배 일 때만 "변함". 지표: `S`, `d_correct`, `d_wrong`, 평균 `Δp` |
| 3 | 이미지별 행동 지표 | `Δp` = mean p(waterbird \| 물 배경 5장) − mean p(waterbird \| 땅 배경 5장). 원본 사진 미사용 |
| 4 | H2 | **H2a** 모델 안에서 `Attr_bg` 가 `Δp` 를 추적하는가 (주 지표: 부분 Spearman ρ(Attr_bg, Δp \| area)). **H2b** 행동이 같은 모델끼리 히트맵이 같은가 (ft0 대비 이미지별 변화 vs seed 간 잡음) |
| 5 | STEP 4 대상 | dense, ft0, 30/50/70% × seed 3 = 15개. 80% 는 seed 0 제외로 짝 비교가 깨져 뺀다 |
| 6 | 설명 대상 | 예측 클래스 margin (`z_pred − z_other`). RISE 는 예측 클래스 확률 |
| 7 | 방법 설정 | Grad-CAM `layer4`·ReLU·224 업샘플 (7×7 도 저장) / IG 검정 baseline 50 step, completeness 중앙값 > 5% 면 100 step / RISE 7×7·p=0.5·마스크 2,000개·고정 시드 |
| 8 | 이미지 수 | Grad-CAM·IG 5,794장 / RISE 그룹당 250장 1,000장 |
| 9 | 마스크 | 224 이진(>0.5). 경계 ±3px 띠 제외 `Attr_bg_band` 도 계산 |
| 10 | stress test | ST-A fine-tuning 5 epoch / ST-B 마스크 띠·Grad-CAM 7×7 / ST-C IG blur baseline. 100% 상관 데이터, 구조적 pruning, 85% 는 하지 않고 한계로 |

### 진행 순서

```
P1 확률 저장 + Δp + H1 재판정 (GPU 약 30분)
  ├─ P2 STEP 4 히트맵 (GPU 수 시간)
  │    └─ P3 STEP 5 채점 H2a·H2b (CPU, srun)
  └─ P4 stress test (GPU, P2 와 병렬 가능. QoS 동시 8잡 / 제출 16잡 지킬 것)
```

### 남은 일 (사람)

- [ ] 결정 사항을 팀원에게 공유하고 이의 확인 (특히 H2 재정의)
- [ ] 다른 계정으로 dense seed 0 STEP 2 측정 재현 (`S` ≈ 0.0790 확인)

### 메모

- H1 기준을 ft0 대비로 고른 것은 STEP 3 결과를 본 뒤의 결정이다. 보고서에 두 기준을 모두 싣고 시점을 밝힌다.
- H2b 의 잡음 기준(ft0 의 seed 간 차이)은 서로 다른 dense 모델에서 출발한 비교라, 같은 seed 안의 비교보다
  잡음이 클 수 있다. 그만큼 "설명이 바뀌었다" 판정이 보수적으로 나온다.
- 원래 맞은 소수 그룹 사진의 flip 이 거의 0 인 근거: dense opp flip 0.0005~0.0009 (landbird/water),
  0.003~0.005 (waterbird/land) (2026-11-03 항목).

---

## 2026-11-03 (2) — STEP 3 pruning + fine-tuning

### 한 줄 요약

평균 정확도가 유지되는 구간(sparsity 70% 이하)에서는 **`S` 가 거의 변하지 않았다.**
fine-tuning 만 한 대조군(ft0) 대비 변화는 -0.0035 ~ +0.0007 로, H1 기준(dense seed 표준편차 2배, 0.0069) 이내다.
`S` 가 크게 오르는 것은 정확도가 2pt 넘게 떨어지는 90%, 95% 에서뿐이다. 재현 전 수치다.

### 실행 방법

```bash
cd /data/$USER/xai_project && mkdir -p logs
for s in 0 1 2; do
  sbatch scripts/prune_slurm.sh $s "0 30 50 70"
  sbatch scripts/prune_slurm.sh $s "80 90 95"
done                                   # 잡 6개, 각 15~26분
srun -p debug_ugrad --cpus-per-task=1 --mem=2G -t 0:05:00      /data/$USER/miniconda3/envs/xai/bin/python summarize_pruning.py --out results/pruning_k5.md
```

### 설정

- Conv/Linear 가중치 전역 magnitude one-shot pruning. BatchNorm, bias 는 자르지 않는다.
- 마스크 고정 후 ERM 과 같은 설정(SGD lr 1e-3, momentum 0.9, wd 1e-4, 증강, AMP)으로 20 epoch fine-tuning.
- 모델 선택은 validation 평균 정확도. fine-tuning 한 epoch 중에서만 고른다.
- sparsity 0 도 같은 fine-tuning 을 거친다 (ft0). fine-tuning 자체의 효과를 분리하기 위한 대조군이다.
- 저장된 모델의 실제 sparsity 가 목표와 0.5%p 넘게 다르면 멈춘다. 21개 모두 목표와 일치했다.
- 20 epoch 은 정한 값이고, 길이에 따른 민감도는 보지 않았다.

### 결과 (seed 3개 평균, 전체는 `results/pruning_k5.md`)

| sparsity | avg | WGA | Δavg vs dense | 제외 seed | S | ΔS vs dense | ΔS vs ft0 |
|---|---|---|---|---|---|---|---|
| dense | 90.00 | 66.25 | - | 0/3 | 0.0812 | - | - |
| 0 (ft0) | 89.56 | 67.45 | -0.44 | 0/3 | 0.0859 | +0.0047 | - |
| 30% | 89.64 | 67.39 | -0.36 | 0/3 | 0.0866 | +0.0054 | +0.0007 |
| 50% | 89.83 | 67.39 | -0.17 | 0/3 | 0.0825 | +0.0013 | -0.0035 |
| 70% | 89.30 | 67.39 | -0.70 | 0/3 | 0.0853 | +0.0041 | -0.0006 |
| 80% | 88.19 | 65.84 | -1.81 | 1/3 | 0.0903 | +0.0091 | +0.0044 |
| 90% | 85.93 | 57.01 | -4.07 | 3/3 | 0.1102 | +0.0290 | +0.0243 |
| 95% | 77.21 | 28.61 | -12.79 | 3/3 | 0.2316 | +0.1504 | +0.1457 |

- **fine-tuning 만으로도 `S` 가 +0.0047 움직인다.** dense 와 비교하면 pruning 효과와 섞인다.
- **80% 는 seed 에 따라 갈린다.** seed 0 은 avg 가 2.30pt 떨어져 제외된다. 남은 seed 1, 2 의 ΔS 는
  dense 대비 +0.0137, +0.0020 (평균 +0.0079, 기준 초과), ft0 대비 +0.0081, -0.0033 (평균 +0.0024).
- **90% 이상은 정확도가 무너진 모델이다.** CLAUDE.md 규칙상 메인 분석에서 제외한다.
- fine-tuning 전 one-shot 은 80% 이상에서 avg 77.84%, WGA 0% 다. 테스트셋 landbird 비율(77.84%)과 같아,
  전부 landbird 로 예측하는 상태다. fine-tuning 으로 회복된다.
- 분해 지표에서 `d_correct` 는 0.042(dense) → 0.048(80%) → 0.061(90%) 로 오른다. 원래 맞은 이미지가
  배경에 더 민감해지는 방향이지만, 70% 이하에서는 seed 간 차이 안이다.

### 해석과 한계

- 이 프로토콜에서는 **정확도가 유지되는 압축이 행동 기반 shortcut 의존을 바꾸지 않는다.**
  CLAUDE.md 의 H1 기각 조건("모든 sparsity 에서 S 변화가 seed 표준편차 2배 이내")에 가깝다.
  다만 80% 는 기준에 따라 판정이 갈리고, 기준인 seed 표준편차를 seed 3개로 추정해 불안정하다.
- **H2 의 모델 단위 Spearman 은 이 범위에서 의미가 약하다.** `S` 의 변동이 잡음 수준이라 순위가 안정되지 않는다.
  이미지 단위 AUROC 는 영향을 덜 받는다.
- ΔS 는 seed 평균 차이만 계산했고 신뢰구간은 없다.
- 다른 pruning 방식(구조적 pruning, fine-tuning 없는 압축), 80~90% 사이 세분화는 해보지 않았다.

### 다음에 할 일

- [ ] H1 판정 기준(dense 대비 / ft0 대비) 결정
- [ ] 80~90% 사이(예: 85%) 추가 여부 결정
- [ ] STEP 4 attribution — 대상은 dense, ft0, 30/50/70% (필요하면 80%)
- [ ] STEP 2·3 재현

---

## 2026-11-03 — STEP 2 반사실 생성 + dense 모델 `S` 측정

### 한 줄 요약

dense 모델 3개의 `S` 는 **0.081** (seed 0.079·0.080·0.085, 표준편차 0.0034) 이다. 세 seed 모두 95% 신뢰구간이 0 을
포함하지 않는다. 재현 전이므로 확정 수치가 아니다.

### 실행 방법

```bash
cd /data/$USER/xai_project && mkdir -p logs
sbatch scripts/download_places.sh     # 약 1시간. 서버가 느리다
srun -p debug_ugrad --cpus-per-task=16 --mem=32G -t 2:00:00      bash scripts/generate_cf_slurm.sh k5 --n_bg 5    # 10분 미만, NAS 에 tar 2.2GB
for s in 0 1 2; do sbatch scripts/measure_shortcut_slurm.sh k5 $s; done    # seed 당 약 1~3분
```

### 구성

- 대상: 테스트 이미지 5,794장. 이미지당 `x_opp` 5장, `x_same` 5장, 총 57,940장
- 합성: 공식 `generate_waterbirds.py` 와 같은 방식 (소프트 마스크, `.jpg` 기본 품질)
- 배경: Places365 평가셋 bamboo_forest, forest/broadleaf, ocean, lake/natural 각 100장.
  Waterbirds 는 학습 풀만 썼으므로 원본 배경과 겹치지 않는다.

### 검증

- 원본 정확도가 세 seed 모두 `final_test.json` 과 일치한다 (불일치 시 측정 코드가 멈춘다)
- 새 안쪽 픽셀과 공식 이미지의 차이: 평균 0.62, 최대 2.03 (50장, 0~255). 배경 차이는 평균 75.8
- 합성 결과 5장(Western Gull 2마리)을 눈으로 확인했다. 전체를 확인한 것은 아니다.

### 결과

| seed | 원본 avg | BFR_opp | BFR_same | S | 95% CI (이미지+배경) |
|---|---|---|---|---|---|
| 0 | 90.37 | 0.1674 | 0.0884 | 0.0790 | 0.0668 ~ 0.0905 |
| 1 | 90.04 | 0.1760 | 0.0965 | 0.0795 | 0.0660 ~ 0.0928 |
| 2 | 89.59 | 0.1828 | 0.0977 | 0.0852 | 0.0723 ~ 0.0979 |
| **평균** | | **0.1754** | **0.0942** | **0.0812** | |

원본 예측의 정오답을 나누면 방향이 드러난다. 값은 opp flip 비율의 세 seed 범위다.

| 그룹 | 원본이 맞은 이미지 | 원본이 틀린 이미지 |
|---|---|---|
| landbird/land | 0.126~0.149 | 0.43~0.65 (4~8장) |
| landbird/water | 0.0005~0.0009 | **0.981~0.985** |
| waterbird/land | 0.003~0.005 | **0.890~0.906** |
| waterbird/water | 0.266~0.309 | 0.019~0.036 (28~32장) |

- **소수 그룹의 오답은 배경을 반대로 바꾸면 대부분 정답 쪽으로 바뀐다.** landbird/water 와 waterbird/land
  오답의 89~99% 가 뒤집힌다. 배경 의존을 뒷받침하는 강한 근거다.
- **다수 그룹의 정답은 반대 배경에서 일부 틀려진다.** 같은 타입 배경에서는 flip 이 0.15~3.6% 다.
- landbird/water 의 그룹 `S` 가 음수(-0.03)인 것은 shortcut 이 없어서가 아니다. 이 그룹의 정답 이미지는
  반대 배경에서 거의 뒤집히지 않고, opp flip 은 오답 쪽에 몰려 있다.

### 한계

- **`S` 는 방향을 구분하지 않는다.** 위처럼 오답 교정과 정답 붕괴가 같은 flip 으로 합쳐진다.
  pruning 후 두 값이 반대로 움직이면 `S` 가 그대로여도 행동은 변한 것일 수 있다.
  STEP 5 전에 방향별 값을 함께 볼지 결정이 필요하다.
- **배경 풀이 타입별 200장이다.** 재표본 시 CI 폭이 약 1.5배가 됐다 (seed 0 기준 0.016 -> 0.024).
- **평가셋과 학습 풀의 배경 분포 차이는 측정하지 않았다.** `BFR_same` 에 섞여 있을 수 있다.
- H1 기각 조건인 seed 표준편차 2배는 약 0.0069 다. seed 3개로 추정한 값이라 불안정할 수 있다.
- CUB 마스크에는 최댓값이 0.8 인 이미지(표본 300장 중 8장)와 아주 작은 경우가 있다. `Attr_bg` 계산 때 확인할 것.

### 다음에 할 일

- [ ] STEP 2 결과 재현
- [ ] 방향별 flip 을 보조 지표로 쓸지 결정
- [ ] STEP 3 pruning — 입력은 `runs/erm_seed{0,1,2}/best.pt`

---

## 2026-10-28 — 클러스터 환경 구축 + STEP 0·1 1회 실행

(2026-11-03 재검증 후 오류 정정)

대상 클러스터: Seraph `aurora`

### 한 줄 요약

aurora 에서 환경을 세우고 STEP 0(데이터 점검)과 STEP 1(dense ERM)을 한 번 실행했다.
평균 90.0% / WGA 66.3% / 격차 23.75pt. **seed 0·1 통과, seed 2 는 평균 89.59% 로 확인 필요.**
재현 전이므로 확정 수치가 아니다. 체크포인트는 git 에 없고
클러스터 `/data/jwook2480/xai_project/runs/erm_seed{0,1,2}/best.pt` 에 있다.

---

### 1. 클러스터 환경 (실측값)

| 항목 | 값 |
|---|---|
| 접속 | `aurora.khu.ac.kr:30080`, 공개키 인증 |
| 스케줄러 | SLURM. 학부 파티션 `batch_ugrad`(무제한) / `debug_ugrad`(4시간) |
| 노드 | `aurora-g1~g8`. 노드당 CPU 64코어(g1은 96) / RAM 약 420GB / GPU 8장(g7만 6장) |
| GPU | **RTX 3090 24GB**, 드라이버 `535.161.07` (CUDA 12.2), nvcc 12.3 |
| QoS 한도 | 동시 실행 8잡 / 제출 16잡 / GPU 8장 / 최대 6일 |
| 마스터 노드 | `aurora-master`. **GPU 없음** (`nvidia-smi` 명령 자체가 없다) |
| 시스템 python | 3.8.10 — `requirements.txt` 설치 불가 |
| conda | `/data/jwook2480/miniconda3` (기존 설치분) |
| `/home` | **계산 노드와 공유 안 됨** (`aurora-g7` 에서 확인). 90% 사용 중 |
| `/data` (NAS) | 109T 중 31T 여유. 코드·환경·tar 를 여기 둔다 |
| `/local_datasets` | 계산 노드 로컬 NVMe. 학습 데이터는 여기서 읽는다 |
| 공용 Waterbirds/CUB | **없음.** 직접 받아야 한다 |

`sinfo` 결과 요약:

```
PARTITION     AVAIL  TIMELIMIT  NODES  NODELIST
debug_ugrad      up    4:00:00      8  aurora-g[1-8]
batch_ugrad      up   infinite      8  aurora-g[1-8]
```

---

### 2. 막혔던 지점과 해결 (전부 `CLAUDE.md` §5 에 반영)

겪지 않아도 될 문제를 다시 겪지 않도록 전부 적어둔다.

#### (1) ★ `CUB_200_2011.tgz` 에 마스크가 없다

가장 크게 막힌 곳. `check_data.py` 가 마스크 200장을 전부 "파일 없음"으로 보고했다.
원인은 `CUB_200_2011.tgz` 안에 `segmentations/` 디렉터리가 **아예 없다**는 것.

Caltech 의 **별도 레코드**에 따로 올라와 있다:

```
https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz   (38MB)
```

원래 CUB 레코드 `65de6-vp158` 에는 `CUB_200_2011.tgz` 하나뿐이다.
예전 호스트였던 `vision.caltech.edu/visipedia-data/...` 는 지금 404 다.

압축 최상위가 `segmentations/` 라서 `CUB_200_2011/` 안에 풀어야 경로가 맞는다.
→ `scripts/download_data.sh`, `scripts/stage_data.sh` 에 반영 완료.

**이걸 모르면 STEP 4 의 `Attr_bg` 도 STEP 2 의 반사실 합성도 불가능하다.**

#### (2) `sbatch` 에서 `dirname $0` 이 깨진다

제출한 스크립트는 `/var/spool/slurm/d/jobNNN/slurm_script` 로 **복사되어** 실행된다.
따라서 `source "$(dirname "$0")/_common.sh"` 는 잡이 시작하자마자 죽는다.
첫 제출 3개가 전부 이걸로 즉사했다.
→ `$SLURM_SUBMIT_DIR` 또는 절대경로를 쓴다.
`srun` 으로 돌릴 때는 멀쩡해서 더 헷갈린다.

#### (3) `/home` 이 계산 노드와 공유되지 않는다

로그인 노드에서 만든 파일을 계산 노드가 못 읽는다. 처음에 홈 디렉터리에 테스트
스크립트를 두고 `srun` 했다가 `No such file or directory` 로 실패했다.
`aurora-g7` 에서 접근을 확인한 공유 경로는 `/data`(NFS)와 `/mnt/ceph_jfs`(JuiceFS)다.
→ 저장소·출력·캐시 전부 `/data/$USER/` 아래로.

#### (4) 사전학습 가중치가 `/home` 으로 떨어진다

torchvision 은 ResNet-50 가중치를 `~/.cache/torch` 에 받는다.
`/home` 은 공유도 안 되고 90% 차 있어서, 노드가 바뀔 때마다 98MB 를 다시 받게 된다.
→ `scripts/_common.sh` 에서 `TORCH_HOME=/data/$USER/.torch` 로 돌렸다.

#### (5) conda 기본 채널이 ToS 동의를 요구한다

`conda create` 가 `CondaToSNonInteractiveError` 로 멈춘다 (`repo.anaconda.com` 채널).
동의 절차 없이 진행할 수 있도록 **conda-forge 만 쓰는 쪽**을 택했다.

```bash
conda create -y -n xai --override-channels -c conda-forge python=3.11
```

#### (6) torch CUDA 빌드를 잘못 고르면 GPU 를 못 잡는다

기존 base 환경(torch 2.14.1+**cu130**)은 GPU 노드에서도
`CUDA driver is too old (found version 12020)` 로 `cuda.is_available()` 이 `False` 였다.
드라이버가 535.161.07 (CUDA 12.2) 이기 때문.
→ **cu121 휠로 고정.** `torch==2.5.1+cu121` / `torchvision==0.20.1+cu121` 로 `True` 확인.
다른 CUDA 빌드는 테스트하지 않았다.

#### (7) 시스템 python 3.8 로는 requirements 를 못 깐다

`scipy>=1.11` 은 py3.8 휠이 없다. 3.8 에서 받을 수 있는 최대가 1.10.1 이고,
torch 도 2.4.1 까지다.
→ conda 로 python 3.11 환경을 따로 만들었다.

#### (8) Waterbirds 다운로드 URL

`nlp.stanford.edu` 는 응답이 없다(타임아웃). `downloads.cs.stanford.edu` 미러는 정상.
→ 미러를 먼저 시도하도록 순서를 바꿨다.

#### (9) 동시 스테이징 경쟁 (2026-11-03 발견·수정)

처음 만든 `stage_data.sh` 는 잠금 없이 파일 존재만 확인했다. 새 노드에 잡이 동시에 걸리면
압축 해제 도중인 데이터로 학습이 시작될 수 있었다. 10-28 학습은 seed 0·1 이 이미 스테이징된
g7, seed 2 가 g5 단독이어서 영향이 없었다.
→ `flock` 잠금과 완료 표식을 추가했다. 동시 2개 실행으로 이미지·마스크 각 11,788장을 확인했다.

---

### 3. Seraph 규칙 대응

튜토리얼 규칙(위반 시 **1학기 사용 정지**)에 맞춰 구조를 바꿨다.

| 규칙 | 어떻게 지켰는가 |
|---|---|
| dataloader 가 NAS 직접 읽기 금지 | `stage_data.sh` 가 tar 를 `/local_datasets/$USER/xai` 로 펼친다 |
| 마스터 노드에서 python 실행 금지 | 모든 스크립트가 `srun`/`sbatch` 전용. conda/pip 도 계산 노드에서 |
| 마스터 노드에서 압축 해제 금지 | `download_data.sh` 는 받기만 하고 풀지 않는다 |
| 마스터 노드에서 AI 에이전트 금지 | **Claude Code 는 로컬 PC 에서 실행**하고 `ssh` 로 명령만 보낸다 |
| `/home` 에 코드·환경 금지 | 전부 `/data/$USER/xai_project` |
| Remote-SSH 로 큰 디렉터리 열기 금지 | `.vscode/settings.json` 에 watcher 제외 설정 |

> **규칙 위반 기록.** 규칙 문서를 받기 전, 환경 확인 중 마스터 노드에서 다음을 실행했다.
> `find`(공용 데이터셋 검색), `/data/jwook2480` 재귀 `du`(도중에 Killed), `pip` dry-run(torch 휠 800MB 수신),
> `python` 의 torch import, `conda env list`. 규칙을 받은 뒤로는 python 과 재귀 탐색을
> 마스터에서 실행하지 않았다. 같은 확인은 `srun` 안에서 하거나 `ls | grep` 으로 한다.

---

### 4. STEP 0 — 데이터 점검 (통과)

```bash
srun -p debug_ugrad --cpus-per-task=8 --mem=16G -t 1:00:00 bash scripts/check_data_slurm.sh
```

```
총 11,788장   train 4,795 / val 1,199 / test 5,794
train 그룹 분포  g0 73.0% | g1 3.8% | g2 1.2% | g3 22.0%
학습셋 새-배경 일치 비율  95.0%          <- shortcut 존재 확인
마스크 파일·크기 검사 200/200 통과,  파일 없음 0,  크기 불일치 0
새 영역 평균 면적 0.133 (중앙값 0.123)
  -> 균일 attribution 의 Attr_bg 기댓값 = 0.867   ★이후 기준선
```

`data_check/` 의 overlay 6장 중 Caspian Tern, European Goldfinch **2장만** 눈으로 확인했다.
두 장 모두 새 윤곽이 밝게 남고 배경이 어두웠다. **나머지 4장은 아직 확인 전이다.**

> **`Attr_bg` 기준선 0.867 을 기억할 것.** attribution 이 아무 정보도 없는
> 균일 맵이어도 배경 비율이 0.867 로 나온다. 실제 `Attr_bg` 가 이 값보다
> **낮아야** "모델이 새를 본다"고 말할 수 있다.

---

### 5. STEP 1 — dense ERM 학습 (seed 0·1 통과, seed 2 확인 필요)

```bash
mkdir -p logs
for s in 0 1 2; do sbatch scripts/train_slurm.sh $s; done
```

ResNet-50 (ImageNet 사전학습) / SGD lr 1e-3, momentum 0.9, wd 1e-4 / batch 128 /
100 epoch / AMP. 모델 선택은 **validation 평균 정확도** 기준.

`best.pt` 의 테스트 성능:

| seed | avg | adj | WGA | 평균-WGA 격차 | best epoch |
|---|---|---|---|---|---|
| 0 | 90.37 | 97.86 | 64.49 | 25.9 | 60 |
| 1 | 90.04 | 97.84 | 70.72 | 19.3 | 75 |
| 2 | 89.59 | 97.78 | 63.55 | 26.0 | 90 |
| **평균** | **90.00** ± 0.39 | **97.83** ± 0.04 | **66.25** ± 3.90 | **23.75** ± 3.83 | |

그룹별 정확도 (seed 0 기준):

| 그룹 | 정확도 |
|---|---|
| g0 landbird / land (다수) | 99.8 |
| g1 landbird / water (소수) | 87.0 |
| g2 waterbird / land (소수) | **64.5** ← 최저 |
| g3 waterbird / water (다수) | 95.0 |

세 seed 모두 최저 그룹이 **g2 (waterbird on land)** 로 일치한다.
학습셋에 56장(1.2%)뿐인 최소 그룹이다.
`landbird/land` 99.8% 대 `waterbird/land` 64.5% 의 차이는 배경 shortcut 의존을 시사한다.
행동 지수 `S` 는 STEP 2 에서 측정하므로 아직 직접 확인한 것은 아니다.

소요 시간: 1 GPU 기준 100 epoch 에 약 14분. seed 3개 병렬로 **15분**.

- `±` 는 seed 3개의 표본 표준편차다. `CLAUDE.md` §8 이 요구하는 bootstrap 신뢰구간은 아직 계산하지 않았다.
- 다른 사람의 재현 전이므로 보고서용 확정 수치가 아니다.

> seed 2 는 avg 89.59 로 README 와 스크립트의 평균 90% 기준에 0.41pt 못 미쳐
> `[확인 필요] 평균 정확도가 낮음` 이 출력됐다. 격차 26.0pt 는 15pt 기준을 넘는다.

격차 기준(15pt 이상)은 세 seed 모두 넘었다. 평균 기준까지 포함한 통과는 seed 0·1 이다.

---

### 6. 바뀐 파일

**신규**

| 파일 | 역할 |
|---|---|
| `SERAPH_RULES.md` | Seraph 사용 규칙 요약 — 먼저 읽을 것 |
| `WORKLOG.md` | 이 파일 |
| `scripts/_common.sh` | 공통 경로·캐시 설정. 모든 잡 스크립트가 source |
| `scripts/setup_env.sh` | conda 환경 `xai` 생성 |
| `scripts/stage_data.sh` | NAS tar → `/local_datasets` 펼치기 |
| `scripts/check_data_slurm.sh` | STEP 0 실행 래퍼 |
| `scripts/smoke_test.sh` | 파이프라인 점검 (subset 128, 2 epoch) |
| `.vscode/settings.json` | Remote-SSH I/O 억제 |
| `.gitattributes` | 줄바꿈을 LF 로 고정 |

**재작성**: `SETUP.md`(실측값), `scripts/download_data.sh`(tar 3종·미러 순서·무결성 검사),
`scripts/train_slurm.sh`(`batch_ugrad`), `scripts/train_direct.sh`

**갱신**: `CLAUDE.md`(§1 실행 환경, §3 상태표, §5 함정 12개, §6 구조), `README.md`, `.gitignore`

**파이썬 코드(`train_erm.py`, `waterbirds.py`, `check_data.py`)는 한 줄도 고치지 않았다.**
전부 환경·실행 스크립트 쪽 문제였다.

---

### 7. 다른 계정에서 똑같이 돌리는 법

본인 계정에서 처음부터 재현하는 절차. `$USER` 는 각자 계정으로 자동 치환된다.

먼저 준비할 것 (아래 명령만으로는 해결되지 않는다)

- **배정 클러스터·포트·파티션:** 본인 포털과 `show-assoc` 로 확인한다. `aurora` 나 `*_ugrad` 가
  아니면 명령과 스크립트의 호스트·파티션을 바꾼다.
- **conda:** `/data/$USER/miniconda3` 에 직접 설치한다. `setup_env.sh` 는 설치된 것을 전제로 한다.
- **GitHub:** 협업자 초대를 수락하고, 클러스터에서 비공개 저장소를 받을 수 있게 인증을 설정한다.

**아직 다른 계정에서 이 절차를 끝까지 실행해 본 적은 없다.**

```bash
# 0) 로컬 PC 에서 접속 설정 (~/.ssh/config)
#    Host aurora
#        HostName aurora.khu.ac.kr
#        Port <본인 포털 기준, 이 계정은 30080>
#        User <본인계정>
#        IdentityFile ~/.ssh/id_ed25519

# 1) 저장소를 NAS 에 둔다 (홈이 아니라 /data 다)
cd /data/$USER && git clone https://github.com/jiwooks/xai-compression-attribution.git xai_project && cd xai_project

# 2) conda 환경 (계산 노드에서. 약 5분)
srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 1:00:00 \
     bash scripts/setup_env.sh

# 3) 데이터 (약 5분, 1.6GB)
srun -p debug_ugrad --cpus-per-task=4 --mem=8G -t 2:00:00 bash scripts/download_data.sh

# 4) STEP 0 점검
srun -p debug_ugrad --cpus-per-task=8 --mem=16G -t 1:00:00 bash scripts/check_data_slurm.sh
#    -> data_check/ 의 overlay 를 눈으로 확인할 것

# 5) 파이프라인 점검 (1~2분)
srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 0:30:00 \
     bash scripts/smoke_test.sh

# 6) 본 학습 (15분)
mkdir -p logs && for s in 0 1 2; do sbatch scripts/train_slurm.sh $s; done
squeue -u $USER
```

주의할 점

- **마스터 노드에서 python 을 직접 실행하지 말 것.** 전부 `srun`/`sbatch` 로.
- `/local_datasets` 는 **노드마다 따로**다. 다른 노드에 걸리면 `stage_data.sh` 가
  알아서 다시 펼친다. 정상 동작이다.
- 잡 취소는 `scancel <JOBID>`.

---

### 8. 다음에 할 일

- [ ] `data_check/` overlay 나머지 4장을 눈으로 확인
- [x] 비공개 GitHub 저장소 생성 및 푸시
- [ ] 협업자 초대 및 STEP 1 재현
- [ ] seed 2(avg 89.59)를 그대로 쓸지 결정, 평균 기준을 `avg` 로 유지할지 검토
- [ ] STEP 1 bootstrap 신뢰구간 계산
- [ ] **STEP 2 — 반사실 이미지 생성**. 이 프로젝트의 핵심 장치이고 난이도가 가장 높다.
      Places 배경(`bamboo_forest`, `forest/broadleaf` / `ocean`, `lake/natural`)을
      카테고리 단위로 받는 방법을 먼저 확인할 것. 전체는 용량이 너무 크다.
- [ ] STEP 3 pruning — 입력은 `runs/erm_seed{0,1,2}/best.pt`
