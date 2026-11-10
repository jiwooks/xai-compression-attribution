# CLAUDE.md

이 파일은 Claude Code 가 이 저장소에서 작업할 때 읽는 프로젝트 컨텍스트다.

---

## 1. 프로젝트 개요

경희대 AI3008 "설명 및 신뢰가능한 AI" 수업 팀 프로젝트 (Team 5).

**Research Question**
> 압축으로 인해 모델의 spurious feature 의존도가 변할 때,
> feature attribution 방법은 그 변화를 정확히 추적하는가?

**핵심 아이디어**
Attribution map 은 별도 도구가 아니라 *압축된 모델 자신의* gradient/activation/출력으로
계산된다. 따라서 압축 후 saliency 가 배경 쪽으로 이동해도
(a) 모델이 실제로 배경을 더 쓰게 된 것인지,
(b) 설명이 압축 모델에서 부정확해진 것인지 구분할 수 없다.

그래서 **attribution 과 무관한 정답**을 따로 만든다:
같은 새를 다른 배경에 합성한 **반사실(counterfactual) 이미지**에서 예측이 뒤집히면,
바뀐 것은 배경뿐이므로 모델이 배경을 사용했음을 의미한다.
이 행동 기반 정답에 attribution 이 얼마나 맞는지 채점한다.

**가설**
- **H1 (모델)**: 평균 정확도가 유지되는 압축 구간에서도 shortcut 지수 `S` 가 sparsity 에 따라 변한다.
  선행연구가 양방향을 시사하므로 방향은 고정하지 않는다.
  *기각 조건*: 모든 sparsity 에서 `S` 변화가 seed 표준편차 2배 이내.
- **H2 (설명)**: attribution 은 낮은 sparsity 에서는 `S` 를 따라가지만 높은 sparsity 에서 어긋난다.
  *기각 조건*: 최대 sparsity 의 tracking AUROC 가 dense 모델의 95% bootstrap CI 안에 있음.
- 2026-11-10 에 H1 판정 방법을 구체화하고 H2 를 H2a/H2b 로 나눴다. **현재 기준은 §4 "2026-11-10 결정 사항"** 이다.
  위 원래 문구는 제안서 기록용으로 남긴다.

**일정**
- 11월 말: 최종 발표
- 12/18: 최종 보고서

**실행 환경** (2026-10-28 실측, 상세는 `SETUP.md` / 규칙은 `SERAPH_RULES.md`)
- 경희대 Seraph `aurora` 클러스터 (`aurora.khu.ac.kr:30080`, 계정 `jwook2480`, account `ugrad`)
- SLURM. 학부 파티션 `batch_ugrad`(무제한) / `debug_ugrad`(4시간).
  QoS 한도: 동시 8잡 / 제출 16잡 / GPU 8장 / 최대 6일
- GPU: RTX 3090 24GB, 드라이버 535.161.07 (CUDA 12.2) → **torch cu121 휠 사용** (cu130 은 동작 안 함, 그 외 미검증)
- 환경: `/data/jwook2480/miniconda3/envs/xai` (py3.11 + torch 2.5.1+cu121).
  잡에서는 `conda activate` 대신 절대경로 python 을 부른다.
- **`/home` 은 계산 노드와 공유되지 않는다.** 코드·출력 모두 `/data/$USER/xai_project`.
- **데이터는 NAS 에서 직접 읽지 않는다.** tar 는 `/data/$USER/datasets/xai/`,
  잡 시작 시 `scripts/stage_data.sh` 가 `/local_datasets/$USER/xai/` 로 펼친다.
- **마스터 노드에서 python 실행·압축 해제·재귀 탐색·AI 에이전트 금지.**
  Claude Code 는 로컬 PC 에서 띄우고 `ssh aurora ...` 로 명령을 보낸다.

---

## 2. 지표 정의 (코드에서 이 정의를 반드시 지킬 것)

**그룹** `group = y * 2 + place`
| g | 의미 | 비중 |
|---|---|---|
| 0 | landbird / land | 다수 |
| 1 | landbird / water | 소수 |
| 2 | waterbird / land | 소수 |
| 3 | waterbird / water | 다수 |

**행동 기반 shortcut 지수 (attribution 미사용)**
```
S = BFR_opp - BFR_same
```
- `BFR_opp` : 새를 **반대 타입** 배경(물↔숲)에 합성했을 때 예측이 뒤집힌 비율
- `BFR_same`: 새를 **같은 타입의 다른** 배경에 합성했을 때 뒤집힌 비율 (대조군)
- `BFR_same` 을 빼는 이유: "배경이 바뀌기만 하면 흔들리는" 일반적 불안정성을 제거하고,
  배경의 *종류*에 의존하는 정도만 남기기 위함.

**설명이 주장하는 배경 의존도**
```
Attr_bg(x) = sum_{i not in M} |s_i| / sum_i |s_i|
```
`M` = 새 영역 마스크. Zhou et al. (AAAI 2022) 의 `1 - Attr%(M)` 에 해당.

- `Attr_bg_band`: 마스크 경계 ±3px 띠를 분자·분모 모두에서 뺀 값 (경계 번짐 대응)
- 마스크는 224 해상도에서 >0.5 로 이진화한다.
- 집중도 `C = (1 − Attr_bg) / area` (`area` = 새 면적 비율). 균일 맵이면 1.

**이미지별 행동 지표 Δp (attribution 미사용, 2026-11-10 추가)**
```
Δp(x) = mean p(waterbird | 물 배경 반사실 5장) − mean p(waterbird | 땅 배경 반사실 5장)
```
- `p` 는 softmax 의 waterbird 확률. 원본 사진은 쓰지 않는다.
- `place=water` 이미지는 `x_same` 이 물 배경, `x_opp` 가 땅 배경이다. `place=land` 는 반대다.
- 배경 타입이 예측을 얼마나 미는지를 이미지마다 연속값으로 준다. flip 과 달리 원래 정오답과 무관하게 정의된다.

**H1 지표**: `S`, `d_correct`, `d_wrong`, 평균 `Δp`. ft0 대비 seed 별 짝지은 차이 + 짝지은 bootstrap CI (§4 2026-11-10).

**추적성 (RQ 의 직접적 답)**
- **H2a (모델 안 추적)**: 주 지표 = 새 면적을 통제한 **부분 Spearman ρ(Attr_bg, Δp | area)**.
  `rank(Attr_bg)` 와 `rank(Δp)` 를 각각 `rank(area)` 에 선형회귀한 잔차끼리의 Pearson.
  보조 = 원시 Spearman ρ(Attr_bg, Δp), AUROC(정답 = `Δp ≥ 0.10`, 점수 = `Attr_bg`), `C` 로 계산한 같은 지표.
  `area` 는 `Attr_bg` 와 `Δp` 양쪽에 영향을 주는 교란변수라 통제한다.
- **H2b (행동이 같은 모델 간 설명 비교)**: 이미지별 `|Attr_bg(sparsity, seed s) − Attr_bg(ft0, seed s)|` 를
  잡음 기준 `|Attr_bg(ft0, seed a) − Attr_bg(ft0, seed b)|` 와 짝지은 bootstrap 으로 비교한다.
- *모델 단위* `mean(Attr_bg)` 와 `S` 의 Spearman 은 참고용으로만 낸다. 정확도 유지 구간에서 `S` 변동이
  잡음 수준이라 순위가 안정되지 않는다 (WORKLOG 2026-11-03 (2)).
- ~~이미지 단위: `Attr_bg(x)` 로 "x 가 반대 배경에서 뒤집히는가" 를 예측한 AUROC~~ — **2026-11-10 주 지표에서 내림.**
  원래 맞은 소수 그룹 사진(테스트의 약 41%)은 반대 배경이 곧 라벨과 일치하는 배경이라, 배경에 의존하더라도
  예측이 뒤집힐 수 없다 (dense opp flip 0.05~0.5%). 이 사진들의 정답이 모두 "안 뒤집힘" 으로 찍혀 정답표가 틀린다.

---

## 3. 파이프라인 단계와 현재 상태

| STEP | 내용 | 상태 |
|---|---|---|
| 0 | 데이터 점검 + 마스크 정렬 검증 | **자동 점검 통과 (2026-10-28)** — 마스크 파일·크기 200/200, 일치율 95.0%. overlay 는 6장 중 2장만 육안 확인 |
| 1 | dense ERM 기준 모델 학습 | **1회 실행, 재현 전 (2026-10-28)** — seed 0·1 통과, seed 2 는 avg 89.59 로 확인 필요. avg 90.00±0.39 / WGA 66.25±3.90 / 격차 23.75pt |
| 2 | 반사실 이미지 생성 + `S` 측정 | **dense 1회 측정, 재현 전 (2026-11-03)** — S 0.081 (seed 0.079·0.080·0.085), 배경 400장 풀. `WORKLOG.md` 참고 |
| 3 | pruning 파이프라인 | **1회 실행, 재현 전 (2026-11-03)** — sparsity 0~95% × seed 3. 정확도 유지 구간(≤70%)에서 S 변화는 H1 기준 이내. `results/pruning_k5.md` |
| 3+ | 반사실 확률 저장 + `Δp` + H1 재판정 (ft0 대비, 짝지은 bootstrap) | **1회 실행, 재현 전 (2026-11-10)** — 24개 모델 회귀 확인 통과. ft0 대비 ≤70% 는 네 지표 모두 "변함 없음". 평균 Δp 는 70% +0.011 (CI 0 미포함, 기준 0.0135 미달), 80% +0.023 (변함). `results/h1_k5.md` |
| 4 | attribution — 모델 15개 (dense, ft0, 30/50/70% × seed 3). Grad-CAM·IG 는 테스트 5,794장, RISE 는 1,000장. `Attr_bg`, `Attr_bg_band`, `C`, Grad-CAM 7×7 | **계획 확정 (2026-11-10)**, 미착수 |
| 5 | 추적성 분석 — H2a (부분 Spearman ρ(Attr_bg, Δp \| area)), H2b (ft0 대비 이미지별 변화 vs seed 간 잡음) | **계획 확정 (2026-11-10)**, 미착수 |
| 6 | stress test — ST-A fine-tuning 5 epoch, ST-B 마스크 띠·Grad-CAM 7×7, ST-C IG blur baseline | **계획 확정 (2026-11-10)**, 미착수 |

STEP 1 과 STEP 2 는 병렬로 진행 가능하다. STEP 2 가 이 프로젝트의 핵심 장치이자
구현 난이도가 가장 높으므로 일찍 시작할 것.

---

## 4. 반드시 지킬 설계 원칙

### STEP 1 의 통과 기준은 "좋은 모델"이 아니다
평균 정확도는 높고 **WGA 는 낮아야** 한다 (격차 15pt 이상 목표).
격차가 없으면 압축으로 변할 shortcut 의존도 자체가 없어 H1 을 검증할 수 없다.
격차가 작으면 `--no_augment` 로 재학습하거나 상관 100% 변형 데이터를 고려한다.

### 모델 선택은 validation "평균" 정확도 기준
WGA 기준으로 고르면 shortcut 이 약한 체크포인트가 뽑혀 실험 전제가 무너진다.
ERM 표준 프로토콜대로 평균 기준으로 고른다.

### 평균 정확도는 두 가지를 모두 보고
- `avg_acc` : 테스트셋 전체 평균 (모든 이미지 동일 가중)
- `adjusted_acc` : 학습셋 그룹 비율로 가중한 평균 (Sagawa et al. 방식)

논문마다 어느 쪽을 쓰는지 달라 숫자가 크게 차이난다. 보고서에 어느 쪽인지 명시할 것.

### 압축 모델 분석 대상 제한
평균 정확도가 dense 대비 2pt 넘게 떨어진 체크포인트는 메인 분석에서 제외한다.
그냥 망가진 모델은 "판단 근거가 바뀌었다"고 말할 수 없다.

### 2026-11-03 결정 사항
- **seed 2 (avg 89.59) 는 재학습 없이 사용한다.** 보고서에 평균 90% 기준 미달을 명시한다.
- **`S` 정의는 그대로 두고 방향별 지표를 보조로 함께 보고한다.**
  `S = S_C→W + S_W→C` (정답 붕괴 + 오답 교정), `d_correct`, `d_wrong`
  (원래 맞은/틀린 이미지 한 장당 opp 뒤집힘 − same 뒤집힘). 계산은 `summarize_pruning.py`.
- **pruning 프로토콜:** Conv/Linear 가중치 전역 magnitude one-shot pruning 후 마스크를 고정하고
  ERM 과 같은 설정으로 20 epoch fine-tuning. 모델 선택은 validation 평균 정확도.
  sparsity 0 도 같은 fine-tuning 을 거쳐 대조군(ft0)으로 쓴다.
- ~~**미결정:** H1 의 `S` 변화를 dense 대비로 볼지 ft0 대비로 볼지.~~ → 2026-11-10 ft0 대비로 확정.
  결과를 본 뒤에 내린 결정이므로 보고서에는 두 기준을 모두 싣고, 결정 시점을 밝힌다.

### 2026-11-10 결정 사항 (STEP 4 전 분석 계획 확정)

STEP 3 에서 정확도 유지 구간(≤70%)의 `S` 가 변하지 않아 H2 분석 계획을 바꾼다. **보고서에 "STEP 3 결과에 따른
분석 계획 변경" 으로 명시한다.**

1. **H1 비교 기준은 ft0 대비 (seed 별로 짝지음).** dense 대비는 함께 보고만 한다.
   이유: fine-tuning 만으로 `S` 가 +0.0047 움직여, dense 대비는 자르기 효과와 fine-tuning 효과가 섞인다.
   70% 이하에서는 어느 기준이든 결론이 같아 기준 선택이 결론을 고르지 않는다.
2. **H1 판정.** 제안서 기준(dense `S` 의 seed 표준편차 2배, 0.0069)은 **실질 크기 기준**으로 두고,
   **짝지은 bootstrap 95% CI** 를 추가한다. 이미지와 배경을 두 모델에 같은 인덱스로 재표본한다
   (모든 모델이 같은 반사실 사진을 쓰므로 짝지은 비교가 된다).
   **CI 가 0 을 포함하지 않고 |Δ| 가 실질 크기 기준 이상일 때만 "변했다"** 고 판정한다.
   지표는 `S`, `d_correct`, `d_wrong`, 평균 `Δp`. `S` 는 정확도에 끌려가므로 단독으로 쓰지 않는다.
   `d_correct`·`d_wrong`·`Δp` 의 실질 크기 기준도 같은 규칙(dense seed 표준편차 2배)으로 정한다.
3. **Δp 정의.** 이미지마다 `Δp = mean p(waterbird | 물 배경 반사실 5장) − mean p(waterbird | 땅 배경 반사실 5장)`.
   원본 사진은 쓰지 않는다. `place=water` 이미지는 same 이 물 배경, opp 가 땅 배경이고 `place=land` 는 반대다.
4. **H2 를 두 갈래로 나눈다.**
   - **H2a**: 각 모델 안에서 `Attr_bg` 가 `Δp` 를 추적하는가, 압축하면 추적이 약해지는가.
     주 지표 = 새 면적을 통제한 부분 Spearman ρ(Attr_bg, Δp | area).
     보조 = 원시 Spearman, AUROC(정답 = `Δp ≥ 0.10`).
     **판정: 70% 의 주 지표가 ft0 의 95% bootstrap CI 밖이고 seed 3개 중 2개 이상이 같은 방향이면 지지.**
   - **H2b**: 행동이 같은 모델끼리 히트맵이 같은가.
     같은 seed 의 (70% − ft0) 이미지별 `Attr_bg` 차이가, 같은 sparsity 의 seed 간 차이(ft0 seed a vs seed b)보다
     큰지 짝지은 bootstrap 으로 본다. 크면 **"행동은 같은데 설명이 바뀌었다"**.
     해당 sparsity 의 H1 판정이 "변함" 이면 H2b 해석에서 뺀다.
   - 기존 §2 의 "반대 배경 flip 을 예측하는 AUROC" 는 **주 지표에서 내린다.** 원래 맞은 소수 그룹 사진
     (테스트의 약 41%)은 반대 배경이 라벨과 일치하는 배경이라 배경에 의존해도 flip 이 구조적으로 거의 0 이다
     (dense 0.05~0.5%). 이 사진들이 모두 "배경 안 씀" 으로 찍혀 정답표가 틀린다. `Δp` 는 이 문제가 없다.
5. **STEP 4 대상 모델: dense, ft0(`prune_sp00`), `prune_sp30`, `sp50`, `sp70` × seed 0,1,2 = 15개.**
   80% 는 seed 0 이 정확도 기준으로 제외돼 짝지은 비교가 깨지므로 넣지 않는다.
6. **설명 대상은 원본 사진의 예측 클래스 margin (`z_pred − z_other`).** RISE 는 예측 클래스 확률.
   이진 분류에서 어느 클래스를 설명하느냐에 따른 비대칭을 없앤다.
7. **방법별 설정.**
   - Grad-CAM: `model.layer4` 출력, ReLU, 224 로 bilinear 업샘플한 맵을 주로 쓴다 (§5 함정 8 고정).
     7×7 에서 마스크를 면적 평균으로 줄인 버전도 함께 저장한다 (stress test 용).
   - IG: 검정 이미지(픽셀 0 을 정규화한 값) baseline, 50 step. completeness 상대오차 중앙값이 5% 를 넘으면 100 step.
   - RISE: 7×7 격자, p=0.5, 마스크 2,000개, 고정 시드로 모든 모델이 같은 마스크를 쓴다.
8. **이미지.** Grad-CAM·IG 는 테스트 5,794장 전부. RISE 는 그룹당 250장, 고정 시드로 뽑은 1,000장
   (비용: 약 9 GPU-시간, 8장 병렬로 약 1.5시간).
9. **마스크.** 224 해상도 이진 마스크(>0.5). 경계 ±3px 띠를 분자·분모에서 모두 뺀 `Attr_bg_band` 도 계산한다.
10. **stress test 는 세 가지만 한다.**
    - ST-A: fine-tuning 5 epoch (재학습이 shortcut 변화를 되돌리는지 직접 확인)
    - ST-B: 마스크 경계 띠 제외 / Grad-CAM 7×7 (재계산만)
    - ST-C: IG blur baseline
    100% 상관 데이터, 구조적 pruning, 85% 는 하지 않고 보고서 한계에 적는다.

### 양자화는 메인에서 제외
PyTorch 의 실제 INT8 연산은 autograd 가 붙지 않아 gradient 기반 attribution 계산이 막힌다.
필요하면 fake-quant 로 우회하되, 부록 수준으로만 다룬다. **pruning 이 메인.**

---

## 5. 알려진 함정

1. **CUB segmentations 는 별도 다운로드다** ★2026-10-28 확인
   `CUB_200_2011.tgz` 안에 `segmentations/` 가 **없다**. Caltech 의 다른 레코드
   (`w9d68-gec53`, "CUB-200-2011 Segmentations", 38MB)에서 따로 받아야 한다.
   모르고 넘어가면 `check_data.py` 가 "파일 없음 200" 을 낸다.
   `scripts/download_data.sh` 에 이미 반영되어 있다.

2. **CUB 마스크 정렬**
   Waterbirds 공식 생성 스크립트는 배경을 *새 이미지 크기*에 맞춰 잘라 붙이고
   CUB 와 같은 상대경로로 저장한다. 따라서 `CUB_200_2011/segmentations/<같은 경로>.png`
   가 Waterbirds 이미지와 픽셀 단위로 정렬된다.
   → `check_data.py` 로 반드시 검증하고 overlay 를 눈으로 확인할 것.

3. **`Image.ANTIALIAS` 삭제됨**
   원본 `generate_waterbirds.py` 는 Pillow 10 에서 동작하지 않는다. `Image.LANCZOS` 로 교체.

4. **공식 배포본 재현 불가**
   생성 스크립트를 다시 돌려도 시드 차이로 공식 tarball 과 동일한 데이터가 나오지 않는다.
   → **학습은 공식 tarball 로**, 생성 스크립트는 **반사실 이미지 생성 용도로만** 사용.

5. **`torch.load` 의 `weights_only`**
   torch 2.6 부터 기본값이 `True` 로 바뀌어 dict 를 담은 체크포인트 로드가 실패한다.
   `weights_only=False` 를 명시한다 (`train_erm.py` 에 처리됨).

6. **라벨 오류 (알려진 사항)**
   Eastern Towhee, Western Meadowlark, Western Wood Pewee 는 실제로 landbird 인데
   Waterbirds 에서 waterbird 로 표기되어 있다. 보고서 Limitations 에 기재할 것.

7. **RISE 연산량**
   이미지 한 장에 수천 번 forward 가 필요하다. 모델 수 × 이미지 수를 곱하면 매우 무거우므로
   마스크 수 또는 이미지 수를 줄이는 조정이 필요하다.

8. **Grad-CAM 해상도**
   마지막 conv 의 7×7 맵이라 새/배경 경계가 흐려진다.
   `Attr_bg` 계산 시 마스크를 7×7 로 내리는 방식과 맵을 224 로 올리는 방식의 결과가 달라질 수 있다.
   **한 가지로 고정하고 보고서에 명시할 것.**
   → 2026-11-10 **224 bilinear 업샘플로 고정**. 7×7 버전은 stress test(ST-B)용으로 함께 저장한다.

9. **`sbatch` 는 `dirname $0` 을 못 쓴다** ★2026-10-28 확인
   제출된 스크립트는 `/var/spool/slurm/d/jobNNN/slurm_script` 로 **복사되어** 실행된다.
   `source "$(dirname "$0")/_common.sh"` 는 잡이 시작하자마자 죽는다.
   `$SLURM_SUBMIT_DIR` 또는 절대경로를 쓸 것 (`scripts/train_slurm.sh` 에 반영됨).

10. **사전학습 가중치 캐시 위치** ★2026-10-28 확인
    torchvision 은 `~/.cache/torch` 에 ResNet-50 가중치를 받는데, `/home` 은
    계산 노드와 공유되지 않고 90% 차 있다. 노드가 바뀔 때마다 다시 받게 된다.
    `scripts/_common.sh` 에서 `TORCH_HOME=/data/$USER/.torch` 로 돌려둔다.

11. **conda 기본 채널 ToS** ★2026-10-28 확인
    `conda create` 가 `repo.anaconda.com` 채널 ToS 동의를 요구하며 멈춘다.
    `--override-channels -c conda-forge` 로 피한다 (`scripts/setup_env.sh` 에 반영됨).

12. **torch CUDA 빌드 선택** ★2026-10-28 확인
    aurora 의 드라이버는 535.161.07 (CUDA 12.2) 이다. cu130 휠은
    `CUDA driver is too old` 로 `torch.cuda.is_available()` 이 `False` 였다.
    cu121 휠은 GPU 동작을 확인했다. 다른 빌드는 테스트하지 않았다. **cu121 휠을 쓸 것.**

13. **`S` 는 방향을 구분하지 않는다** ★2026-11-03 확인
    원래 틀린 예측이 맞게 바뀌어도 flip 으로 센다. dense 모델의 opp flip 상당수가 이런 경우이고,
    그래서 `landbird/water` 의 `S` 가 음수로 나온다. 그룹별 `S` 만 보고 해석하지 말고
    `by_group_and_orig_correct` 도 함께 볼 것.

14. **`srun` 은 표준입력을 삼킨다** ★2026-11-03 확인
    `ssh aurora 'bash -s' <<EOF` 안에서 `srun` 을 부르면 뒤 명령이 실행되지 않는다.
    `</dev/null` 을 붙이거나 스크립트 파일로 실행한다.

15. **`srun` 은 프로젝트 폴더에서 실행한다** ★2026-11-03 확인
    잡 스크립트가 `SLURM_SUBMIT_DIR` 로 프로젝트 경로를 잡으므로, 다른 폴더에서 실행하면
    `_common.sh` 를 찾지 못한다.

16. **GPU 없는 잡은 CPU 1개만 받는다** ★2026-11-10 확인
    `--cpus-per-task=4` 로 CPU 잡을 내면 `SUBMISSION REJECTED` 가 난다. `--ntasks=1 --cpus-per-task=1` 로 낸다.

17. **NAS 데이터 tar 가 없어졌다** ★2026-11-10 확인
    `/data/$USER/datasets/xai/` 가 통째로 없다 (11-06 경). g1, g3, g4, g7, g8 의 `/local_datasets` 에는 남아 있어
    `--exclude=aurora-g2,aurora-g5,aurora-g6` 로 잡을 건다. 복구 전까지는 새 노드에서 스테이징이 실패한다.

---

## 6. 코드 구조

```
CLAUDE.md               이 파일 (프로젝트 컨텍스트)
SETUP.md                환경 구축 순서 (aurora 실측값)
SERAPH_RULES.md         Seraph 사용 규칙 — 위반 시 1학기 정지
WORKLOG.md              날짜별 작업 기록
README.md               STEP 0-1 실행 방법
requirements.txt        torch/torchvision 은 setup_env.sh 에서 cu121 로 따로 설치
check_data.py           STEP 0. 데이터/마스크 점검 (torch 불필요)
waterbirds.py           데이터셋 로더, 전처리, 그룹별 평가 유틸
train_erm.py            STEP 1. dense ERM 학습
generate_counterfactuals.py  STEP 2. 반사실 이미지 생성
measure_shortcut.py     STEP 2. BFR, S 측정. --save_probs 로 반사실 확률·이미지별 Δp 저장 (STEP 3+)
prune_finetune.py       STEP 3. 전역 magnitude pruning + fine-tuning
summarize_pruning.py    STEP 3. sparsity 별 정확도·S 요약표 (numpy 만 필요)
analyze_h1.py           STEP 3+. H1 재판정 — ft0/dense 대비 짝지은 two-way bootstrap (numpy/pandas)
scripts/
  _common.sh            잡 스크립트 공통 경로·캐시 설정
  setup_env.sh          conda 환경 xai 생성 (계산 노드에서 실행)
  download_data.sh      tar 3종을 NAS 에 내려받기 (압축 해제 안 함)
  stage_data.sh         tar -> /local_datasets 로 펼치기 (모든 잡이 먼저 호출)
  check_data_slurm.sh   STEP 0 실행 래퍼
  smoke_test.sh         파이프라인 점검 (subset 128, 2 epoch)
  train_slurm.sh        STEP 1 sbatch 제출 (batch_ugrad)
  train_direct.sh       srun 대화형 세션용
  download_places.sh    Places 평가셋에서 배경 400장 추출 (sbatch)
  stage_cf.sh           배경·반사실 tar 를 로컬로 펼치기
  generate_cf_slurm.sh  반사실 생성 후 NAS 에 tar 로 저장
  measure_shortcut_slurm.sh  S 측정 (sbatch)
  prune_slurm.sh        seed 하나의 sparsity 목록을 pruning + fine-tuning + S·Δp 측정 (sbatch)
  remeasure_slurm.sh    학습된 모델들의 반사실 확률 재측정 + 회귀 확인 (sbatch)
results/                요약표 (git 추적)
data_check/             STEP 0 결과. overlay 이미지만 git 추적
runs/                   실행 산출물. args.json, log.csv, final_test.json, shortcut_*.json,
                        shortcut_*_rows.csv.gz, shortcut_*_dp.csv.gz 만 git 추적
logs/                   SLURM 출력. erm-*.out, shortcut-*.out, prune-*.out, remeasure-*.out 만 git 추적
```

`waterbirds.py` 의 공개 API
- `WaterbirdsDataset(wb_root, split, transform, return_mask, cub_root)`
  → `(image, y, group, index)` 또는 `return_mask=True` 면 `(..., mask)`
- `build_transform(train, augment, resolution)`
- `evaluate_groups(model, loader, device, train_group_counts)` → dict
- `format_eval(res)` → 출력용 문자열

전처리는 Sagawa et al. 공식 구현을 따른다.
- eval: `Resize((256,256))` → `CenterCrop(224)`
- train: `RandomResizedCrop(224, scale=(0.7,1.0))` → `RandomHorizontalFlip`

---

## 7. STEP 2 설계 메모

구현은 `generate_counterfactuals.py`, `measure_shortcut.py` 에 있다. 결과는 `WORKLOG.md` 참고.
배경은 Places365 평가셋 4개 카테고리 400장(타입별 200장)을 쓴다. 학습 풀 `train_large` 는
112.6GB 인데 서버가 연결당 약 0.5MB/s 라 받지 않았다.

입력: Waterbirds 테스트 이미지 + CUB 마스크 + Places 배경 풀
출력: 이미지마다
- `x_opp` : 반대 타입 배경에 합성 (여러 장 샘플링 권장)
- `x_same`: 같은 타입의 **다른** 배경에 합성

구현 시 주의
- 합성 방식은 원본과 동일해야 한다 (`crop_and_resize` → `combine_and_mask`).
  원본과 반사실이 같은 파이프라인을 공유해야 붙여넣기 artifact 가 조건 간에 일정하게 유지된다.
- 배경은 학습에 쓰이지 않은 held-out 이미지에서 가져온다.
- Places 카테고리: 땅 = `bamboo_forest`, `forest/broadleaf` / 물 = `ocean`, `lake/natural`
- 공식 스크립트처럼 소프트 마스크를 그대로 쓰고 `.jpg` 기본 품질로 저장한다.
- 배경을 여러 장 샘플링해 평균을 내면 `S` 의 분산이 줄어든다.

이후 `S` 측정 스크립트는 `evaluate_groups` 와 같은 구조로 작성하되,
원본/`x_opp`/`x_same` 세 조건의 예측을 비교해 flip 여부를 센다.

---

## 8. 작업 규칙

- 새 스크립트는 `--help` 가 동작하도록 argparse 를 붙인다.
- 무거운 실행 전에는 `--subset` 같은 점검 모드를 먼저 돌린다.
- 결과는 항상 seed 3개로 내고 bootstrap 신뢰구간을 함께 보고한다.
- 체크포인트/데이터는 git 에 올리지 않는다. 추적하는 결과 파일은 `.gitignore` 참고.
- 수치를 보고서에 쓸 때는 다른 사람이 한 번 재현한 뒤 확정한다.
