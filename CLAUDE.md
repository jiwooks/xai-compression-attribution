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

**추적성 (RQ 의 직접적 답)**
- *모델 단위*: 압축 모델들에 걸친 `mean(Attr_bg)` 와 `S` 의 Spearman 순위 상관
- *이미지 단위*: `Attr_bg(x)` 로 "x 가 반대 배경에서 실제로 뒤집히는가" 를 예측한 **AUROC**
  (0.5 = 무작위, 1 = 완벽). sparsity 에 따라 이 값이 떨어지는지가 H2 의 핵심 판정.

---

## 3. 파이프라인 단계와 현재 상태

| STEP | 내용 | 상태 |
|---|---|---|
| 0 | 데이터 점검 + 마스크 정렬 검증 | **자동 점검 통과 (2026-10-28)** — 마스크 파일·크기 200/200, 일치율 95.0%. overlay 는 6장 중 2장만 육안 확인 |
| 1 | dense ERM 기준 모델 학습 | **1회 실행, 재현 전 (2026-10-28)** — seed 0·1 통과, seed 2 는 avg 89.59 로 확인 필요. avg 90.00±0.39 / WGA 66.25±3.90 / 격차 23.75pt |
| 2 | 반사실 이미지 생성 + `S` 측정 | **미착수** |
| 3 | pruning 파이프라인 | **미착수** |
| 4 | attribution (Grad-CAM/IG/RISE) + `Attr_bg` | **미착수** |
| 5 | 추적성 분석 (Spearman, AUROC) | **미착수** |
| 6 | stress test S1~S5 | **미착수** |

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
scripts/
  _common.sh            잡 스크립트 공통 경로·캐시 설정
  setup_env.sh          conda 환경 xai 생성 (계산 노드에서 실행)
  download_data.sh      tar 3종을 NAS 에 내려받기 (압축 해제 안 함)
  stage_data.sh         tar -> /local_datasets 로 펼치기 (모든 잡이 먼저 호출)
  check_data_slurm.sh   STEP 0 실행 래퍼
  smoke_test.sh         파이프라인 점검 (subset 128, 2 epoch)
  train_slurm.sh        STEP 1 sbatch 제출 (batch_ugrad)
  train_direct.sh       srun 대화형 세션용
data_check/             STEP 0 결과. overlay 이미지만 git 추적
runs/                   실행 산출물. args.json, log.csv, final_test.json 만 git 추적
logs/                   SLURM 출력. erm-*.out 만 git 추적
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

## 7. 다음에 할 일 (STEP 2 설계 메모)

반사실 이미지 생성기를 만든다.

입력: Waterbirds 테스트 이미지 + CUB 마스크 + Places 배경 풀
출력: 이미지마다
- `x_opp` : 반대 타입 배경에 합성 (여러 장 샘플링 권장)
- `x_same`: 같은 타입의 **다른** 배경에 합성

구현 시 주의
- 합성 방식은 원본과 동일해야 한다 (`crop_and_resize` → `combine_and_mask`).
  원본과 반사실이 같은 파이프라인을 공유해야 붙여넣기 artifact 가 조건 간에 일정하게 유지된다.
- 배경은 학습에 쓰이지 않은 held-out 이미지에서 가져온다.
- Places 카테고리: 땅 = `bamboo_forest`, `forest/broadleaf` / 물 = `ocean`, `lake/natural`
- Places 전체는 용량이 크다. 해당 카테고리만 받는 방법을 먼저 확인할 것.
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
