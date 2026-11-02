# STEP 0-1: 데이터 준비 + dense 기준 모델 학습

## 지금 단계의 목표

**좋은 모델이 아니라 "배경 shortcut 에 의존하는 모델"을 만드는 것.**

- 평균 정확도는 높게 (대략 90% 이상)
- worst-group 정확도(WGA)는 그보다 한참 낮게

이 격차가 없으면 압축에 따라 변할 shortcut 의존도 자체가 없어서
H1(압축이 shortcut 의존을 바꾸는가)을 검증할 수 없다.
그래서 이번 학습의 통과 기준은 "평균은 높고 WGA 는 낮다" 이다.

---

## 0. 먼저 읽을 것

- `SETUP.md` — aurora 클러스터 환경 구축 (실측값)
- `SERAPH_RULES.md` — Seraph 사용 규칙. **위반 시 1학기 사용 정지**

아래 명령은 전부 `/data/$USER/xai_project` 에서, **계산 노드를 받아서** 실행한다.
마스터 노드(`aurora-master`)에서 python 을 돌리면 안 된다.

---

## 1. 데이터 내려받기 (tar 3개)

```bash
srun -p debug_ugrad --cpus-per-task=4 --mem=8G -t 2:00:00 bash scripts/download_data.sh
```

NAS(`/data/$USER/datasets/xai/`)에 tar 로만 둔다. 압축은 풀지 않는다.

| 파일 | 크기 | 용도 |
|---|---|---|
| `waterbird_complete95_forest2water2.tar.gz` | 468MB | 학습/평가용 공식 배포본 |
| `CUB_200_2011.tgz` | 1.1GB | 원본 새 이미지·메타데이터 |
| `segmentations.tgz` | 38MB | **새 영역 마스크** |

> ⚠️ **`CUB_200_2011.tgz` 안에는 `segmentations/` 가 없다.**
> Caltech 의 **별도 레코드**(`w9d68-gec53`)에서 따로 받아야 한다.
> 이 마스크가 없으면 `Attr_bg` 도 STEP 2 반사실 합성도 불가능하다.
>
> Waterbirds 는 CUB 새를 segmentation 으로 오려 배경에 합성해 만들었고,
> **배경을 새 이미지 크기에 맞춰 잘라 붙인 뒤 CUB 와 같은 상대경로로 저장**한다.
> 따라서 CUB segmentation 이 Waterbirds 이미지와 픽셀 단위로 정렬된다.

학습 잡은 시작할 때 `scripts/stage_data.sh` 로 tar 를 계산 노드의
`/local_datasets/$USER/xai/` 에 펼친다. **dataloader 가 NAS 를 직접 읽으면 안 된다.**

---

## 2. 데이터 점검 (GPU 불필요, 먼저 실행)

```bash
srun -p debug_ugrad --cpus-per-task=8 --mem=16G -t 1:00:00 bash scripts/check_data_slurm.sh
```

확인할 것
1. 학습셋에서 새-배경 일치 비율이 **95% 내외**인가
2. 마스크 **파일 없음·크기 불일치 0건**인가
3. `data_check/` 의 overlay 이미지에서 새 윤곽만 정확히 밝게 남았는가

3번은 눈으로 직접 볼 것. 여기서 어긋나면 이후 `Attr_bg` 가 전부 무의미해진다.

**2026-10-28 실행 결과 (통과)**
```
train 4795 / val 1199 / test 5794,  새-배경 일치 95.0%
train 그룹 분포  g0 73.0% | g1 3.8% | g2 1.2% | g3 22.0%
마스크 파일·크기 검사 200/200 통과,  크기 불일치 0
새 영역 평균 면적 0.133 (중앙값 0.123)
  -> 균일 attribution 의 Attr_bg 기댓값 = 0.867  ★기준선
```
overlay 는 6장 중 2장만 눈으로 확인했다. **나머지 4장은 아직 확인 전이다.**

실제 `Attr_bg` 가 0.867 보다 **낮아야** 모델이 새를 본다고 말할 수 있다.

---

## 3. dense 기준 모델 학습

**본 실행 전에 1~2분짜리 점검부터:**
```bash
srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 0:30:00 \
     bash scripts/smoke_test.sh
```
에러 없이 끝까지 돌면 경로/로더/평가 코드가 정상이다.
(정확도 숫자는 의미 없음. 파이프라인만 확인하는 용도.)

**본 학습 — seed 3개를 한꺼번에 제출 (QoS 한도: 동시 8잡)**
```bash
mkdir -p logs
for s in 0 1 2; do sbatch scripts/train_slurm.sh $s; done
squeue -u $USER
tail -f logs/erm-<JOBID>.out
```

산출물은 `runs/erm_seed{0,1,2}/` 에 쌓인다.

주요 인자
| 인자 | 기본값 | 설명 |
|---|---|---|
| `--epochs` | 100 | Sagawa et al. 은 300 을 썼지만 사전학습 모델 + 4.8k 장이라 보통 더 일찍 수렴 |
| `--lr` | 1e-3 | SGD, momentum 0.9 |
| `--weight_decay` | 1e-4 | |
| `--batch_size` | 128 | |
| `--no_augment` | off | 증강을 끔. WGA 격차가 작을 때 shortcut 을 강화하는 용도 |
| `--amp` | off | mixed precision. 켜면 빠름 |

산출물
- `best.pt` — validation **평균** 정확도 기준 최고 체크포인트
- `last.pt`, `log.csv`, `final_test.json`

> 모델 선택을 WGA 기준으로 하면 안 된다. shortcut 이 약한 체크포인트가 뽑혀
> 실험 전제가 무너진다. ERM 표준 프로토콜대로 평균 기준으로 고른다.

---

## 4. 통과 여부 판정

스크립트가 마지막에 자동으로 출력한다.

| 결과 | 해석 | 다음 행동 |
|---|---|---|
| 평균 90%+ , 격차 15pt+ | shortcut 모델 확보 | STEP 2 로 진행 |
| 평균 90% 미만 | 학습 부족 | epoch 늘리기 / lr 조정 |
| 격차 15pt 미만 | shortcut 약함 | `--no_augment` 재학습, 그래도 안 되면 상관 100% 변형 데이터 |

### 2026-10-28 실행 결과 — seed 0·1 통과, seed 2 확인 필요

`best.pt`(validation 평균 정확도 기준 선택)의 테스트 성능:

| seed | avg | adj | WGA | 평균-WGA 격차 | best epoch |
|---|---|---|---|---|---|
| 0 | 90.37 | 97.86 | 64.49 | 25.9 | 60 |
| 1 | 90.04 | 97.84 | 70.72 | 19.3 | 75 |
| 2 | 89.59 | 97.78 | 63.55 | 26.0 | 90 |
| **평균** | **90.00** ± 0.39 | **97.83** ± 0.04 | **66.25** ± 3.90 | **23.75** ± 3.83 | |

- 최저 그룹은 세 seed 모두 **g2 (waterbird on land)** — 학습셋에 56장(1.2%)뿐인 최소 그룹
- landbird/land 99.6~99.8% vs waterbird/land 63.6~70.7% → 배경 shortcut 의존을 시사한다.
  행동 지수 `S` 는 STEP 2 에서 측정하므로 아직 직접 확인한 것은 아니다.
- 1 GPU(RTX 3090) 기준 100 epoch 에 약 14분. seed 3개 병렬로 15분

- `±` 는 seed 3개의 표본 표준편차다. `CLAUDE.md` §8 이 요구하는 bootstrap 신뢰구간은 아직 계산하지 않았다.
- 다른 사람의 재현 전이므로 보고서용 확정 수치가 아니다.

> seed 2 는 avg 89.59 로 위 표와 스크립트의 평균 90% 기준에 0.41pt 못 미쳐
> `[확인 필요] 평균 정확도가 낮음` 이 출력됐다. 격차 26.0pt 는 15pt 기준을 넘는다.
> 재학습 없이 이 모델을 쓸지는 팀에서 결정해야 한다.

참고로 문헌에 보고된 Waterbirds ERM 수치는 논문마다 편차가 크다.
평균 정확도도 "전체 평균"인지 "학습셋 그룹 비율로 가중한 평균"인지에 따라 달라지므로,
우리 스크립트는 **두 값을 모두 출력**한다 (`avg` / `adj`).
보고서에는 어느 쪽을 쓰는지 명시할 것.

---

## 다음 단계 (미리보기)

- **STEP 2**: 반사실 이미지 생성 + S 측정. 코드 완료, dense 결과는 `WORKLOG.md` 참고
- **STEP 3**: pruning 파이프라인
- **STEP 4**: attribution (Grad-CAM / IG / RISE) + Attr_bg
- **STEP 5**: 추적성 분석 (Spearman, AUROC)

STEP 2 가 사실상 이 프로젝트의 핵심 장치다.
