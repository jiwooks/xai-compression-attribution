# 작업 기록

진행 상황과 그 과정에서 알아낸 것을 날짜순으로 남긴다.
최신 항목이 위로 온다. 새 항목은 `## YYYY-MM-DD` 로 시작한다.

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
