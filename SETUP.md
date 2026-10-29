# 환경 구축 순서 (aurora 클러스터)

클러스터: `aurora.khu.ac.kr:30080` / 계정 `jwook2480` / 소속 account `ugrad`

> 아래 내용은 2026-10-28 에 실제로 클러스터에서 확인한 값이다.
> Seraph 공통 규칙은 `SERAPH_RULES.md` 를 먼저 읽을 것. **위반 시 1학기 사용 정지.**

---

## 0. 확인된 환경 요약

| 항목 | 값 |
|---|---|
| 스케줄러 | SLURM. 학부 파티션 `batch_ugrad`(무제한) / `debug_ugrad`(4시간) |
| 노드 | `aurora-g1~g8`, 노드당 CPU 64코어(g1은 96) / RAM 약 420GB / **GPU 8장** |
| GPU | **RTX 3090 24GB**, 드라이버 `535.161.07` (CUDA 12.2), nvcc 12.3 |
| QoS 한도 | 동시 실행 8잡 / 제출 16잡 / GPU 8장 / 최대 6일 |
| 마스터 노드 | `aurora-master`. **GPU 없음**, `nvidia-smi` 자체가 없다 |
| 시스템 python | 3.8.10 — `requirements.txt` 설치 불가(`scipy>=1.11` 휠 없음). conda 를 쓴다 |
| conda | `/data/jwook2480/miniconda3` (이미 설치되어 있음) |
| `/home` | **계산 노드와 공유되지 않는다.** 90% 사용 중. 설정 파일만 둘 것 |
| `/data` (NAS) | 109T 중 31T 여유. 코드·환경·tar 를 여기 둔다 |
| `/local_datasets` | 계산 노드 로컬 NVMe. 데이터는 잡 안에서 여기로 펼쳐 쓴다 |
| 외부 네트워크 | 열려 있음 (PyPI 89MB/s) |
| 공용 Waterbirds/CUB | **없음**. 직접 받아야 한다 |

---

## 1. VSCode 로 접속

`~/.ssh/config` (로컬 PC):

```
Host aurora
    HostName aurora.khu.ac.kr
    User jwook2480
    Port 30080
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60
```

VSCode 확장 **Remote - SSH** → `Remote-SSH: Connect to Host` → `aurora`

열 폴더는 `/data/jwook2480/xai_project`. **`/data` 나 `/data/datasets` 같은 상위 디렉터리를
통째로 열지 말 것** — Remote-SSH 가 재귀 인덱싱을 돌려 NAS 에 부하를 준다.

`.vscode/settings.json` 에 git 자동 fetch 를 꺼 둔다 (저장소에 포함되어 있음).

---

## 2. Claude Code 는 **로컬 PC 에서** 실행한다

마스터 노드에서 AI 에이전트 실행은 금지되어 있다 (Seraph 규칙).
계산 노드에 설치할 이유도 없다. 로컬 PC 에서 Claude Code 를 띄우고
클러스터 작업은 `ssh aurora ...` 로 보내면 된다.

---

## 3. 파이썬 환경

**마스터 노드에서 python/conda/pip 를 실행하지 말 것.** 계산 노드를 받아서 한다.

```bash
cd /data/$USER/xai_project
srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 1:00:00 \
     bash scripts/setup_env.sh
```

`scripts/setup_env.sh` 가 하는 일
- `/data/$USER/miniconda3/envs/xai` 생성 (python 3.11, **conda-forge 전용**)
  — Anaconda 기본 채널은 ToS 동의를 요구하므로 쓰지 않는다
- `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121` 설치
  — 드라이버가 535 라서 cu124/cu126/cu130 휠은 `driver is too old` 로 GPU 를 못 잡는다
- `requirements.txt` 설치

기존 `ax` 환경(py3.11 + torch 2.5.1+cu121)은 다른 작업용이므로 건드리지 않는다.

> 잡 스크립트에서는 `conda activate` 대신 `envs/xai/bin/python` 을 직접 부른다.
> `~/.bashrc` 가 `/home` 에 있어 처음 가는 노드에서는 conda init 이 안 되어 있다.

---

## 4. 데이터

**2단계로 나뉜다.** NAS 에는 tar 만 두고, 압축 해제는 계산 노드에서 한다.

```bash
# (1) NAS 로 내려받기 (한 번만). 압축은 풀지 않는다.
srun -p debug_ugrad --cpus-per-task=4 --mem=8G -t 2:00:00 bash scripts/download_data.sh
#     -> /data/$USER/datasets/xai/*.tar.gz

# (2) 계산 노드 로컬로 펼치기 — 학습 스크립트가 자동으로 호출한다.
bash scripts/stage_data.sh
#     -> /local_datasets/$USER/xai/{waterbird_complete95_forest2water2,CUB_200_2011}
```

- Waterbirds 압축 467MB / 해제 후 약 1.3GB. `nlp.stanford.edu` 는 응답이 없어
  `downloads.cs.stanford.edu` 미러를 먼저 쓴다.
- CUB-200-2011 약 1.1GB. 마스크(`segmentations/`) 때문에 반드시 필요하다.
- `/local_datasets` 는 **노드마다 따로**다. `stage_data.sh` 는 이미 있으면 건너뛰므로
  잡마다 불러도 된다.

---

## 5. 점검 → 학습

```bash
cd /data/$USER/xai_project
mkdir -p logs

# (1) 데이터/마스크 점검
srun -p debug_ugrad --cpus-per-task=8 --mem=16G -t 1:00:00 bash scripts/check_data_slurm.sh

# (2) 파이프라인 점검 (1분)
srun -p debug_ugrad --gres=gpu:1 --cpus-per-gpu=8 --mem-per-gpu=32G -t 0:30:00 \
     bash scripts/smoke_test.sh

# (3) 본 학습 — seed 0,1,2 (동시 8잡까지 가능)
for s in 0 1 2; do sbatch scripts/train_slurm.sh $s; done
squeue -u $USER
```

취소는 `scancel <JOBID>`.

판정 기준은 `README.md` 와 `CLAUDE.md` 참고.
**평균 정확도는 높고 WGA 는 낮아야 한다** (격차 15pt 이상이 목표).

---

## 6. 팀 공유

```bash
git init
git add .
git commit -m "STEP 0-1: 데이터 점검 및 ERM 학습 코드"
git remote add origin <저장소 URL>
git push -u origin main
```

`.gitignore` 에 데이터/체크포인트/로그가 이미 제외되어 있다.
