# Seraph 사용 규칙 (요약)

출처: KHU Vision and Learning Lab 튜토리얼 1·3·4·5 (2023-09) + 6 (2026-07).
**위반 적발 시 1학기 사용 정지.** 관리자가 사전 통보 없이 job 을 종료할 수 있다.

---

## 하지 말 것

| 금지 | 이유 / 대안 |
|---|---|
| 마스터 노드에서 python 실행 | `srun` / `sbatch` 로 계산 노드를 받아서 할 것 |
| 마스터 노드에서 데이터셋 압축 해제 | `stage_data.sh` 가 계산 노드에서 수행 |
| 마스터 노드에서 재귀 탐색 (`find`, `rg`, `grep -R`) | NAS 에 큰 부하. 꼭 필요하면 계산 노드에서 |
| **마스터 노드에서 AI 에이전트 실행** | Claude Code 는 로컬 PC 에서 띄우고 ssh 로 명령을 보낸다 |
| dataloader 가 NAS(`/data`) 를 직접 읽기 | tar 로 두고 `/local_datasets` 에 풀어서 읽는다 |
| 생성 이미지를 NAS 에 직접 저장 | 로컬에 쌓고 끝날 때 정리해서 올린다 |
| `/home` 에 코드·환경 저장 | 계산 노드와 공유되지 않는다. `/data/$USER` 를 쓸 것 |
| public conda 환경에 `pip/conda install` | 개인 환경(`/data/$USER/miniconda3/envs/...`)을 만들어 쓴다 |
| Remote-SSH 로 큰 상위 디렉터리 열기 | `/data/$USER/xai_project` 만 연다 |
| 계정 공유 | 블랙리스트 + 사용 제한 |
| GPU 남용 | QoS 한도 안에서 쓰되 놀리는 잡을 띄워두지 않는다 |

## 할 것

- 코드·conda 환경·tar: `/data/$USER/`
- 설정 파일만: `/home/$USER/`
- 학습 데이터 읽기: `/local_datasets/$USER/...` (노드 로컬 NVMe)
- 먼저 `srun` 으로 소규모 점검 → 그 다음 `sbatch` 로 본 학습
- 제출 전 가상환경 확인. `sbatch` 는 PATH 를 그대로 상속한다
  (이 저장소는 `envs/xai/bin/python` 절대경로를 써서 이 문제를 피한다)
- 로그 디렉터리는 제출 전에 미리 만든다 (`mkdir -p logs`)

## 이 프로젝트에 적용된 내용

| 규칙 | 어디서 처리하는가 |
|---|---|
| NAS 직접 읽기 금지 | `scripts/stage_data.sh` — tar → `/local_datasets/$USER/xai` |
| 마스터 노드 python 금지 | 모든 스크립트가 `srun`/`sbatch` 전용 |
| conda init 미설정 노드 | `conda activate` 대신 절대경로 python 호출 |
| `/home` 미공유 | 저장소·출력 모두 `/data/$USER/xai_project` |
| Remote-SSH I/O | `.vscode/settings.json` 에 autofetch/watcher 제외 설정 |

## 모니터링 명령

```bash
squeue -u $USER              # 내 잡
show-qos                     # 내 한도 (동시 8잡 / 제출 16 / GPU 8 / 6일)
show-assoc                   # 소속 account, QoS
slurm-gres-viz -i            # GPU 점유 현황
scancel <JOBID>              # 취소
ssh aurora-g3 -t htop        # 노드 부하 (잡이 돌고 있을 때만)
```
