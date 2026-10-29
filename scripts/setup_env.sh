#!/usr/bin/env bash
# conda 환경 생성.
#
# ★ 반드시 계산 노드에서 실행할 것 (Seraph 규칙: 마스터 노드에서 python 실행 금지)
#   srun -p debug_ugrad --cpus-per-task=8 --mem=16G -t 1:00:00 bash scripts/setup_env.sh
#
# 기존 ax 환경은 건드리지 않는다. 이 프로젝트 전용으로 xai 를 새로 만든다.
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/data/$USER/miniconda3}"
ENV_NAME="${ENV_NAME:-xai}"
ENV_DIR="$CONDA_ROOT/envs/$ENV_NAME"
PROJ="${PROJ:-/data/$USER/xai_project}"

# 캐시는 공유 NAS 안으로. 홈(/home)은 계산 노드와 공유되지 않고 용량도 부족하다.
export PIP_CACHE_DIR="/data/$USER/pipcache"
export CONDA_PKGS_DIRS="$CONDA_ROOT/pkgs"

echo "[1/3] conda 환경 생성: $ENV_NAME (python 3.11)"
if [ -d "$ENV_DIR" ]; then
  echo "      이미 존재 -> 건너뜀"
else
  # Anaconda 기본 채널(repo.anaconda.com)은 ToS 동의를 요구하고 상업적 이용 제약이 있다.
  # conda-forge 만 쓰면 그 문제를 피할 수 있다.
  "$CONDA_ROOT/bin/conda" create -y -n "$ENV_NAME" \
      --override-channels -c conda-forge python=3.11
fi

PY="$ENV_DIR/bin/python"

echo "[2/3] torch 2.5.1 + torchvision 0.20.1 (cu121) 설치"
# 드라이버가 535.161.07 (CUDA 12.2) 이라 cu121 휠을 쓴다.
# cu124/cu126/cu130 휠은 "driver is too old" 로 cuda.is_available() 이 False 가 된다.
"$PY" -m pip install --no-input \
    torch==2.5.1+cu121 torchvision==0.20.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "[3/3] 공통 의존성 설치"
"$PY" -m pip install --no-input -r "$PROJ/requirements.txt"

echo
echo "=== 설치 확인 ==="
"$PY" - <<'PY'
import sys, torch, torchvision
print("python  ", sys.version.split()[0])
print("torch   ", torch.__version__, "| cuda_build", torch.version.cuda)
print("tv      ", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
for m in ["numpy", "pandas", "PIL", "scipy", "sklearn", "matplotlib", "tqdm"]:
    print(f"  {m:12s}", getattr(__import__(m), "__version__", "?"))
PY
echo "=== setup_env.sh 완료 ==="
