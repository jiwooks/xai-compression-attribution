# 모든 잡 스크립트가 source 한다. 단독 실행용이 아니다.
#
#   PROJ="${PROJ:-${SLURM_SUBMIT_DIR:-/data/$USER/xai_project}}"
#   source "$PROJ/scripts/_common.sh"
#
# sbatch 는 스크립트를 복사해 실행하므로 dirname $0 을 쓰면 안 된다.

PROJ="${PROJ:-/data/$USER/xai_project}"
CONDA_ROOT="${CONDA_ROOT:-/data/$USER/miniconda3}"
ENV_NAME="${ENV_NAME:-xai}"
PY="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
LOCAL_ROOT="${LOCAL_ROOT:-/local_datasets/$USER/xai}"
TAR_DIR="${TAR_DIR:-/data/$USER/datasets/xai}"
WB_ROOT="$LOCAL_ROOT/waterbird_complete95_forest2water2"
CUB_ROOT="$LOCAL_ROOT/CUB_200_2011"

# 캐시를 전부 NAS 로 돌린다.
# 기본값은 ~/.cache 인데 /home 은 (1) 계산 노드와 공유되지 않아 노드마다 다시 받고
# (2) 90% 차 있다. torchvision 의 ResNet-50 사전학습 가중치가 여기로 떨어진다.
export TORCH_HOME="/data/$USER/.torch"
export HF_HOME="/data/$USER/hf"
export PIP_CACHE_DIR="/data/$USER/pipcache"
export XDG_CACHE_HOME="/data/$USER/.cache"
mkdir -p "$TORCH_HOME" "$XDG_CACHE_HOME"

[ -x "$PY" ] || { echo "[중단] python 이 없다: $PY — 먼저 scripts/setup_env.sh 를 실행할 것"; exit 1; }
