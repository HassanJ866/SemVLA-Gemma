#!/usr/bin/env bash
# Linux server setup + SemVLA expert training launcher.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_YAML="${ENV_YAML:-${REPO_ROOT}/env/semvla-server-conda.yaml}"
ENV_NAME="${ENV_NAME:-semvla-server}"
MERGED_DIR="${MERGED_DIR:-${REPO_ROOT}/ckpts/brain_phase1/final}"
ADAPTER_CKPT="${ADAPTER_CKPT:-}"
DATASET_REPO_ID="${DATASET_REPO_ID:-HuggingFaceVLA/libero}"
STEPS="${STEPS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
SAVE_FREQ="${SAVE_FREQ:-100}"
LOG_FREQ="${LOG_FREQ:-100}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-semvla-lerobot}"
DRY_RUN="${DRY_RUN:-0}"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }
log() { echo "[$(timestamp)] $*"; }
die() { echo "[$(timestamp)] ERROR: $*" >&2; exit 1; }
require_cmd() { command -v "$1" >/dev/null 2>&1 || die "Missing required command: $1"; }
require_dir() { [ -d "$1" ] || die "Missing directory: $1"; }

log "Starting SemVLA server setup."
require_cmd conda
require_cmd python
require_dir "$REPO_ROOT"
require_dir "${REPO_ROOT}/scripts"
require_dir "${REPO_ROOT}/lerobot"
[ -f "$ENV_YAML" ] || die "Missing conda env file: $ENV_YAML"

log "Using repo root: $REPO_ROOT"
log "Using conda env yaml: $ENV_YAML"

if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  log "Conda env '${ENV_NAME}' already exists; updating in place."
  conda env update -n "$ENV_NAME" -f "$ENV_YAML" --prune
else
  log "Creating conda env '${ENV_NAME}'."
  conda env create -n "$ENV_NAME" -f "$ENV_YAML"
fi

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

log "Verifying python/pip/lerobot-train entrypoint."
python --version
pip --version
python -c "import torch, torchvision, numpy, scipy, transformers; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'numpy', numpy.__version__, 'scipy', scipy.__version__, 'transformers', transformers.__version__)"

if ! command -v lerobot-train >/dev/null 2>&1; then
  log "lerobot-train not found; reinstalling editable lerobot package."
  pip install -e "${REPO_ROOT}/lerobot[smolvla,libero,training]"
fi
require_cmd lerobot-train

if [ -z "$ADAPTER_CKPT" ]; then
  ADAPTER_CKPT="$(ls -d "${REPO_ROOT}"/ckpts/brain_phase1/checkpoint-* 2>/dev/null | sort -V | tail -n 1 || true)"
fi
ADAPTER_CKPT="${ADAPTER_CKPT:-${REPO_ROOT}/ckpts/brain_phase1/checkpoint-1500}"

if [ ! -d "$MERGED_DIR" ]; then
  log "Merged brain checkpoint not found, will merge from adapter: $ADAPTER_CKPT"
  [ -d "$ADAPTER_CKPT" ] || die "Adapter checkpoint directory not found: $ADAPTER_CKPT"
  python "${REPO_ROOT}/scripts/merge_brain.py" --adapter "$ADAPTER_CKPT" --output "$MERGED_DIR"
fi
require_dir "$MERGED_DIR"

if [ "${WANDB_ENABLE}" = "true" ] && [ -z "${WANDB_API_KEY:-}" ]; then
  log "WANDB is enabled but WANDB_API_KEY is not set; wandb may prompt interactively."
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"
cd "$REPO_ROOT"

TRAIN_CMD=(
  lerobot-train
  --policy.type=semvla
  --policy.brain_model_path="$MERGED_DIR"
  --policy.device=cuda
  --policy.push_to_hub=false
  --dataset.repo_id="$DATASET_REPO_ID"
  --batch_size="$BATCH_SIZE"
  --steps="$STEPS"
  --save_freq="$SAVE_FREQ"
  --log_freq="$LOG_FREQ"
  --wandb.enable="$WANDB_ENABLE"
  --wandb.project="$WANDB_PROJECT"
)

log "Resolved training command:"
printf '  %q' "${TRAIN_CMD[@]}"
printf '\n'

if [ "$DRY_RUN" = "1" ]; then
  log "Dry run enabled; skipping launch."
  exit 0
fi

log "Launching SemVLA expert training."
"${TRAIN_CMD[@]}"
