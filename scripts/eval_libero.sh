#!/usr/bin/env bash
# End-to-end evaluation: brain + franka adapter on LIBERO-Spatial.
# Run from the headless-vla/ root directory.
set -euo pipefail

BRAIN_CKPT=${1:-ckpts/brain_phase1/final}
ADAPTER_CKPT=${2:-ckpts/franka_7dof/final}
ACTION_STATS=${3:-ckpts/franka_7dof/final/action_stats.json}

echo "=== LIBERO-Spatial Evaluation ==="
echo "  Brain:   ${BRAIN_CKPT}"
echo "  Adapter: ${ADAPTER_CKPT}"
echo "  Stats:   ${ACTION_STATS}"

python -m models.adapter.eval \
    --config-name=libero_full \
    brain_ckpt="${BRAIN_CKPT}" \
    adapter_ckpt="${ADAPTER_CKPT}" \
    action_stats="${ACTION_STATS}"
