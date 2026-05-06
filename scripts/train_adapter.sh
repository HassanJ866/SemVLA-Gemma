#!/usr/bin/env bash
# Phase 2: adapter training for franka_7dof on LIBERO-Spatial.
# Run from the headless-vla/ root directory.
# Requires brain training (train_brain.sh) to have completed first.
set -euo pipefail

ARM=${1:-franka_7dof}

# ── Step 0: extract motion-primitive windows ──────────────────────────────────
echo "=== Labelling motion primitives for ${ARM} ==="
python -m data.annotation.motion_primitive_label \
    --data-dir  ~/vlm_benchmarking/data/libero_spatial_v5 \
    --out        data/splits/adapter_${ARM}.jsonl \
    --chunk-size 16 \
    --val-frac   0.1 \
    --seed       42

# ── Step 1: compute action normalisation stats (MANDATORY before training) ────
echo "=== Computing action normalisation stats ==="
python -c "
from models.middleware.normalize import compute_and_save_stats
compute_and_save_stats(
    'data/splits/adapter_${ARM}.jsonl',
    'ckpts/${ARM}/action_stats.json',
    action_dim=7,
)
"

# ── Step 2: adapter training ──────────────────────────────────────────────────
echo "=== Training adapter (${ARM}) ==="
python -m models.adapter.train --config-name=${ARM}

echo "Adapter training complete for ${ARM}."
