#!/usr/bin/env bash
# Phase 1: brain fine-tuning on LIBERO-Spatial augmented data.
# Run from the headless-vla/ root directory.
set -euo pipefail

# ── Step 0: build the 3-task JSONL from HDF5 ─────────────────────────────────
echo "=== Building JSONL training data ==="
python -m data.annotation.libero_build_graph \
    --data-dir  ~/vlm_benchmarking/data/libero_spatial_v5 \
    --out        data/splits/libero_train.jsonl \
    --image-root data/images \
    --val-frac   0.1 \
    --subsample  1 \
    --seed       42

# ── Step 1: brain fine-tuning ─────────────────────────────────────────────────
echo "=== Training brain (Phase 1) ==="
python -m models.brain.train --config-name=phase1_libero

echo "Brain training complete."
