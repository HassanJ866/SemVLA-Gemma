#!/bin/bash
#SBATCH --job-name=semvla_train
#SBATCH --partition=TODO_PARTITION        # <-- replace with your partition name
#SBATCH --account=TODO_ACCOUNT           # <-- replace with your account/project name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/semvla_%j.out
#SBATCH --error=logs/semvla_%j.err

set -euo pipefail

# ── 1. Environment ────────────────────────────────────────────────────────────
source ~/.bashrc
conda activate vla_bench            # <-- replace with your conda env name if different

PROJECT_DIR="$HOME/SemVLA-Gemma"    # <-- replace if your repo lives elsewhere
cd "$PROJECT_DIR"

mkdir -p logs

# ── 2. Merge LoRA brain (skip if ckpts/brain_phase1/final already exists) ────
MERGED="$PROJECT_DIR/ckpts/brain_phase1/final"
if [ ! -d "$MERGED" ]; then
    echo "[$(date)] Merging LoRA adapter ..."
    python scripts/merge_brain.py \
        --adapter ckpts/checkpoint-1500 \
        --output  "$MERGED"
else
    echo "[$(date)] Merged model already exists at $MERGED, skipping merge."
fi

# ── 3. Install lerobot (fast — skips if already installed) ───────────────────
echo "[$(date)] Installing lerobot extras ..."
pip install -q -e "lerobot/[smolvla,libero,training]"

# ── 4. Train action head ──────────────────────────────────────────────────────
echo "[$(date)] Starting SemVLA action-head training ..."
python -m lerobot.scripts.train \
    --policy.type=semvla \
    --policy.brain_model_path="$MERGED" \
    --policy.device=cuda \
    --dataset.repo_id=lerobot/libero_spatial \
    --dataset.image_transforms.enable=true \
    --training.num_epochs=100 \
    --training.batch_size=16 \
    --training.grad_clip_norm=10.0 \
    --output_dir=ckpts/semvla_action_head \
    --wandb.enable=true \
    --wandb.project=semvla-lerobot

echo "[$(date)] Training finished."
