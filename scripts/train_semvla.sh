#!/usr/bin/env bash
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-23:59:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s338920@studenti.polito.it
#SBATCH --output=logs/slurm_%j_train_semvla.out
#SBATCH --error=logs/slurm_%j_train_semvla.err

module purge
module load miniforge/24.3.0-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vla_bench_py312

cd ~/SemVLA-Gemma

mkdir -p logs

PROJECT_DIR="$HOME/SemVLA-Gemma"

# ── 1. Merge LoRA brain (skip if already done) ────────────────────────────────
MERGED="$PROJECT_DIR/ckpts/brain_phase1/final"
if [ ! -d "$MERGED" ]; then
    echo "========================================================"
    echo "[$(date +'%H:%M:%S')] Merging LoRA adapter ..."
    echo "========================================================"
    python scripts/merge_brain.py \
        --adapter ckpts/brain_phase1/checkpoint-1500 \
        --output  "$MERGED"
else
    echo "[$(date +'%H:%M:%S')] Merged model already exists, skipping merge."
fi

# ── 2. Install lerobot (fast — skips if already installed) ───────────────────
echo "========================================================"
echo "[$(date +'%H:%M:%S')] Installing lerobot extras ..."
echo "========================================================"
pip install -q -e "lerobot/[smolvla,libero,training]"

# ── 3. Train action head ──────────────────────────────────────────────────────
echo "========================================================"
echo "[$(date +'%H:%M:%S')] Starting SemVLA action-head training ..."
echo "========================================================"

lerobot-train \
    --policy.type=semvla \
    --policy.brain_model_path="$MERGED" \
    --policy.device=cuda \
    --policy.push_to_hub=false \
    --dataset.repo_id=lerobot/libero_spatial \
    --batch_size=16 \
    --steps=50000 \
    --save_freq=100 \
    --log_freq=100 \
    --wandb.enable=true \
    --wandb.project=semvla-lerobot

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Done."
echo "========================================================"
