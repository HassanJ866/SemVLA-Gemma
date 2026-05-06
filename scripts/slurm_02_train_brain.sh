#!/usr/bin/env bash
#SBATCH --job-name=semvla_train_brain
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s338920@studenti.polito.it
#SBATCH --output=logs/slurm_%j_train_brain.out
#SBATCH --error=logs/slurm_%j_train_brain.err

set -euo pipefail

module purge
module load miniforge/24.3.0-0
conda activate vlm

cd ~/SemVLA-Gemma

mkdir -p logs ckpts/brain_phase1

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Starting brain fine-tune (Phase 1)"
echo "  Model : google/gemma-4-E4B-it"
echo "  LoRA  : r=32, alpha=64"
echo "  Steps : 50000"
echo "========================================================"

python -m models.brain.train --config-name=phase1_libero

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Brain training complete."
echo "  Checkpoint: ckpts/brain_phase1/final"
echo "========================================================"
