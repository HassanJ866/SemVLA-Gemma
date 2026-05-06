#!/usr/bin/env bash
#SBATCH --job-name=semvla_train_adapter
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-08:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s338920@studenti.polito.it
#SBATCH --output=logs/slurm_%j_train_adapter.out
#SBATCH --error=logs/slurm_%j_train_adapter.err

set -euo pipefail

module purge
module load miniforge/24.3.0-0
conda activate vlm

cd ~/SemVLA-Gemma

mkdir -p logs ckpts/franka_7dof

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Starting adapter training (franka_7dof)"
echo "  Action dim : 7"
echo "  Chunk size : 16"
echo "  Steps      : 30000"
echo "========================================================"

python -m models.adapter.train --config-name=franka_7dof

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Adapter training complete."
echo "  Checkpoint : ckpts/franka_7dof/final"
echo "  Stats      : ckpts/franka_7dof/final/action_stats.json"
echo "========================================================"
