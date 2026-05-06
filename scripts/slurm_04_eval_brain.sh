#!/usr/bin/env bash
#SBATCH --job-name=semvla_eval_brain
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=0-04:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s338920@studenti.polito.it
#SBATCH --output=logs/slurm_%j_eval_brain.out
#SBATCH --error=logs/slurm_%j_eval_brain.err

set -euo pipefail

module purge
module load miniforge/24.3.0-0
conda activate vlm

cd ~/SemVLA-Gemma

mkdir -p logs

CKPT=${1:-ckpts/brain_phase1/final}

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Evaluating brain"
echo "  Checkpoint : $CKPT"
echo "  Metrics    : IoU (grounding), F1 (parsing), exact-match (task_synthesis)"
echo "========================================================"

python -m models.brain.eval \
    --ckpt        "$CKPT" \
    --val-jsonl   data/splits/libero_train_val.jsonl \
    --image-root  data/images \
    --n-samples   500

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Evaluation complete."
echo "========================================================"

# Phase gates (informational — not enforced here):
#   grounding IoU       >= 0.70
#   parsing   F1        >= 0.80
#   task_synthesis JSON validity >= 0.97
