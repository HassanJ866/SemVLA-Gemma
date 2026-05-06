#!/usr/bin/env bash
#SBATCH --job-name=semvla_build_data
#SBATCH --partition=gpu_a40
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=0-02:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s338920@studenti.polito.it
#SBATCH --output=logs/slurm_%j_build_data.out
#SBATCH --error=logs/slurm_%j_build_data.err

set -euo pipefail

module purge
module load miniforge/24.3.0-0
conda activate vlm

cd ~/SemVLA-Gemma

# HDF5 source lives in the sibling repo; all generated outputs stay in this repo
DATA_DIR=~/vlm_benchmarking/data/libero_spatial_v5

mkdir -p logs data/splits data/images

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Building brain JSONL"
echo "  Source : $DATA_DIR"
echo "  Output : data/splits/  data/images/"
echo "========================================================"

python -m data.annotation.libero_build_graph \
    --data-dir  "$DATA_DIR" \
    --out        data/splits/libero_train.jsonl \
    --image-root data/images \
    --val-frac   0.1 \
    --subsample  1 \
    --seed       42

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Verifying task type balance"
python -c "
import json
from collections import Counter
counts = Counter(json.loads(l)['task_type']
    for l in open('data/splits/libero_train.jsonl'))
print('  Train split:', dict(counts))
counts_val = Counter(json.loads(l)['task_type']
    for l in open('data/splits/libero_train_val.jsonl'))
print('  Val   split:', dict(counts_val))
"

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Building adapter JSONL"
echo "========================================================"

python -m data.annotation.motion_primitive_label \
    --data-dir  "$DATA_DIR" \
    --out        data/splits/adapter_franka.jsonl \
    --chunk-size 16 \
    --val-frac   0.1 \
    --seed       42

echo "========================================================"
echo "[$(date +'%H:%M:%S')] Done. Data ready."
echo "========================================================"
