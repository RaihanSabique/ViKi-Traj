#!/bin/bash
# One GPU step in benchmark/: CMD="python e5_seq.py --part seq2d --base sapiens2_1b" sbatch slurm/bench_gpu_step.sbatch
#SBATCH --job-name=bench-gpu-step
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -e
source .venv/bin/activate
cd benchmark
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=6 PYTHONWARNINGS=ignore
eval "$CMD"
