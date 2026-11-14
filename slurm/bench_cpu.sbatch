#!/bin/bash
# One CPU step in benchmark/: CMD="python e4_kinematics.py" sbatch --job-name=<name> slurm/bench_cpu.sbatch
#SBATCH --job-name=bench-cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -e
source .venv/bin/activate
cd benchmark
export OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
eval "$CMD"
