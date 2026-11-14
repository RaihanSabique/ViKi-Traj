#!/bin/bash
# E3, fingertip trajectory on the screen against the game trajectory, all model outputs: sbatch slurm/bench_e3.sbatch
#SBATCH --job-name=bench-e3
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -e
source .venv/bin/activate
cd benchmark
export OMP_NUM_THREADS=16
python e3_screen.py
