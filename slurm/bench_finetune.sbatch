#!/bin/bash
# ViconTraj-MS fine-tuning run: RUN=<name> ARGS="--views top,side --mode full ..." sbatch slurm/bench_finetune.sbatch
#SBATCH --job-name=bench-ft
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -e
source .venv/bin/activate
cd benchmark
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=2
python -W ignore finetune.py --run "$RUN" $ARGS
