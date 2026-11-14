#!/bin/bash
# Pilot: pose models on every aligned trial of one visit video: VIDEO=<video stem> sbatch slurm/pilot_models.sbatch
#SBATCH --job-name=pilot-models
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -e
source .venv/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=16
python pilot/run_models.py --participant ${PARTICIPANT:-XXXXXXX115} --video ${VIDEO}
