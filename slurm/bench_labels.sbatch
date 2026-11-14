#!/bin/bash
# Training labels and crops (labels.py), one participant per array task: sbatch --array=1-29 slurm/bench_labels.sbatch
#SBATCH --job-name=bench-labels
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -e
source .venv/bin/activate
cd benchmark
export OMP_NUM_THREADS=4
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../outputs/game_vicon_aligned/reaching/video_participants.txt)
python -W ignore labels.py --participant "$PID"
