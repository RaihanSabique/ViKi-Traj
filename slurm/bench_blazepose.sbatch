#!/bin/bash
# Zero-shot BlazePose+Hands on CPU, one participant per array task: sbatch --array=1-29 slurm/bench_blazepose.sbatch
#SBATCH --job-name=bench-blazepose
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -e
source .venv/bin/activate
cd benchmark
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../outputs/game_vicon_aligned/reaching/video_participants.txt)
python zero_shot.py --participant "$PID" --models blazepose --no-blobs
