#!/bin/bash
# Top and side videos aligned to the game clock and cut into clips: sbatch --array=1-29 slurm/video_align.sbatch
#SBATCH --job-name=video-align
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -e
source .venv/bin/activate
export OMP_NUM_THREADS=4
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" outputs/game_vicon_aligned/reaching/video_participants.txt)
python alignment/video/align_videos.py --participant "$PID"
