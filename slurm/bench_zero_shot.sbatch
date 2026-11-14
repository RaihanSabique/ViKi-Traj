#!/bin/bash
# Person boxes, zero-shot ViTPose+-B and RTMPose-WB, marker blobs: sbatch --array=1-29 slurm/bench_zero_shot.sbatch
#SBATCH --job-name=bench-zeroshot
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -e
source .venv/bin/activate
cd benchmark
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../outputs/game_vicon_aligned/reaching/video_participants.txt)
python zero_shot.py --participant "$PID" --models "${MODELS:-vitpose_b,rtmpose_wb}"
