#!/bin/bash
# Zero-shot PyTorch model, after bench_zero_shot: MODEL=sapiens2_1b sbatch --array=1-29 slurm/bench_gpu_model.sbatch
#SBATCH --job-name=bench-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -e
source .venv/bin/activate
cd benchmark
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=4
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../outputs/game_vicon_aligned/reaching/video_participants.txt)
python zero_shot.py --participant "$PID" --models "$MODEL" --no-blobs
