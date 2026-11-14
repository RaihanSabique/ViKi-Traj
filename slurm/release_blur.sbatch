#!/bin/bash
# Release clips (side head-blurred), task N -> sub-N, env OUT, BLUR_ARGS: sbatch --array=1-29 slurm/release_blur.sbatch
#SBATCH --job-name=release-blur
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -e
source .venv/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
PID=$(python -c "import sys; sys.path.insert(0, 'dataset'); import common; print(common.pids()[$SLURM_ARRAY_TASK_ID - 1])")
python dataset/blur_clips.py --participant "$PID" --out "${OUT:-release/ViKi-Traj}" ${BLUR_ARGS}
