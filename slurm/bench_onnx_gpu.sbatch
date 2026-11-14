#!/bin/bash
# DWPose-L, RTMW-X, then RTMPose-m hand on RTMW-X crops (ONNX, GPU): sbatch --array=1-29 slurm/bench_onnx_gpu.sbatch
#SBATCH --job-name=bench-onnx
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%A_%a.out
set -e
source .venv/bin/activate
cd benchmark
# onnxruntime-gpu (CUDA 12 / cuDNN 9) installed with pip install --target models/pylib_ort_gpu; BENCH_ORT_GPU=1 uses it
export PYTHONPATH=../models/pylib_ort_gpu BENCH_ORT_GPU=1 OMP_NUM_THREADS=4
PID=$(sed -n "${SLURM_ARRAY_TASK_ID}p" ../outputs/game_vicon_aligned/reaching/video_participants.txt)
python zero_shot.py --participant "$PID" --models "${MODELS:-dwpose_l,rtmw_x,rtmpose_hand}" --no-blobs
