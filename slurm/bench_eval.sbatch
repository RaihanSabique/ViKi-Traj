#!/bin/bash
# Camera calibration (with CALIBRATE=1) and E1 / E2 evaluation: CALIBRATE=1 sbatch slurm/bench_eval.sbatch
#SBATCH --job-name=bench-eval
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -e
source .venv/bin/activate
cd benchmark
export OMP_NUM_THREADS=16
if [ "${CALIBRATE:-0}" = "1" ]; then python calibrate.py; fi
python evaluate.py --task all
