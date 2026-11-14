#!/bin/bash
# Evaluation chain E3 -> E1 / E2 -> E4 -> E6 -> E7 over all model outputs: sbatch slurm/bench_final.sbatch
#SBATCH --job-name=bench-final
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -e
source .venv/bin/activate
cd benchmark
export OMP_NUM_THREADS=16 PYTHONWARNINGS=ignore
step(){ echo "=== $(date '+%F %T') $*"; "$@"; }
# E3 first: it writes the trajectory QA that benchlib.manifest() applies in the later steps
step python e3_screen.py
step python evaluate.py --task all
step python e4_kinematics.py
step python e6_analyses.py
step python e7_qualitative.py
echo "=== $(date '+%F %T') done"
