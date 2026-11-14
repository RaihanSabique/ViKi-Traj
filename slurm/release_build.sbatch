#!/bin/bash
# Release builders (core, sourcedata, derivatives, frame tables, docs) into OUT: sbatch slurm/release_build.sbatch
#SBATCH --job-name=release-build
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -e
source .venv/bin/activate
export HF_HUB_OFFLINE=1 OMP_NUM_THREADS=8
OUT=${OUT:-release/ViKi-Traj}
for b in build_core build_sourcedata build_derivatives build_frame_tables; do
  echo "== $b $(date +%T)"
  python dataset/$b.py --all --out "$OUT"
done
python dataset/build_docs.py --out "$OUT" || echo "build_docs.py not run"
echo "== done $(date +%T)"
