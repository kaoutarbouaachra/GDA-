#!/bin/bash
#SBATCH --job-name=GDA-CAM
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=12:00:00
#SBATCH --output=cluster/logs/%A_%a.out
#SBATCH --error=cluster/logs/%A_%a.err
# Intentionally no hardcoded partition/account or array range.
# Submit with: sbatch --array=0-N%8 cluster/array.slurm cluster/jobs.jsonl
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repo root}"
MANIFEST="${1:-cluster/jobs.jsonl}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Activate YOUR cluster environment before this line if needed:
# source /path/to/venv/bin/activate
# or: module load ...
python cluster/run_manifest.py --manifest "$MANIFEST" --index "${SLURM_ARRAY_TASK_ID:?}"
