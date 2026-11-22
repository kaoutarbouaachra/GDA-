#!/bin/bash
#SBATCH --job-name=GDA-JSD-select
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=cluster/logs/select_%A_%a.out
#SBATCH --error=cluster/logs/select_%A_%a.err
# Stage 2 of the JSD protocol: one array task = one (dataset, model, fold, seed).
# Picks lambda on validation only, then evaluates the test split once.
# Submit AFTER the stage-1 grid, e.g. with --dependency=afterany:<grid_job_id>:
#   sbatch --array=0-N cluster/select_array.slurm cluster/jobs_jsd_grid.jsonl results/jsd_selected
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from repo root}"
MANIFEST="${1:?grid manifest}"
OUTPUT_ROOT="${2:-results/jsd_selected}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
# Activate the SAME environment as in cluster/array.slurm before this line:
# source /path/to/venv/bin/activate
# or: module load ...
python cluster/select_jsd.py --grid_manifest "$MANIFEST" --output_root "$OUTPUT_ROOT" --group_index "${SLURM_ARRAY_TASK_ID:?}"
