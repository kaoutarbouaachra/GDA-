#!/bin/bash
# SLURM job array for Experiment 1: array task i runs line i of the job list.
#
# Submit from the repository root (see EXPERIMENT1.md):
#   mkdir -p logs/slurm                                        # SLURM does not create it
#   python run_exp1.py --reg_type none --prepare_only          # once, before submitting
#   python exp1_grid.py --name exp1                            # -> jobs/exp1.txt
#   N=$(wc -l < jobs/exp1.txt)
#   sbatch --array=1-${N}%100 --export=ALL,JOBS=jobs/exp1.txt cluster/exp1_array.slurm
#   (%100 = at most 100 tasks at once; adapt to your cluster's fair-share rules)
#
# Resubmit only what failed / was killed / never ran:
#   sbatch --array=$(python exp1_status.py jobs/exp1.json --failed-lines) \
#          --export=ALL,JOBS=jobs/exp1.txt cluster/exp1_array.slurm
#
# One GKNN run is single-threaded CPU work (GraKeL kernels): no GPU needed.
#SBATCH --job-name=gknn-exp1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/%x_%A_%a.out
#SBATCH --error=logs/slurm/%x_%A_%a.out
# Uncomment / adapt to your cluster:
##SBATCH --partition=cpu
##SBATCH --account=YOUR_ACCOUNT

set -uo pipefail   # no -e: we want to log the exit code below
cd "${SLURM_SUBMIT_DIR}"

# ---- Python environment (EDIT: one of these) ---------------------------------
# module load python/3.11
# source .venv/bin/activate
# conda activate gknn
# -------------------------------------------------------------------------------

JOBS="${JOBS:-jobs/exp1.txt}"
CMD="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "${JOBS}")"
if [ -z "${CMD}" ]; then
  echo "No line ${SLURM_ARRAY_TASK_ID} in ${JOBS}"; exit 1
fi

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
echo "[$(date '+%F %T')] job ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} on $(hostname)"
echo "[$(date '+%F %T')] ${CMD}"
eval "${CMD}"
status=$?
echo "[$(date '+%F %T')] exit ${status}"
exit ${status}
