#!/bin/bash
# Protocol v2: GKNN + JSD vs CAM + JSD (see README_START_HERE.md, section 9).
# Generates the synthetic data and ALL stage-1 manifests. Submits nothing.
# Run from the repository root. New output folders only: results/v2_*.
set -euo pipefail
cd "$(dirname "$0")/.."

LAMBDAS="100 1000 10000"
COMMON="--reg_types jsd_code --lambda_values $LAMBDAS --hidden 8 --nodes 6 --hops 1 --kernel wl \
 --pooling add --max_epochs 400 --patience 75 --batch_size 16 --cam_samples 4 --cam_eval_samples 16 \
 --cam_multilayer_inputs gknn --no-benchmark_inference"
SWEEP="--reg_type jsd_code --lambda_values $LAMBDAS --fold_ids 0 1 2 --seeds 0 1 2 --max_epochs 400 \
 --patience 75 --cam_multilayer_inputs gknn --no-benchmark_inference"

echo "== 1. Main comparison: MUTAG, PTC, PROTEINS x 2 models x L{1,2,3} x 3 lambdas x 10 folds x 3 seeds"
# One manifest per dataset (540 jobs, indices 0-539): stays below SLURM's default
# MaxArraySize (1001) and lets each dataset get its own --time.
for DS in MUTAG PTC PROTEINS; do
  python cluster/make_manifest.py --datasets $DS --models original cam \
    --layers 1 2 3 --folds 10 --seeds 0 1 2 $COMMON \
    --manifest cluster/v2_main_$DS.jsonl --output_root results/v2_main_grid
done

echo "== 2. Figure 8: five paired single-motif tasks (FIG8_RING ... FIG8_CLIQUES), L=1"
# Paired positive/negative graphs and grouped folds (scripts/generate_fig8_paired.py).
# Masks of <= 10 nodes fit every motif; each motif uses the smallest egonet radius
# that contains it (graph radius): ring 4, wheel 1, grid 2, ladder 3, cliques 2.
if [ -f synthetic/fig8_datasets.json ]; then
  echo "   (synthetic/fig8_* already generated, reusing them)"
else
  python scripts/generate_fig8_paired.py --n_graphs 2000 --seed 7 --n_folds 10 \
    --total_min 30 --total_max 50 --p_base 0.10 --p_attach 0.02 \
    --out_dir synthetic --fold_dir data/folds
fi
for MR in RING:4 WHEEL:1 GRID:2 LADDER:3 CLIQUES:2; do
  M=${MR%%:*}; R=${MR##*:}
  python cluster/make_manifest.py --datasets FIG8_$M --synthetic_mapping synthetic/fig8_datasets.json \
    --models original cam --layers 1 --folds 10 --fold_ids 0 1 2 --seeds 0 1 2 \
    --reg_types jsd_code --lambda_values $LAMBDAS --hidden 8 --nodes 10 --hops $R --kernel wl --pooling add \
    --max_epochs 400 --patience 75 --batch_size 16 --cam_samples 4 --cam_eval_samples 16 --no-benchmark_inference \
    --manifest cluster/v2_fig8_$M.jsonl --output_root results/v2_fig8_grid
done

echo "== 3. Mask-count ablation (PROTEINS, L=2)"
python cluster/make_sweeps.py --sweep mask_count --dataset PROTEINS --models original cam --layers 2 $SWEEP \
  --manifest cluster/v2_masks.jsonl --output_root results/v2_ablations_grid

echo "== 4. Depth ablation (PROTEINS, L in {1,2,3,5,7}, r=1, both models)"
python cluster/make_sweeps.py --sweep depth --dataset PROTEINS --models original cam $SWEEP \
  --manifest cluster/v2_depth.jsonl --output_root results/v2_ablations_grid

echo "== 5. CAM Monte Carlo budget (MUTAG, L=1)"
python cluster/make_sweeps.py --sweep cam_mc_samples --dataset MUTAG $SWEEP \
  --manifest cluster/v2_mc.jsonl --output_root results/v2_ablations_grid

echo
echo "== Stage-2 selection groups"
for m in main_MUTAG main_PTC main_PROTEINS fig8_RING fig8_WHEEL fig8_GRID fig8_LADDER fig8_CLIQUES masks depth mc; do
  printf '%-10s ' "$m"; python cluster/select_jsd.py --grid_manifest cluster/v2_$m.jsonl --list | tail -1
done
