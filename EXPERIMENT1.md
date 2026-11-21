# Experiment 1 — Do diversity regularizers reduce redundancy between GKNN masks?

Four conditions, trained with **identical** splits, seeds (hence identical initial
weights), architecture, training budget and evaluation protocol. Only the
regularizer and its weight λ differ.

| `--reg_type` | Training objective | Defined in |
|---|---|---|
| `none` | CE | — |
| `jsd_code` | CE − λ · mean_ℓ JSD(softmax over filters of R^(ℓ)) | `src/model.py: jsd_code` (released code) |
| `jsd_paper` | CE − λ · mean_ℓ JSD(p_i(v) = R_vi / Σ_{u∈G} R_ui, per graph) | `src/model.py: jsd_paper` (Cosmo et al.) |
| `logdet` | CE + λ · mean_ℓ L_logdet(Z^(ℓ)), L_logdet(Z) = −(1/m) log det(Ĉ + ε I_m) | `src/model.py: logdet_penalty` (ours) |

`mean_ℓ` is the average over the L kernel layers (λ/L Σ_ℓ). Before training, every run checks
that `model.regularizers` matches an independent re-implementation of its formula (value and gradient).

**LogDet.** For a layer's node responses Z ∈ R^{N×m} (one column per filter): centre each column,
divide it by its norm (→ U), Ĉ = UᵀU (Pearson correlations). The log-determinant is computed as
2 Σ log diag(chol(Ĉ + εI)) in float64, never as a determinant. ε = `--logdet_eps` (default 1e-4).
The loss is ≥ −log(1+ε) ≈ 0 (uncorrelated filters) and ≤ −log ε ≈ 9.21; every near-zero eigenvalue
of Ĉ (a redundant direction) costs up to −log(ε)/m. It penalises the whole spectrum, not pairs:
pairwise correlations are **only evaluation metrics** here, no method is trained on them.

*Collapsed filters* (response std over the batch < 1e-6): their correlation is undefined, so their
column of U is set to 0. They then cost exactly −log(ε)/m each in the loss value (so collapse is not
"free", unlike a pairwise-correlation loss) but receive no gradient from it (the decision is not
differentiated; dividing only by norms above the threshold keeps gradients bounded). Their share is
logged every epoch (`frac_const_filters`, `const=` in run.log), together with the smallest eigenvalue
of Ĉ over active filters (`corr_min_eig`, `min_eig=`; ≈ 0 means nearly singular / redundant).
`python test_regularizers.py` checks closed-form values, finite-difference gradients (`gradcheck`)
and finite values/gradients for singular Ĉ and (nearly) constant filters.

λ is **tuned separately for each method on validation data** (the scales differ by orders of
magnitude). The test folds are never used in this experiment. ε is part of the protocol (same value
for every run; change it for all jobs with `exp1_grid.py --extra "--logdet_eps 1e-3"`).

## Files (all at the repository root)

| File | Role |
|---|---|
| `run_exp1.py` | One run = one (method, λ, fold, seed). Trains, reloads the best-validation checkpoint, exports diagnostics. |
| `exp1_diagnostics.py` | Correlations, effective rank, constant filters; node level and graph level. `--self-test` checks the maths. |
| `exp1_structure.py` | Decodes the learned masks and computes their WL similarity. |
| `exp1_grid.py` | Writes the job list (`jobs/<name>.txt`, one command per line) and a manifest (`jobs/<name>.json`). |
| `run_jobs_local.py` | Runs a job list in parallel on one machine. |
| `cluster/exp1_array.slurm` | SLURM job array: task *i* runs line *i* of the job list. |
| `exp1_status.py` | Live progress: done / running / stalled / failed / pending, epochs, ETA, errors. |
| `plot_exp1.py` | Selects λ per method, then writes tables and figures. |

## What each run writes

`results/<name>/MUTAG/<method>_w<λ>/fold<f>_seed<s>/`

| File | Content |
|---|---|
| `run.log` | Everything the run printed, incl. **one line per epoch** (losses, val loss/acc, best so far, constant filters, s/epoch). |
| `status.json` | `running` → `done` / `failed` (+ error and traceback), host, SLURM ids, wall time, best epoch, early stop. |
| `progress.json` | Updated every epoch (epoch, s/epoch, val loss, best val loss): what `exp1_status.py` reads. |
| `config.json` | Full protocol, package versions, source hashes, split ids, initial-weight hash. |
| `training/version_0/metrics.csv` | Per-epoch training/validation metrics (CSV). |
| `checkpoints/` | Best-validation and last checkpoints. |
| `diagnostics.json`, `correlations.npz`, `responses.npz` | Validation response statistics, correlation matrices, spectra, raw responses. |
| `structural.npz`, `masks.json` | WL similarity between decoded masks; the masks themselves. |

## 1. Check on your machine (minutes)

```bash
python exp1_diagnostics.py --self-test            # metric maths
python test_regularizers.py                       # the three losses
python run_exp1.py --reg_type none --prepare_only # build processed data + splits once
python exp1_grid.py --name smoke --folds 0 --seeds 0 1 --max_epochs 2 --patience 2 \
       --weights jsd_code=100,1000 jsd_paper=1 logdet=0.1,1
python run_jobs_local.py jobs/smoke.txt --workers 4
python exp1_status.py jobs/smoke.json --all
python plot_exp1.py --root results/smoke --layer 1 2
```

## 2. Run on the cluster

```bash
# once, on the cluster, from the repository root, inside your Python environment
mkdir -p logs/slurm
python run_exp1.py --reg_type none --prepare_only      # avoids parallel jobs racing on data/
python exp1_grid.py --name exp1                        # default grid, see below
# edit the "Python environment" block (and partition/account) in cluster/exp1_array.slurm
N=$(wc -l < jobs/exp1.txt)
sbatch --array=1-${N}%100 --export=ALL,JOBS=jobs/exp1.txt cluster/exp1_array.slurm
```

Follow it:

```bash
squeue -u $USER                                  # scheduler view
python exp1_status.py jobs/exp1.json --watch 300 # experiment view (refresh 5 min)
tail -f results/exp1/MUTAG/logdet_w1/fold0_seed0/run.log   # one run, epoch by epoch
ls logs/slurm/                                   # scheduler-side logs (one per array task)
```

Resubmit what failed, was killed (time/memory limit) or never ran. Completed runs are
skipped; an unfinished run directory is renamed `*_incomplete_<time>` (never deleted) and restarted:

```bash
sbatch --array=$(python exp1_status.py jobs/exp1.json --failed-lines) \
       --export=ALL,JOBS=jobs/exp1.txt cluster/exp1_array.slurm
```

Without SLURM (PBS, a single big node, …): `python run_jobs_local.py jobs/exp1.txt --workers <cores>`;
logs go to `logs/exp1/job_XXXX.log`, and runs can be split with `--lines 1-100`.

### Default grid and budget

| Method | λ grid |
|---|---|
| `none` | 0 |
| `jsd_code` | 10, 100, **1000** (released default), 10000 |
| `jsd_paper` | 0.1, 1, 10, 100 |
| `logdet` | 0.01, 0.1, 1, 10 |

13 configurations × 10 folds × 3 seeds = **390 jobs**, max 1000 epochs, early stopping with
patience 300 (the original GKNN settings). On this laptop one epoch takes about 12–35 s, so one
job takes **up to ~3–10 h** and the whole grid **at most ~1,300–3,800 CPU-hours** (less with early
stopping). Each job needs 1 CPU, no GPU (the graph kernels are CPU-only GraKeL code). Measure the
s/epoch on your cluster with `exp1_status.py` after the first jobs, and set `--time` accordingly.

To spend less: fewer folds/seeds first (`--folds 0-4 --seeds 0`), or a narrower λ grid
(`--weights logdet=0.1,1 jsd_paper=1,10`), or `--max_epochs 300 --patience 50` for a pilot.

## 3. Analysis

```bash
python plot_exp1.py --root results/exp1 --layer 1 2
```

For each layer and each **view**, it writes `results/exp1/analysis_layer<L>_<view>/`:

| View | Rows of the matrix | Meaning |
|---|---|---|
| `raw` | validation nodes | GKernel responses before the learned mask gain |
| `scaled` | validation nodes | after the gain (what the model uses) |
| `pooled_raw` | validation graphs | add-pooled raw responses |
| `pooled_scaled` | validation graphs | add-pooled scaled responses = the block of the representation fed to the MLP |

Node level answers "do masks respond to the same substructures?"; graph level answers "is the
graph representation redundant?". They can disagree (e.g. add pooling makes every filter grow with
graph size). The graph-level effective rank is capped by (number of validation graphs − 1).

Outputs: `tuning.csv` (every method × λ), `metrics.csv` (every run), `summary.csv` (selected λ,
matched runs), `paired_differences.csv` (method − none, same fold/seed/initial weights),
`analysis_info.json`, and figures (PDF + PNG):

* `correlation_heatmaps` — |correlation| between filters, 4 methods side by side (gray = constant filter)
* `normalized_eigenspectra` — spectrum of the correlation matrix
* `paired_metrics` — mean |corr|, mean corr², effective rank, inactive filters, accuracy
* `accuracy_vs_diversity` — accuracy against mean |corr| and effective rank, every run
* `lambda_tuning` — validation CE / accuracy / diversity as a function of λ
* `structural_wl_heatmaps` — WL similarity between the decoded masks

### Reading the results

* Constant filters have no correlation: they are **excluded** from correlations and effective rank
  and **counted separately** (`constant_filters`). A method can lower correlations by switching
  filters off; always read the two together. The `logdet` loss charges −log(ε)/m per collapsed
  filter but gives it no gradient, so check `constant_filters` for it too.
* λ is chosen on the same validation graphs that are summarized, so validation numbers are
  optimistic for every regularized method. Final accuracy claims need the test folds.
* No relation between diversity and accuracy is assumed: `accuracy_vs_diversity` shows the
  data; it is not evidence that one causes the other.
