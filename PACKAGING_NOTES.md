# Important: exact source versus assembled files

## Files copied directly from this conversation's user-supplied sources
- `src/layers.py` from supplied GKNN layer code.
- `src/dataset.py` from supplied GKNN dataset code.
- `src/model.py` from supplied GKNN model code, then updated to log-det and CAM.

## Files copied from the unified CAM experiment suite
- `src/cam.py`, experiment tracking, performance tracking and synthetic data.
- `run_tracked.py`, scripts, SLURM job manifests, tests, Figure 8 extension.

## Files not available for packaging
- Exact content of the current GitHub `GDA-` main branch and local uncommitted changes.
- `src/utils.py` and original `run_training.py` (the latter is represented by a wrapper).
- `exp1_*.py`, `cluster/exp1_array.slurm`, `test_regularizers.py`, and the
  current `data/` dataset/folds files listed in the user's terminal.
- Any saved experiments, trained checkpoints or synthetic `.pkl` datasets.

These files must be copied from your LOCAL `GDA-` repository to make a full,
faithful copy. No network download or GitHub content was fabricated.

## Safer alternative for an exact complete ZIP (PowerShell)
From the parent folder of `GDA-`:

    Compress-Archive -Path .\GDA- -DestinationPath .\GDA-_exact_local.zip -Force

You can add this experiment suite to the local repository before zipping.

## Validation
The source files are syntactically checked when the ZIP is created. Integrated
end-to-end training with GraKeL/PyG must be tested in your actual environment.
