#!/usr/bin/env python3
"""Generate the Experiment 1 job list (one run_exp1.py command per line).

    python exp1_grid.py --name exp1 --folds 0-9 --seeds 0 1 2
    python exp1_grid.py --name smoke --folds 0 --seeds 0 --max_epochs 2 \
        --weights jsd_code=1000 jsd_paper=1 logdet=1

Writes
    jobs/<name>.txt   one shell command per line (line i = SLURM array task i)
    jobs/<name>.json  the same jobs with their output directories, for exp1_status.py

Every method gets the same folds, seeds, architecture and training budget;
only reg_type and lambda differ. Lambda grids are per method because the
regularizers have different scales; the analysis (plot_exp1.py) picks one
lambda per method on validation data.
"""
import argparse
import json
import shlex
import sys
from pathlib import Path

METHODS = ("none", "jsd_code", "jsd_paper", "logdet")
# Log-spaced grids around each method's natural scale. Edit with --weights.
DEFAULT_GRID = {
    "none": [0.0],
    "jsd_code": [10.0, 100.0, 1000.0, 10000.0],   # 1000 = released-code default
    "jsd_paper": [0.1, 1.0, 10.0, 100.0],
    "logdet": [0.01, 0.1, 1.0, 10.0],             # loss in [~0, -log(eps)] per layer
}


def int_list(items):
    out = []
    for item in items:
        for part in str(item).split(","):
            if "-" in part:
                a, b = part.split("-")
                out.extend(range(int(a), int(b) + 1))
            elif part:
                out.append(int(part))
    return sorted(set(out))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="exp1", help="Job list name (jobs/<name>.txt).")
    p.add_argument("--output", default=None,
                   help="Results root passed to run_exp1.py (default results/<name>).")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--weights", nargs="+", default=[], metavar="METHOD=W1,W2",
                   help="Override a lambda grid, e.g. logdet=0.1,1,10")
    p.add_argument("--folds", nargs="+", default=["0-9"], help="e.g. 0-9 or 0 1 2")
    p.add_argument("--seeds", nargs="+", default=["0-2"])
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--patience", type=int, default=300)
    p.add_argument("--python", default="python", help="Interpreter in the job commands.")
    p.add_argument("--extra", default="", help="Extra run_exp1.py arguments for every job.")
    p.add_argument("--epoch_seconds", type=float, default=12.0,
                   help="Measured seconds per epoch, for the cost estimate only.")
    args = p.parse_args()

    grid = {m: list(DEFAULT_GRID[m]) for m in args.methods}
    for item in args.weights:
        method, _, values = item.partition("=")
        if method not in grid or method == "none" or not values:
            p.error(f"Bad --weights {item!r}; use e.g. logdet=0.1,1,10 (method must be in --methods).")
        grid[method] = sorted({float(v) for v in values.split(",")})
        if any(w <= 0 for w in grid[method]):
            p.error("Weights of regularized methods must be positive.")
    folds, seeds = int_list(args.folds), int_list(args.seeds)
    output = args.output or f"results/{args.name}"

    jobs = []
    for fold in folds:                     # fold-major: early lines already cover all methods
        for seed in seeds:
            for method in args.methods:
                for w in grid[method]:
                    cmd = [args.python, "run_exp1.py", "--reg_type", method,
                           "--reg_weight", f"{w:g}", "--fold", str(fold), "--seed", str(seed),
                           "--max_epochs", str(args.max_epochs), "--patience", str(args.patience),
                           "--output", output, "--skip_existing", "--threads", "1"]
                    cmd += shlex.split(args.extra)
                    jobs.append({
                        "index": len(jobs) + 1, "reg_type": method, "weight": w,
                        "fold": fold, "seed": seed,
                        "run_dir": f"{output}/MUTAG/{method}_w{w:g}/fold{fold}_seed{seed}",
                        "command": " ".join(shlex.quote(c) for c in cmd),
                    })

    Path("jobs").mkdir(exist_ok=True)
    txt, man = Path("jobs") / f"{args.name}.txt", Path("jobs") / f"{args.name}.json"
    txt.write_text("\n".join(j["command"] for j in jobs) + "\n", encoding="utf-8")
    man.write_text(json.dumps({
        "name": args.name, "output": output, "grid": grid, "folds": folds, "seeds": seeds,
        "max_epochs": args.max_epochs, "patience": args.patience,
        "generated_with": " ".join(sys.argv), "jobs": jobs}, indent=2), encoding="utf-8")

    hours = len(jobs) * args.max_epochs * args.epoch_seconds / 3600
    print(f"{len(jobs)} jobs -> {txt}  (manifest {man})")
    for m in args.methods:
        print(f"  {m:10s} lambda in {grid[m]}  x {len(folds)} folds x {len(seeds)} seeds")
    print(f"Upper bound if no job stops early: {hours:,.1f} CPU-hours "
          f"({args.max_epochs} epochs x {args.epoch_seconds:g} s); early stopping usually cuts this.")
    print(f"Longest single job: about {args.max_epochs * args.epoch_seconds / 3600:.1f} h "
          "-> set the scheduler time limit above that.")


if __name__ == "__main__":
    main()
