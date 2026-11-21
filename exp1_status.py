#!/usr/bin/env python3
"""Progress of an Experiment 1 job list: what is done, running, stalled or failed.

    python exp1_status.py jobs/exp1.json            # summary + problems
    python exp1_status.py jobs/exp1.json --all      # one line per run
    python exp1_status.py jobs/exp1.json --watch 60 # refresh every 60 s
    python exp1_status.py jobs/exp1.json --failed-lines   # e.g. for sbatch --array=...

Reads each run's status.json (written by run_exp1.py at start/end/failure)
and progress.json (updated every epoch). A run is "stalled" if it says
running but has not logged an epoch for --stale minutes (typically killed by
the scheduler's time limit or memory limit: check its run.log / slurm log).
Also writes <jobs>.status.csv.
"""
import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path


def load(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def state_of(job, stale_minutes):
    run_dir = Path(job["run_dir"])
    status, progress = load(run_dir / "status.json"), load(run_dir / "progress.json")
    if (run_dir / "diagnostics.json").exists() and status.get("state", "done") == "done":
        state = "done"
    elif status.get("state") == "failed":
        state = "failed"
    elif run_dir.exists():
        stamp = max((p.stat().st_mtime for p in (run_dir / "progress.json", run_dir / "status.json",
                                                 run_dir / "run.log") if p.exists()), default=0)
        state = "running" if time.time() - stamp < stale_minutes * 60 else "stalled"
    else:
        state = "pending"
    return state, status, progress


def fmt_hours(seconds):
    if seconds != seconds or seconds is None:
        return "-"
    return f"{seconds / 3600:.1f}h" if seconds >= 3600 else f"{seconds / 60:.0f}m"


def report(manifest_path, stale, show_all):
    manifest = load(manifest_path)
    if not manifest:
        raise SystemExit(f"Cannot read {manifest_path} (generate it with exp1_grid.py).")
    rows = []
    for job in manifest["jobs"]:
        state, status, progress = state_of(job, stale)
        epoch, max_ep = progress.get("epoch"), progress.get("max_epochs", manifest.get("max_epochs"))
        mean_ep = progress.get("mean_epoch_seconds")
        eta = (max_ep - epoch) * mean_ep if state == "running" and epoch and mean_ep else float("nan")
        rows.append({
            "line": job["index"], "reg_type": job["reg_type"], "weight": job["weight"],
            "fold": job["fold"], "seed": job["seed"], "state": state,
            "epoch": epoch, "max_epochs": max_ep, "mean_epoch_s": mean_ep,
            "best_val_loss": progress.get("best_val_loss"), "val_acc": progress.get("val_acc"),
            "final_val_acc": status.get("val_accuracy"), "best_epoch": status.get("best_epoch"),
            "stopped_early": status.get("stopped_early"), "wall_s": status.get("wall_seconds"),
            "eta_upper_s": eta, "host": status.get("host"), "updated": progress.get("updated"),
            "error": status.get("error", ""), "run_dir": job["run_dir"],
        })

    print(f"\n=== {manifest.get('name')} | {time.strftime('%Y-%m-%d %H:%M:%S')} | "
          f"{len(rows)} jobs -> {manifest.get('output')} ===")
    counts = Counter(r["state"] for r in rows)
    print("  " + "  ".join(f"{s}: {counts.get(s, 0)}" for s in
                           ("done", "running", "stalled", "failed", "pending")))
    by_cfg = defaultdict(list)
    for r in rows:
        by_cfg[(r["reg_type"], r["weight"])].append(r)
    print(f"\n  {'method':10s} {'lambda':>8s}  done/total  running  failed  "
          f"mean s/epoch  mean val_acc(done)")
    for (method, w), group in by_cfg.items():
        c = Counter(r["state"] for r in group)
        speeds = [r["mean_epoch_s"] for r in group if r["mean_epoch_s"]]
        accs = [100 * r["final_val_acc"] for r in group if r["final_val_acc"] is not None]
        print(f"  {method:10s} {w:8g}  {c['done']:4d}/{len(group):<5d}  {c['running']:7d}  "
              f"{c['failed'] + c['stalled']:6d}  "
              f"{(sum(speeds) / len(speeds)) if speeds else float('nan'):12.1f}  "
              f"{(sum(accs) / len(accs)) if accs else float('nan'):10.1f}")

    running = [r for r in rows if r["state"] == "running"]
    if running:
        etas = [r["eta_upper_s"] for r in running if r["eta_upper_s"] == r["eta_upper_s"]]
        print(f"\n  Running: {len(running)}; longest remaining (if no early stop): "
              f"{fmt_hours(max(etas)) if etas else '-'}")
    problems = [r for r in rows if r["state"] in ("failed", "stalled")]
    if problems:
        print("\n  Problems (see run.log in the run directory):")
        for r in problems[:30]:
            print(f"    line {r['line']:4d} {r['state']:7s} {r['reg_type']} w={r['weight']:g} "
                  f"fold={r['fold']} seed={r['seed']}  epoch={r['epoch']}  "
                  f"{(r['error'] or '')[:100]}\n      {r['run_dir']}/run.log")
        if len(problems) > 30:
            print(f"    ... and {len(problems) - 30} more (see the CSV)")
    if show_all:
        print("\n  line state    method     lambda fold seed epoch      best_val  val_acc  host")
        for r in rows:
            print(f"  {r['line']:4d} {r['state']:8s} {r['reg_type']:10s} {r['weight']:6g} "
                  f"{r['fold']:4d} {r['seed']:4d} {str(r['epoch'] or '-'):>5s}/{r['max_epochs']}  "
                  f"{r['best_val_loss'] if r['best_val_loss'] is not None else '-':>8}  "
                  f"{r['val_acc'] if r['val_acc'] is not None else '-':>7}  {r['host'] or ''}")

    out = manifest_path.with_suffix(".status.csv")
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n  Full table: {out}")
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("manifest", type=Path, help="jobs/<name>.json from exp1_grid.py")
    p.add_argument("--stale", type=float, default=30,
                   help="Minutes without a new epoch before a running job counts as stalled.")
    p.add_argument("--all", action="store_true", help="Print one line per run.")
    p.add_argument("--watch", type=float, default=0, help="Refresh every N seconds.")
    p.add_argument("--failed-lines", action="store_true",
                   help="Only print the job lines to resubmit (failed, stalled, pending).")
    args = p.parse_args()
    if args.failed_lines:
        manifest = load(args.manifest)
        lines = [str(j["index"]) for j in manifest["jobs"]
                 if state_of(j, args.stale)[0] in ("failed", "stalled", "pending")]
        print(",".join(lines))
        return
    while True:
        report(args.manifest, args.stale, args.all)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
