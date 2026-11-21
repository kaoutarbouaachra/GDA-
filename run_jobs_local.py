#!/usr/bin/env python3
"""Run a job list in parallel on one machine (laptop, or one cluster node).

    python run_jobs_local.py jobs/smoke.txt --workers 4
    python run_jobs_local.py jobs/exp1.txt --workers 32 --lines 1-100

Each job's stdout/stderr goes to logs/<list name>/job_<line>.log, and one
line per started/finished job is printed here. Jobs use --skip_existing, so
re-running the same list only runs what is missing or failed.
Use exp1_status.py (another terminal) to see per-run epochs and ETA.
"""
import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def parse_lines(spec, n):
    if not spec:
        return list(range(1, n + 1))
    a, _, b = spec.partition("-")
    return list(range(int(a), int(b or a) + 1))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("jobs", type=Path)
    p.add_argument("--workers", type=int, default=2,
                   help="Parallel jobs; use at most the number of physical cores.")
    p.add_argument("--lines", default="", help="Subset of lines, e.g. 1-50 (1-based).")
    args = p.parse_args()

    commands = [c for c in args.jobs.read_text(encoding="utf-8").splitlines()]
    selected = [i for i in parse_lines(args.lines, len(commands)) if commands[i - 1].strip()]
    log_dir = Path("logs") / args.jobs.stem
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(selected)} jobs, {args.workers} workers, logs in {log_dir}", flush=True)

    def run(i):
        start = time.time()
        with (log_dir / f"job_{i:04d}.log").open("w", encoding="utf-8") as log:
            log.write(f"# {commands[i - 1]}\n")
            log.flush()
            code = subprocess.call(commands[i - 1], shell=True, stdout=log,
                                   stderr=subprocess.STDOUT)
        return i, code, time.time() - start

    failed, done = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run, i): i for i in selected}
        for future in as_completed(futures):
            i, code, seconds = future.result()
            done += 1
            state = "ok" if code == 0 else f"FAILED (exit {code})"
            if code:
                failed.append(i)
            print(f"[{time.strftime('%H:%M:%S')}] {done}/{len(selected)}  job {i}: {state} "
                  f"after {seconds / 60:.1f} min  log: {log_dir / f'job_{i:04d}.log'}", flush=True)
    print(f"Finished: {len(selected) - len(failed)} ok, {len(failed)} failed"
          + (f" (lines {failed})" if failed else ""))
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
