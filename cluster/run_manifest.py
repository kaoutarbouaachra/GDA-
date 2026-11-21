"""Execute one manifest line. Idempotent if the run has DONE.json.

Used by SLURM array and can run locally with --index 0.
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def command_from_job(job,python_executable=None):
    args=job['args']
    cmd=[python_executable or sys.executable,'run_tracked.py']
    for key,val in args.items():
        if isinstance(val,bool):
            if val:
                cmd.append('--'+key)
        else:
            cmd.extend(['--'+key,str(val)])
    return cmd


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--index',type=int,required=True)
    p.add_argument('--dry_run',action='store_true')
    args=p.parse_args()
    with open(args.manifest,encoding='utf-8') as f:
        rows=[json.loads(x) for x in f if x.strip()]
    if args.index<0 or args.index>=len(rows):
        raise SystemExit('Array index outside manifest: '+str(args.index))
    job=rows[args.index]
    dest=Path(job['args']['output'])
    if (dest/'DONE.json').exists():
        print('SKIP completed:',job['name'])
        return
    cmd=command_from_job(job)
    print('JOB',job['id'],job['name'],flush=True)
    print(' '.join(cmd),flush=True)
    if args.dry_run:
        return
    # Use unique stdout/err paths; no collision between array tasks.
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'job_stdout.log').open('w',encoding='utf-8') as log:
        p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=os.environ.copy(),check=False)
    print('Completed with exit code',p.returncode,flush=True)
    if p.returncode!=0:
        print('See:',dest/'job_stdout.log')
        raise SystemExit(p.returncode)


if __name__=='__main__':
    main()
