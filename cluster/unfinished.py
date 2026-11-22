"""Print the array indices of a manifest whose run has no DONE.json.

Use to resubmit only interrupted/failed jobs; interrupted runs resume from
checkpoints/last.ckpt (run_tracked.py --resume, the default):

  sbatch --array=$(python cluster/unfinished.py --manifest M) cluster/array.slurm M

With --selected R, checks stage-2 selections (SELECTED_DONE.json) instead.
Prints nothing (and exits 0) when everything is finished.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'cluster'))


def compress(ids):
    """[0,1,2,5,7,8] -> '0-2,5,7-8' (SLURM --array syntax)."""
    out, start, prev = [], None, None
    for i in ids:
        if start is None:
            start = prev = i
        elif i == prev + 1:
            prev = i
        else:
            out.append(f'{start}-{prev}' if prev > start else str(start))
            start = prev = i
    if start is not None:
        out.append(f'{start}-{prev}' if prev > start else str(start))
    return ','.join(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', required=True)
    p.add_argument('--selected', default=None, help='Stage-2 output root (select_jsd.py)')
    p.add_argument('--vary', nargs='+', default=['jsd_weight'])
    p.add_argument('--verbose', action='store_true')
    a = p.parse_args()
    rows = [json.loads(x) for x in open(a.manifest, encoding='utf-8') if x.strip()]
    missing = []
    total = len(rows)
    if a.selected:
        import select_jsd as s
        groups = s.groups_from_manifest(rows, a.vary)
        total = len(groups)
        for i, (key, _) in enumerate(groups.items()):
            fixed = json.loads(key)
            d = Path(a.selected) / fixed['dataset'] / fixed['model'] / s.group_name(fixed, a.vary)
            d = d if d.is_absolute() else ROOT / d
            if not (d / 'SELECTED_DONE.json').exists():
                missing.append(i)
    else:
        for i, row in enumerate(rows):
            d = Path(row['args']['output'])
            d = d if d.is_absolute() else ROOT / d
            if not (d / 'DONE.json').exists():
                missing.append(i)
                if a.verbose:
                    state = 'FAILED' if (d / 'FAILED.json').exists() else (
                        'interrupted' if (d / 'config.json').exists() else 'not started')
                    print(i, row['name'], state, file=sys.stderr)
    print(compress(missing))
    print('%d of %d unfinished' % (len(missing), total), file=sys.stderr)


if __name__ == '__main__':
    main()
