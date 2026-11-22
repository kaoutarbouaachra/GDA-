"""Stage 2: choose hyperparameters on VALIDATION only, then evaluate TEST once.

Stage 1 (cluster/make_manifest.py / make_sweeps.py with --reg_types jsd_code
--lambda_values ...) trains every candidate WITHOUT touching the test split.
This script groups those runs: inside a group only the --vary hyperparameters
(default: jsd_weight) and the output directory differ. For one group it:

  1. requires every candidate run to be finished (DONE.json) and test-free;
  2. selects the candidate whose best checkpoint has the LOWEST VALIDATION
     cross-entropy, recomputed exactly over all validation graphs
     (summary.json -> outputs.val.cross_entropy). Ties: higher validation
     accuracy, then smaller layers, then smaller lambda. Only validation
     numbers are read; selection.json is written BEFORE the test evaluation;
  3. copies the chosen run to --output_root and only then evaluates the
     held-out test split at that checkpoint (+ filter ablation, inference cost).

Same grid, criterion and code for GKNN and CAM; one choice per
(dataset, model, configuration, fold, seed).

Depth selection. With the main grid layers x lambda, select lambda per depth
(default --vary jsd_weight): this gives the fixed-depth results L=1,2,3. Then
`--summarize --select_over layers` picks, per (dataset, model, fold, seed), the
depth whose lambda-winner has the lowest validation cross-entropy. This is
exactly the joint (layers, lambda) validation minimum, and reads no test value.

Usage (from the repository root):
  python cluster/select_jsd.py --grid_manifest M --list
  python cluster/select_jsd.py --grid_manifest M --group_index 0 --output_root R
  python cluster/select_jsd.py --output_root R --summarize [--select_over layers]
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
from argparse import Namespace
from collections import OrderedDict, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
TAG_KEYS = ('layers', 'hidden', 'nodes', 'hops', 'cam_samples')


def load_manifest(path):
    with open(path, encoding='utf-8') as f:
        return [json.loads(x) for x in f if x.strip()]


def _fixed(a, vary):
    return {k: v for k, v in a.items() if k not in set(vary) | {'output'}}


def group_name(fixed, vary):
    tag = '_'.join('%s%s' % (k, fixed[k]) for k in TAG_KEYS if k in fixed)
    return '%s_%s_%s_%s_sel-%s_f%02d_s%02d' % (
        fixed['dataset'], fixed['model'], fixed['reg_type'], tag, '+'.join(vary),
        int(fixed['fold']), int(fixed['seed']))


def groups_from_manifest(rows, vary=('jsd_weight',)):
    groups = OrderedDict()
    for row in rows:
        a = row['args']
        missing = [k for k in vary if k not in a]
        if missing:
            raise SystemExit('Manifest rows lack varied keys %s' % missing)
        key = json.dumps(_fixed(a, vary), sort_keys=True)
        groups.setdefault(key, []).append(a)
    names = set()
    for key, members in groups.items():
        combos = [tuple(a[k] for k in vary) for a in members]
        if len(set(combos)) != len(combos):
            raise SystemExit('Group %s has duplicate candidates' % key)
        name = group_name(json.loads(key), vary)
        if name in names:
            raise SystemExit('Two groups map to the same directory name: ' + name)
        names.add(name)
    return groups


def _read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _run_path(a):
    p = Path(a['output'])
    return p if p.is_absolute() else ROOT / p


def candidate_scores(members):
    """Validation-only scores per candidate; refuses incomplete/test-touched runs."""
    cands, problems = [], []
    for a in members:
        run = _run_path(a)
        if not (run / 'DONE.json').exists():
            why = ''
            if (run / 'FAILED.json').exists():
                why = ' FAILED: ' + _read_json(run / 'FAILED.json').get('message', '')[:200]
            problems.append('%s not finished%s' % (run, why))
            continue
        summary = _read_json(run / 'summary.json')
        cfg = _read_json(run / 'config.json')['configuration']
        # Blindness check: grid runs must never have evaluated the test split.
        if cfg.get('evaluate_test') or 'test' in summary.get('outputs', {}) \
                or (run / 'best_test.npz').exists():
            raise SystemExit('Grid run touched the test split, selection would not be blind: %s' % run)
        val = summary.get('outputs', {}).get('val', {})
        if val.get('cross_entropy') is None:
            problems.append('%s has no validation cross-entropy' % run)
            continue
        cands.append(dict(jsd_weight=float(a['jsd_weight']), layers=int(a.get('layers', 1)),
                          run=str(run), val_cross_entropy=float(val['cross_entropy']),
                          val_accuracy=float(val['accuracy']),
                          val_loss_checkpoint_monitor=summary.get('best_val_loss'),
                          best_epoch=summary.get('best_epoch')))
    return cands, problems


def choose(cands):
    """Lowest validation CE; ties -> higher val accuracy -> fewer layers -> smaller lambda."""
    return min(cands, key=lambda c: (round(c['val_cross_entropy'], 12), -c['val_accuracy'],
                                     c.get('layers', 1), c['jsd_weight']))


def evaluate_test(run_dir: Path):
    """Rebuild the model from the copied run and evaluate its held-out test split."""
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from torch import nn
    import pytorch_lightning as pl
    from model import Model
    from dataset import TUDataset, EgonetLoader
    from experiment_tracking import export_predictions_and_responses, posthoc_filter_ablation
    from performance_tracking import benchmark_inference

    cfg = _read_json(run_dir / 'config.json')['configuration']
    split = _read_json(run_dir / 'split.json')
    pl.seed_everything(int(cfg['seed']), workers=True)
    if cfg.get('synthetic_path'):
        from synthetic_data import SyntheticGraphsDataset
        ds = SyntheticGraphsDataset(cfg['synthetic_path'], hops=cfg['hops'])
    else:
        ds = TUDataset(str(ROOT / 'data'), cfg['dataset'], hops=cfg['hops'])
    te = np.asarray(split['test'], dtype=int)
    test_data = [ds[int(i)] for i in te]
    model = Model(Namespace(**cfg, loss=nn.CrossEntropyLoss()))
    ckpts = sorted((run_dir / 'checkpoints').glob('best-*.ckpt'))
    if len(ckpts) != 1:
        raise RuntimeError('Expected exactly one best checkpoint in %s, found %d' % (run_dir, len(ckpts)))
    state = torch.load(ckpts[0], map_location='cpu', weights_only=False)
    model.load_state_dict(state['state_dict'], strict=True)
    model.eval()
    bs = int(cfg['eval_batch_size'])
    test = export_predictions_and_responses(
        model, EgonetLoader(test_data, batch_size=bs, shuffle=False),
        run_dir / 'best_test.npz', 'test', te)
    ablation = posthoc_filter_ablation(model, run_dir / 'best_test.npz',
                                       run_dir / 'best_test_ablation.json', cfg['pooling'])
    bench = benchmark_inference(model, EgonetLoader(test_data, batch_size=bs, shuffle=False),
                                split='test', warmup=int(cfg.get('benchmark_warmup', 2)),
                                repeats=int(cfg.get('benchmark_repeats', 5)),
                                seed=int(cfg.get('cam_eval_seed', 1729)))
    return test, ablation, bench


def run_group(manifest, index, output_root, vary):
    sys.path.insert(0, str(ROOT / 'src'))
    from experiment_tracking import write_json
    groups = list(groups_from_manifest(load_manifest(manifest), vary).items())
    if not 0 <= index < len(groups):
        raise SystemExit('group_index outside 0..%d' % (len(groups) - 1))
    key, members = groups[index]
    fixed = json.loads(key)
    dest = Path(output_root) / fixed['dataset'] / fixed['model'] / group_name(fixed, vary)
    if not dest.is_absolute():
        dest = ROOT / dest
    if (dest / 'SELECTED_DONE.json').exists():
        print('SKIP already selected and tested:', dest)
        return
    cands, problems = candidate_scores(members)
    if problems:
        raise SystemExit('Cannot select, incomplete group %s:\n  %s' % (dest.name, '\n  '.join(problems)))
    best = choose(cands)
    print('GROUP', index, dest.name)
    for c in sorted(cands, key=lambda c: (c['layers'], c['jsd_weight'])):
        print('  L=%d lambda=%-8g val_CE=%.5f val_acc=%.4f%s' % (
            c['layers'], c['jsd_weight'], c['val_cross_entropy'], c['val_accuracy'],
            '   <== selected' if c is best else ''))
    # Work in a temporary copy so a crash never leaves a half-finished result.
    tmp = dest.with_name(dest.name + '.partial')
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(best['run'], tmp)
    for stale in ('DONE.json', 'RUNNING', 'FAILED.json', 'job_stdout.log'):
        (tmp / stale).unlink(missing_ok=True)
    selection = dict(group={k: fixed[k] for k in ('dataset', 'model', 'reg_type', 'fold', 'seed')
                            + TAG_KEYS if k in fixed},
                     varied=list(vary), criterion=(
        'min validation cross-entropy over all validation graphs at each run\'s best '
        'checkpoint; ties -> higher validation accuracy -> fewer layers -> smaller lambda'),
        candidates=cands, selected_jsd_weight=best['jsd_weight'], selected_layers=best['layers'],
        selected_run=best['run'], test_used_for_selection=False, grid_manifest=str(manifest))
    write_json(tmp / 'selection.json', selection)  # written BEFORE any test evaluation
    t0 = time.perf_counter()
    test, ablation, bench = evaluate_test(tmp)
    summary = _read_json(tmp / 'summary.json')
    summary['outputs']['test'] = test
    summary['test_ablation'] = ablation
    summary['inference_benchmark'] = bench
    summary['jsd_weight_selected'] = best['jsd_weight']
    summary['layers_selected'] = best['layers']
    # Label shared by both models so analyses pair GKNN+JSD with CAM+JSD.
    summary['jsd_weight'] = 'val_selected'
    summary['jsd_selection'] = selection
    write_json(tmp / 'inference_benchmark.json', bench)
    write_json(tmp / 'summary.json', summary)
    write_json(tmp / 'SELECTED_DONE.json', dict(success=True, seconds=time.perf_counter() - t0))
    if dest.exists():
        shutil.rmtree(dest)
    tmp.rename(dest)
    print('Selected L=%d lambda=%g, test accuracy=%.4f -> %s' % (
        best['layers'], best['jsd_weight'], test['accuracy'], dest))


# ----------------------------------------------------------------- reporting
def _num(x):
    try:
        x = float(x)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def run_metrics(d: Path):
    """One row: configuration, choice, accuracy, cost, diversity, motif recovery."""
    sel = _read_json(d / 'selection.json')
    summ = _read_json(d / 'summary.json')
    cfg = _read_json(d / 'config.json')['configuration']
    out = summ.get('outputs', {})
    row = OrderedDict(dataset=cfg['dataset'], model=cfg['model'], reg_type=cfg['reg_type'],
                      fold=int(cfg['fold']), seed=int(cfg['seed']))
    for k in TAG_KEYS:
        row[k] = cfg.get(k)
    row.update(selected_layers=sel.get('selected_layers', cfg.get('layers')),
               selected_jsd_weight=sel['selected_jsd_weight'],
               val_cross_entropy=out['val']['cross_entropy'], val_accuracy=out['val']['accuracy'],
               test_accuracy=out.get('test', {}).get('accuracy'),
               test_balanced_accuracy=out.get('test', {}).get('balanced_accuracy'),
               best_epoch=summ.get('best_epoch'), total_train_seconds=summ.get('total_train_seconds'))
    epochs = list(csv.DictReader((d / 'epochs.csv').open(encoding='utf-8')))
    times = [_num(r.get('train_epoch_wall_excluding_validation_s')) for r in epochs[1:]]
    times = [t for t in times if t is not None]
    row['train_epoch_s'] = float(np.mean(times)) if times else None
    row['n_epochs'] = len(epochs)
    at_best = [r for r in epochs if str(r.get('epoch')) == str(summ.get('best_epoch'))]
    row['val_jsd_at_best'] = _num(at_best[0].get('val_jsd')) if at_best else None
    bench = summ.get('inference_benchmark', {})
    row['inference_ms_per_graph'] = _num(bench.get('per_graph_ms_mean'))
    stats = out.get('test', {}).get('response_stats', {})
    for k in range(1, int(cfg.get('layers', 1)) + 1):
        s = stats.get('layer%d_raw' % k, {})
        row['test_layer%d_effective_rank' % k] = s.get('effective_rank')
        row['test_layer%d_mean_abs_corr' % k] = s.get('mean_abs_corr')
    for split in ('test', 'val'):  # motif recovery, if fig8_extended.py was run
        f = d / 'fig8_extended' / split / 'per_motif_summary.csv'
        if f.exists():
            for m in csv.DictReader(f.open(encoding='utf-8')):
                for key in ('max_wl_similarity', 'n_exact_representative_matches',
                            'max_localization_auc', 'mean_localization_auc'):
                    row['motif_%s_%s_%s' % (m['motif'], split, key)] = _num(m.get(key))
    return row


def _write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def aggregate(rows, config_keys):
    """Per configuration: mean over seeds within a fold, then mean/SD across folds."""
    by = defaultdict(list)
    for r in rows:
        by[tuple(r[k] for k in config_keys)].append(r)
    numeric = [k for k in rows[0] if k not in config_keys + ('fold', 'seed', 'reg_type')
               and not k.startswith('selected_')]
    numeric += sorted({k for r in rows for k in r if k.startswith('motif_') or k.startswith('test_layer')}
                      - set(numeric))
    table = []
    for key, rs in sorted(by.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        t = OrderedDict(zip(config_keys, key))
        folds = sorted({r['fold'] for r in rs})
        t.update(n_folds=len(folds), n_runs=len(rs))
        for col in ('selected_layers', 'selected_jsd_weight'):
            counts = defaultdict(int)
            for r in rs:
                counts['%g' % float(r[col])] += 1
            t[col + '_counts'] = json.dumps(dict(sorted(counts.items())))
        for col in numeric:
            per_fold = []
            for f in folds:
                v = [_num(r.get(col)) for r in rs if r['fold'] == f]
                v = [x for x in v if x is not None]
                if v:
                    per_fold.append(float(np.mean(v)))
            t[col + '_mean'] = float(np.mean(per_fold)) if per_fold else None
            t[col + '_sd_folds'] = float(np.std(per_fold, ddof=1)) if len(per_fold) > 1 else None
        table.append(t)
    return table


def summarize(output_root, select_over=None):
    out = Path(output_root)
    dirs = [f.parent for f in sorted(out.rglob('selection.json'))
            if not f.parent.name.endswith('.partial') and (f.parent / 'SELECTED_DONE.json').exists()]
    if not dirs:
        raise SystemExit('No finished selection under ' + str(output_root))
    rows = [run_metrics(d) for d in dirs]
    _write_csv(out / 'selected_per_run.csv', rows)
    cfg_keys = ('dataset', 'model') + TAG_KEYS
    table = aggregate(rows, cfg_keys)
    _write_csv(out / 'selected_summary.csv', table)
    _print(table, 'Per configuration (e.g. fixed depth L)', cfg_keys)
    if select_over:
        # Validation-only choice among the per-configuration winners.
        rest = [k for k in TAG_KEYS if k not in select_over]
        groups = defaultdict(list)
        for r in rows:
            groups[tuple(r[k] for k in ('dataset', 'model', 'fold', 'seed', *rest))].append(r)
        chosen = []
        for rs in groups.values():
            best = min(rs, key=lambda r: (round(float(r['val_cross_entropy']), 12),
                                          -float(r['val_accuracy']), int(r['layers']),
                                          float(r['selected_jsd_weight'])))
            chosen.append(dict(best, selected_layers=best['layers'],
                               n_candidate_configs=len(rs)))
        _write_csv(out / 'overall_selected_per_run.csv', chosen)
        keys = ('dataset', 'model', *rest)
        overall = aggregate(chosen, keys)
        _write_csv(out / 'overall_selected_summary.csv', overall)
        _print(overall, 'Validation-selected over %s' % '+'.join(select_over), keys)
    print('CSV files written in', out)


def _print(table, title, keys):
    print('\n== ' + title)
    for t in table:
        acc, sd = t.get('test_accuracy_mean'), t.get('test_accuracy_sd_folds')
        print('  %s  test acc %s +/- %s | L chosen %s | lambda chosen %s | epoch %.1fs | %.2f ms/graph' % (
            ' '.join('%s=%s' % (k, t[k]) for k in keys),
            '%.2f' % (100 * acc) if acc is not None else 'n/a',
            '%.2f' % (100 * sd) if sd is not None else 'n/a',
            t['selected_layers_counts'], t['selected_jsd_weight_counts'],
            t.get('train_epoch_s_mean') or float('nan'), t.get('inference_ms_per_graph_mean') or float('nan')))


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--grid_manifest', help='Stage-1 manifest')
    p.add_argument('--output_root', default='results/jsd_selected')
    p.add_argument('--vary', nargs='+', default=['jsd_weight'], choices=['jsd_weight', 'layers'],
                   help='Hyperparameters selected on validation inside each group')
    p.add_argument('--select_over', nargs='+', choices=['layers'], default=None,
                   help='With --summarize: also pick these across configurations on validation')
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--list', action='store_true', help='Print the groups and their count')
    mode.add_argument('--group_index', type=int, help='Select + test one group (SLURM array index)')
    mode.add_argument('--summarize', action='store_true', help='Aggregate all selections')
    args = p.parse_args()
    if args.summarize:
        summarize(args.output_root, args.select_over)
        return
    if not args.grid_manifest:
        raise SystemExit('--grid_manifest is required')
    if args.list:
        groups = groups_from_manifest(load_manifest(args.grid_manifest), args.vary)
        for i, (key, members) in enumerate(groups.items()):
            print(i, group_name(json.loads(key), args.vary), 'candidates:',
                  sorted(tuple(a[k] for k in args.vary) for a in members))
        print('Groups:', len(groups), ' -> sbatch --array=0-%d cluster/select_array.slurm %s'
              % (len(groups) - 1, args.grid_manifest))
        return
    run_group(args.grid_manifest, args.group_index, args.output_root, args.vary)


if __name__ == '__main__':
    main()
