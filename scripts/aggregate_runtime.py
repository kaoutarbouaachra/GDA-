"""Aggregate per-epoch training and best-checkpoint per-graph inference time.

Run from repository root, e.g.:
  python scripts/aggregate_runtime.py --root results/cam_full --out results/timing

The main summary is mean +/- SAMPLE SD ACROSS FOLDS; when several seeds per
fold exist, seed results are averaged within each fold first. NEVER mix
hyperparameter settings, datasets, or thread allocations silently.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _mean_std(xs):
    a=np.asarray([x for x in xs if x is not None and np.isfinite(x)],dtype=float)
    if a.size==0:
        return None,None
    return float(a.mean()),float(a.std(ddof=1)) if len(a)>1 else None


def _number(value):
    try:
        x=float(value)
        return x if np.isfinite(x) else None
    except (ValueError,TypeError):
        return None


def collect(root, skip_first_epoch=1):
    result=[]
    for summary_file in Path(root).rglob('summary.json'):
        d=summary_file.parent
        try:
            summary=json.loads(summary_file.read_text(encoding='utf-8'))
            cfg=json.loads((d/'config.json').read_text(encoding='utf-8'))
            opts=cfg.get('configuration',{})
            timing=summary.get('inference_benchmark',{})
            epochs_file=d/'epochs.csv'
            with epochs_file.open(newline='',encoding='utf-8') as f:
                epochs=list(csv.DictReader(f))
            # Exclude initialization/warmup epoch, not the final selected epoch.
            epoch_times=[_number(row.get('train_epoch_wall_excluding_validation_s'))
                         or _number(row.get('train_epoch_batch_s'))
                         for row in epochs[skip_first_epoch:]]
            epoch_times=[e for e in epoch_times if e is not None]
            if not epoch_times:
                # Old runs require re-instrumentation, not total wall/epochs as surrogate.
                continue
            training_s, training_sd=_mean_std(epoch_times)
            timing_definition=('train_epoch_wall_excluding_validation_s'
                if all(_number(r.get('train_epoch_wall_excluding_validation_s')) is not None
                       for r in epochs[skip_first_epoch:])
                else 'train_epoch_batch_s_or_mixed')
            row={
                'dataset':summary['dataset'], 'model':summary['model'],
                'reg_type':summary.get('reg_type','none'),
                'lambda':summary.get('jsd_weight',0),
                'fold':int(summary['fold']), 'seed':int(summary['seed']),
                'layers':opts.get('layers'), 'hidden':opts.get('hidden'),
                'nodes':opts.get('nodes'), 'hops':opts.get('hops'),
                'kernel':opts.get('kernel'), 'pooling':opts.get('pooling'),
                'batch_size':opts.get('batch_size'),
                'eval_batch_size':opts.get('eval_batch_size'),
                'cam_samples':opts.get('cam_samples'),
                'cam_eval_samples':opts.get('cam_eval_samples'),
                'cpu_threads':cfg.get('environment',{}).get('threads',{}).get('OMP_NUM_THREADS'),
                'slurm_cpus':cfg.get('environment',{}).get('slurm',{}).get('SLURM_CPUS_PER_TASK'),
                'training_epoch_s_mean':training_s,
                'training_timing_definition':timing_definition,
                'training_epoch_s_sd_across_epochs':training_sd,
                'n_timed_epochs':len(epoch_times),
                'inference_per_graph_ms':_number(timing.get('per_graph_ms_mean')),
                'inference_sd_across_repeats_ms':_number(timing.get('per_graph_ms_sd')),
                'n_inference_graphs':timing.get('n_graphs'),
                'inference_split':timing.get('split'),
                'inference_batch_count':timing.get('n_batches'),
                # Accuracy is descriptive only; test was not used for selection.
                'test_accuracy_percent':(
                    100.0*float(summary['outputs']['test']['accuracy'])
                    if summary.get('outputs',{}).get('test',{}).get('accuracy') is not None
                    else None),
                'val_accuracy_percent':(
                    100.0*float(summary['outputs']['val']['accuracy'])
                    if summary.get('outputs',{}).get('val',{}).get('accuracy') is not None
                    else None),
                'run':str(d),
            }
            result.append(row)
        except (KeyError, OSError, json.JSONDecodeError, ValueError):
            continue
    return result


GROUP_FIELDS=('dataset','model','reg_type','lambda','layers','hidden','nodes',
              'hops','kernel','pooling','batch_size','eval_batch_size',
              'cam_samples','cam_eval_samples','cpu_threads','slurm_cpus',
              'inference_split','training_timing_definition')


def aggregate(rows):
    groups=defaultdict(list)
    for row in rows:
        groups[tuple(row.get(k) for k in GROUP_FIELDS)].append(row)
    by_fold=[]
    summary=[]
    for key,runs in sorted(groups.items(),key=lambda z:str(z[0])):
        definition=dict(zip(GROUP_FIELDS,key))
        folds=defaultdict(list)
        for run in runs:
            folds[run['fold']].append(run)
        fm=[]
        for fold,fruns in sorted(folds.items()):
            train,_=_mean_std([r['training_epoch_s_mean'] for r in fruns])
            infer,_=_mean_std([r['inference_per_graph_ms'] for r in fruns])
            fr=dict(definition,fold=int(fold),n_seed_runs=len(fruns),
                    training_epoch_s=train,inference_per_graph_ms=infer)
            fm.append(fr)
            by_fold.append(fr)
        mt,st=_mean_std([f['training_epoch_s'] for f in fm])
        mi,si=_mean_std([f['inference_per_graph_ms'] for f in fm])
        summary.append(dict(definition,n_folds=len(fm),n_seed_runs=len(runs),
                            mean_training_epoch_s=mt,sd_training_epoch_s=st,
                            mean_inference_per_graph_ms=mi,sd_inference_per_graph_ms=si))
    return by_fold,summary


def pair_model_cost(rows):
    """Pair CAM and original runs, same dataset/fold/seed/hardware/hparams.

    Original can be reused as the comparator for different CAM sample budgets.
    The derived ratios remain identified by CAM's sample budget.
    """
    pairing_fields=('dataset','fold','seed','reg_type','lambda','layers',
                    'hidden','nodes','hops','kernel','pooling','batch_size',
                    'eval_batch_size','cpu_threads','slurm_cpus','inference_split')
    baselines=defaultdict(list)
    for r in rows:
        if r.get('model')=='original':
            baselines[tuple(r.get(k) for k in pairing_fields)].append(r)
    paired=[]
    for r in rows:
        if r.get('model')!='cam':
            continue
        lookup=tuple(r.get(k) for k in pairing_fields)
        for base in baselines.get(lookup,[]):
            # same original model but distinct runs only if input files contain duplicates
            t1,t0=r.get('training_epoch_s_mean'),base.get('training_epoch_s_mean')
            i1,i0=r.get('inference_per_graph_ms'),base.get('inference_per_graph_ms')
            paired.append(dict(**{k:r.get(k) for k in pairing_fields},
                cam_samples=r.get('cam_samples'),cam_eval_samples=r.get('cam_eval_samples'),
                ratio_training_time_cam_over_gknn=(t1/t0 if t1 is not None and t0 and t0>0 else None),
                ratio_inference_time_cam_over_gknn=(i1/i0 if i1 is not None and i0 and i0>0 else None),
                cam_dir=r.get('run'),gknn_dir=base.get('run')))
    return paired


def aggregate_ratios(paired):
    dimensions=('dataset','reg_type','lambda','layers','hidden','nodes','hops',
                'kernel','pooling','batch_size','eval_batch_size','cpu_threads',
                'slurm_cpus','inference_split','cam_samples','cam_eval_samples')
    groups=defaultdict(list)
    for p in paired:
        groups[tuple(p.get(k) for k in dimensions)].append(p)
    summary=[]
    for key,runs in sorted(groups.items(),key=lambda x:str(x[0])):
        folds=defaultdict(list)
        for r in runs:
            folds[r['fold']].append(r)
        fold_values=[]
        for fold,fruns in folds.items():
            a,_=_mean_std([r['ratio_training_time_cam_over_gknn'] for r in fruns])
            b,_=_mean_std([r['ratio_inference_time_cam_over_gknn'] for r in fruns])
            fold_values.append((a,b))
        mt,st=_mean_std([a for a,b in fold_values])
        mi,si=_mean_std([b for a,b in fold_values])
        summary.append(dict(**dict(zip(dimensions,key)),n_folds=len(folds),
            n_pairs=len(runs),mean_training_ratio_cam_gknn=mt,
            sd_training_ratio_cam_gknn=st,mean_inference_ratio_cam_gknn=mi,
            sd_inference_ratio_cam_gknn=si))
    return summary


def _save_csv(rows,path):
    if not rows:
        return
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def draw(summary,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Only draw pairs with same dataset / compute configuration / evaluator.
    cohorts=defaultdict(list)
    for r in summary:
        config=(r['dataset'],r['layers'],r['hidden'],r['nodes'],r['hops'],r['kernel'],
                r['batch_size'],r['eval_batch_size'],r['cpu_threads'],r['slurm_cpus'])
        cohorts[config].append(r)
    for n,(cohort,rs) in enumerate(cohorts.items()):
        fig,axes=plt.subplots(1,2,figsize=(11,4.5))
        names=[r['model']+' / '+r['reg_type'] for r in rs]
        for ax,col,err,title in zip(axes,['mean_training_epoch_s','mean_inference_per_graph_ms'],
                                    ['sd_training_epoch_s','sd_inference_per_graph_ms'],
                                    ['Training time / epoch (s)', 'Inference / graph (ms)']):
            values=[r.get(col) if r.get(col) is not None else np.nan for r in rs]
            errors=[r.get(err) if r.get(err) is not None else 0.0 for r in rs]
            ax.bar(np.arange(len(rs)),values,yerr=errors,capsize=3)
            ax.set_xticks(np.arange(len(rs)),names,rotation=30,ha='right')
            ax.set_ylabel(title)
            ax.grid(axis='y',alpha=0.25)
        fig.suptitle(f'{cohort[0]} | CPU time | folds: mean +/- SD')
        fig.tight_layout()
        for ext in ('.pdf','.png'):
            fig.savefig(out/f'{cohort[0]}_time_cohort_{n}{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)


def draw_accuracy_runtime(rows, out):
    """Accuracy vs inference latency; per-run raw points, not implied causality."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    by_dataset=defaultdict(list)
    for r in rows:
        if r.get('inference_per_graph_ms') is not None:
            by_dataset[r['dataset']].append(r)
    for dataset, runs in by_dataset.items():
        # Never merge validation and test accuracies in one plot.
        key=('test_accuracy_percent' if all(r.get('test_accuracy_percent') is not None
                                            for r in runs) else 'val_accuracy_percent')
        fig,ax=plt.subplots(figsize=(7,4.7))
        methods=sorted(set((r['model'],r['reg_type']) for r in runs))
        for name in methods:
            rs=[r for r in runs if (r['model'],r['reg_type'])==name]
            coords=[(r['inference_per_graph_ms'],r.get(key)) for r in rs if r.get(key) is not None]
            if coords:
                ax.scatter([p[0] for p in coords],[p[1] for p in coords],
                           label='/'.join(name),alpha=.75)
        ax.set(xlabel='Best-checkpoint inference time per graph (ms)',
               ylabel=('Test accuracy (%)' if key.startswith('test') else 'Validation accuracy (%)'),
               title=f'{dataset}: predictive accuracy vs inference cost')
        ax.grid(alpha=.25)
        if ax.collections:
            ax.legend()
        fig.tight_layout()
        for ext in ('.png','.pdf'):
            fig.savefig(out/(dataset+'_accuracy_vs_latency'+ext),dpi=180)
        plt.close(fig)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--skip_first_epoch',type=int,default=1)
    args=p.parse_args()
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    rows=collect(args.root,args.skip_first_epoch)
    if not rows:
        raise SystemExit('No instrumented runs: need epochs.csv with train_epoch_batch_s')
    folds,summary=aggregate(rows)
    _save_csv(rows,out/'timing_per_run.csv')
    _save_csv(folds,out/'timing_per_fold.csv')
    _save_csv(summary,out/'timing_summary.csv')
    paired=pair_model_cost(rows)
    _save_csv(paired,out/'paired_cam_gknn_cost_ratios.csv')
    _save_csv(aggregate_ratios(paired),out/'cam_gknn_cost_ratio_summary.csv')
    draw(summary,out)
    draw_accuracy_runtime(rows,out)
    print(f'{len(rows)} runs, {len(summary)} configurations -> {out}')


if __name__=='__main__':
    main()
