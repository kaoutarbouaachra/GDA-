"""Generate independent CPU/SLURM tasks for GKNN vs CAM.

Each line is one JSON object: deterministic, reproducible, no shell injection.
Default grid is *model generator comparison* with no JSD/logdet.
"""
from __future__ import annotations
import argparse
import itertools
import json
import sys
from pathlib import Path


def parser():
    p=argparse.ArgumentParser()
    p.add_argument('--datasets',nargs='+',default=['MUTAG','PTC','PROTEINS'])
    p.add_argument('--synthetic_mapping',default=None,
                   help='JSON mapping dataset labels to synthetic JSONL paths')
    p.add_argument('--models',nargs='+',default=['original','cam'],
                   choices=['original','cam'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--folds',type=int,default=10,
                   help='Total K folds of each dataset, must match existing split file')
    p.add_argument('--fold_ids',nargs='*',type=int,default=None,
                   help='Subset of folds; default means all folds')
    p.add_argument('--reg_types',nargs='+',choices=['none','jsd_code','jsd_paper','logdet'],
                   default=['none'])
    p.add_argument('--lambda_values',nargs='+',type=float,default=None,
                   help='Use only with ONE non-none reg_type; tune on validation')
    p.add_argument('--layers',type=int,nargs='+',default=[1],
                   help='One or more depths; several = depth grid selected on validation')
    p.add_argument('--cam_multilayer_inputs',choices=('gknn','categorical'),default='gknn')
    p.add_argument('--allow_partial_motif_view',action='store_true',
                   help='Allow synthetic runs whose egonet radius cannot contain the motif')
    p.add_argument('--nodes',type=int,default=6)
    p.add_argument('--hidden',type=int,default=8)
    p.add_argument('--hops',type=int,default=1)
    p.add_argument('--kernel',default='wl')
    p.add_argument('--pooling',default='add')
    p.add_argument('--cam_samples',type=int,default=4)
    p.add_argument('--cam_eval_samples',type=int,default=16)
    p.add_argument('--max_epochs',type=int,default=400)
    p.add_argument('--patience',type=int,default=75)
    p.add_argument('--batch_size',type=int,default=16)
    p.add_argument('--eval_batch_size',type=int,default=8)
    p.add_argument('--snapshot_every',type=int,default=50)
    p.add_argument('--evaluate_test',action='store_true')
    p.add_argument('--benchmark_inference',action=argparse.BooleanOptionalAction,default=True,
                   help='--no-benchmark_inference for hyperparameter-grid runs; the '
                        'benchmark is then run once on the selected run (select_jsd.py)')
    p.add_argument('--output_root',default='results/cam_gknn')
    p.add_argument('--manifest',default='cluster/jobs.jsonl')
    return p


def check_motif_visibility(ds,path,hops,nodes,allow_partial):
    """Refuse synthetic jobs whose egonets cannot contain the inserted motif."""
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'scripts'))
    from generate_motifs import required_hops, motif_graph
    with open(path,encoding='utf-8') as f:
        names=sorted(json.loads(f.readline())['motifs'])
    need=required_hops(names)
    size=max(motif_graph(n).number_of_nodes() for n in names)
    print(f'  {ds}: motifs {names} need --hops >= {need} and --nodes >= {size} '
          f'(given hops={hops}, nodes={nodes})')
    if (hops<need or nodes<size) and not allow_partial:
        raise SystemExit(f'{ds}: egonet radius {hops} / mask size {nodes} cannot contain the '
                         f'whole motif (need hops>={need}, nodes>={size}). '
                         'Use --allow_partial_motif_view only for a deliberate fragment study.')


def main():
    args=parser().parse_args()
    if args.lambda_values and (len(args.reg_types)!=1 or args.reg_types[0]=='none'):
        raise SystemExit('Provide --lambda_values only with one non-none --reg_types')
    if ((args.lambda_values and len(args.lambda_values)>1) or len(args.layers)>1) \
            and args.evaluate_test:
        raise SystemExit('A hyperparameter grid must not touch the test split. Drop --evaluate_test; '
                         'cluster/select_jsd.py selects on validation, then tests once.')
    fold_ids=args.fold_ids if args.fold_ids is not None else list(range(args.folds))
    if any(f<0 or f>=args.folds for f in fold_ids):
        raise SystemExit('fold_ids must be within 0,...,folds-1')
    out=Path(args.manifest)
    out.parent.mkdir(parents=True,exist_ok=True)
    jobs=[]
    synthetic_mapping=(json.loads(Path(args.synthetic_mapping).read_text(encoding='utf-8'))
                       if args.synthetic_mapping else {})
    for ds in args.datasets:
        if ds in synthetic_mapping:
            check_motif_visibility(ds,synthetic_mapping[ds],args.hops,args.nodes,
                                   args.allow_partial_motif_view)
    tag_depth=len(args.layers)>1
    # Order: dataset, depth, model, ... so each (dataset, depth) block is contiguous
    # and can be submitted with its own --time.
    for ds,L,model,reg,fold,seed in itertools.product(
        args.datasets,args.layers,args.models,args.reg_types,fold_ids,args.seeds):
        grid=[0.0] if reg=='none' else (args.lambda_values or [1.0])
        for lam in grid:
            # Do not reuse the same output directory across hyperparameter choices.
            label=('%.8g'%lam).replace('.','p').replace('-','n')
            depth=f'_L{L}' if tag_depth else ''
            jobname=f'{ds}_{model}_{reg}{depth}_w{label}_f{fold:02d}_s{seed:02d}'
            dest=str(Path(args.output_root)/ds/model/reg/jobname)
            params=dict(dataset=ds,model=model,reg_type=reg,jsd_weight=lam,
                        fold=fold,folds=args.folds,seed=seed,layers=L,
                        cam_multilayer_inputs=args.cam_multilayer_inputs,
                        nodes=args.nodes,hidden=args.hidden,hops=args.hops,
                        kernel=args.kernel,pooling=args.pooling,
                        cam_samples=args.cam_samples,cam_eval_samples=args.cam_eval_samples,
                        max_epochs=args.max_epochs,patience=args.patience,
                        batch_size=args.batch_size,eval_batch_size=args.eval_batch_size,
                        snapshot_every=args.snapshot_every,
                        evaluate_test=args.evaluate_test,output=dest)
            if ds in synthetic_mapping:
                params['synthetic_path']=synthetic_mapping[ds]
            if not args.benchmark_inference:
                params['no-benchmark_inference']=True  # run_manifest emits --no-benchmark_inference
            jobs.append({'id':len(jobs),'name':jobname,'args':params})
    out.write_text(''.join(json.dumps(j,sort_keys=True)+'\n' for j in jobs),encoding='utf-8')
    print('Manifest:',out,' jobs:',len(jobs))
    if jobs:
        print('SLURM: sbatch --array=0-%d%%8 cluster/array.slurm %s' % (len(jobs)-1,out))
        # (dataset, depth) are contiguous blocks: submit each with its own --time.
        for ds,L in itertools.product(args.datasets,args.layers):
            ids=[j['id'] for j in jobs if j['args']['dataset']==ds and j['args']['layers']==L]
            if ids:
                print('  %-12s L=%d  array indices %d-%d (%d jobs)' % (ds,L,ids[0],ids[-1],len(ids)))
        print('CPU budget note: CAM uses GraKeL on CPU, not GPU acceleration.')


if __name__=='__main__':
    main()
