"""One-factor-at-a-time ablations and CAM sampling budgets.

These are NEW ablations, not an exact replication of the original paper.
Default: 1 dataset, 1 fold, 1 seed to sanity-check each configuration.
"""
import argparse
import itertools
import json
from pathlib import Path

SWEEPS={
    'mask_count': [('hidden',x) for x in [4,8,16,32,64]],
    'mask_size': [('nodes',x) for x in range(1,11)],
    'depth_radius': [('layers_hops',(l,h)) for l in [1,2,3,5,7] for h in [1,2,3]],
    'radius': [('hops',x) for x in [1,2,3]],
    'graph_kernel': [('kernel',x) for x in ['wl','wloa','gl','prop','py']],
    'jsd_weight': [('jsd_weight',x) for x in [0,100,10000]],
    'cam_mc_samples': [('cam_samples',x) for x in [1,2,6,10,50]],
}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--sweep',choices=SWEEPS,required=True)
    p.add_argument('--dataset',default='PROTEINS')
    p.add_argument('--models',nargs='+',choices=['original','cam'],default=['original','cam'])
    p.add_argument('--seeds',nargs='+',type=int,default=[0])
    p.add_argument('--fold_ids',nargs='+',type=int,default=[0])
    p.add_argument('--folds',type=int,default=10)
    p.add_argument('--max_epochs',type=int,default=400)
    p.add_argument('--patience',type=int,default=75)
    p.add_argument('--manifest',default='cluster/sweep.jsonl')
    p.add_argument('--output_root',default='results/cam_ablations')
    args=p.parse_args()
    if args.sweep=='cam_mc_samples':
        args.models=['cam']
    rows=[]
    for model,seed,fold,(key,value) in itertools.product(args.models,args.seeds,args.fold_ids,
                                                         SWEEPS[args.sweep]):
        param=dict(dataset=args.dataset,model=model,seed=seed,fold=fold,folds=args.folds,
                   reg_type='none',jsd_weight=0,max_epochs=args.max_epochs,
                   patience=args.patience,layers=1,hops=1,nodes=6,hidden=8,
                   cam_samples=4,cam_eval_samples=16)
        if key=='layers_hops':
            param['layers'],param['hops']=value
        elif key=='jsd_weight':
            param['reg_type']='none' if value==0 else 'jsd_code'
            param['jsd_weight']=value
        else:
            param[key]=value
        val='x'.join(str(z) for z in value) if isinstance(value,tuple) else str(value)
        name=f"{args.dataset}_{model}_{args.sweep}_{val}_fold{fold}_seed{seed}"
        param['output']=str(Path(args.output_root)/args.sweep/name)
        rows.append(dict(id=len(rows),name=name,args=param))
    out=Path(args.manifest)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
    print('Saved',len(rows),'jobs to',out)
    print('SLURM: sbatch --array=0-%d%%8 cluster/array.slurm %s' % (len(rows)-1,out))
    if args.sweep=='depth_radius':
        print('WARNING multilayer CAM changes intermediate-label semantics; not a pure mask-law ablation')
    if args.sweep=='cam_mc_samples':
        print('WARNING paper Fig.7 also varied original estimator sample count; this is only a CAM sweep')
    if args.sweep=='graph_kernel':
        print('WARNING verify imports and availability of all GraKeL kernels in src/layers.py before submitting')
    if args.sweep=='jsd_weight':
        print('WARNING JSD_code is not identical to paper JSD; these are code-regime ablations')


if __name__=='__main__':
    main()
