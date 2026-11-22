"""One-factor-at-a-time ablations and CAM sampling budgets.

These are NEW ablations, not an exact replication of the original paper.
Each value of the swept factor is one configuration; with --reg_type jsd_code
and --lambda_values, every configuration is trained for every lambda and
cluster/select_jsd.py then picks lambda on VALIDATION per configuration, fold
and seed (same grid and criterion for GKNN and CAM). Grid runs never touch the
test split.
"""
import argparse
import itertools
import json
from pathlib import Path

SWEEPS={
    'mask_count': [('hidden',x) for x in [4,8,16,32,64]],
    'mask_size': [('nodes',x) for x in range(1,11)],
    'depth': [('layers',x) for x in [1,2,3,5,7]],
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
    p.add_argument('--batch_size',type=int,default=16)
    # Fixed (non-swept) architecture; defaults = previous behaviour.
    p.add_argument('--layers',type=int,default=1,help='Depth when depth is not swept')
    p.add_argument('--hops',type=int,default=1)
    p.add_argument('--nodes',type=int,default=6)
    p.add_argument('--hidden',type=int,default=8)
    p.add_argument('--cam_samples',type=int,default=4)
    p.add_argument('--cam_eval_samples',type=int,default=16)
    p.add_argument('--cam_multilayer_inputs',choices=('gknn','categorical'),default='gknn')
    p.add_argument('--reg_type',choices=['none','jsd_code','jsd_paper','logdet'],default='none')
    p.add_argument('--lambda_values',nargs='+',type=float,default=None,
                   help='Regularization grid, selected on validation by select_jsd.py')
    p.add_argument('--benchmark_inference',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--manifest',default='cluster/sweep.jsonl')
    p.add_argument('--output_root',default='results/cam_ablations')
    args=p.parse_args()
    if args.sweep=='cam_mc_samples':
        args.models=['cam']
    if args.sweep=='jsd_weight' and (args.reg_type!='none' or args.lambda_values):
        raise SystemExit('The jsd_weight sweep sets the regularizer itself')
    if args.lambda_values and args.reg_type=='none':
        raise SystemExit('--lambda_values needs a non-none --reg_type')
    lambdas=args.lambda_values or ([1.0] if args.reg_type!='none' else [0.0])
    rows=[]
    for model,(key,value),seed,fold,lam in itertools.product(
            args.models,SWEEPS[args.sweep],args.seeds,args.fold_ids,lambdas):
        param=dict(dataset=args.dataset,model=model,seed=seed,fold=fold,folds=args.folds,
                   reg_type=args.reg_type,jsd_weight=lam,max_epochs=args.max_epochs,
                   patience=args.patience,batch_size=args.batch_size,layers=args.layers,
                   hops=args.hops,nodes=args.nodes,hidden=args.hidden,
                   cam_samples=args.cam_samples,cam_eval_samples=args.cam_eval_samples,
                   cam_multilayer_inputs=args.cam_multilayer_inputs)
        if key=='layers_hops':
            param['layers'],param['hops']=value
        elif key=='jsd_weight':
            param['reg_type']='none' if value==0 else 'jsd_code'
            param['jsd_weight']=value
        else:
            param[key]=value
        if not args.benchmark_inference:
            param['no-benchmark_inference']=True
        val='x'.join(str(z) for z in value) if isinstance(value,tuple) else str(value)
        reg='' if args.reg_type=='none' else '_%s_w%s' % (args.reg_type,('%.8g'%lam).replace('.','p'))
        name=f"{args.dataset}_{model}_{args.sweep}_{val}{reg}_fold{fold}_seed{seed}"
        param['output']=str(Path(args.output_root)/args.sweep/name)
        rows.append(dict(id=len(rows),name=name,args=param))
    out=Path(args.manifest)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(''.join(json.dumps(x,sort_keys=True)+'\n' for x in rows))
    print('Saved',len(rows),'jobs to',out)
    print('SLURM: sbatch --array=0-%d%%8 cluster/array.slurm %s' % (len(rows)-1,out))
    if args.lambda_values:
        n=len(rows)//len(lambdas)
        print('Then select lambda on validation: %d groups -> sbatch --array=0-%d '
              'cluster/select_array.slurm %s <output_root_selected>' % (n,n-1,out))
    if args.sweep in ('depth','depth_radius'):
        print('NOTE CAM layers>=2 use --cam_multilayer_inputs=%s' % args.cam_multilayer_inputs)
    if args.sweep=='cam_mc_samples':
        print('WARNING paper Fig.7 also varied original estimator sample count; this is only a CAM sweep')
    if args.sweep=='graph_kernel':
        print('WARNING gl/py kernels are much slower than wl; check timing before submitting')
    if args.sweep=='jsd_weight':
        print('WARNING JSD_code is not identical to paper JSD; these are code-regime ablations')


if __name__=='__main__':
    main()
