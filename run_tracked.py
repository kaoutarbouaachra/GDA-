"""Reproducible, instrumented GKNN/CAM experiment runner (CPU + GraKeL).

Run from the GDA- repository root. It does NOT modify model.py/layers.py.
One invocation = one model x dataset x fold x seed x hyperparameter setting.
Test is evaluated only after selecting a checkpoint on validation.
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from sklearn.model_selection import StratifiedKFold, train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent/'src'))
from model import Model
from dataset import TUDataset, EgonetLoader
from experiment_tracking import (RichRecorder, export_predictions_and_responses,
    save_mask_snapshot, software_environment, git_info, sha256_file,
    posthoc_filter_ablation, write_json)
from performance_tracking import benchmark_inference


def arguments():
    p=argparse.ArgumentParser(description='Tracked CPU GKNN/CAM training')
    p.add_argument('--dataset',default='MUTAG')
    p.add_argument('--synthetic_path',default=None,
                   help='Optional synthetic JSONL from generate_motifs.py')
    p.add_argument('--model',choices=('original','cam'),default='original')
    p.add_argument('--reg_type',choices=('none','jsd_code','jsd_paper','logdet'),default='none')
    p.add_argument('--jsd_weight',type=float,default=0.0,
                   help='Legacy name: coefficient of chosen regularizer, including logdet')
    p.add_argument('--logdet_eps',type=float,default=1e-4)
    p.add_argument('--fold',type=int,default=0)
    p.add_argument('--folds',type=int,default=10)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--nodes',type=int,default=6)
    p.add_argument('--hidden',type=int,default=8)
    p.add_argument('--layers',type=int,default=1)
    p.add_argument('--hops',type=int,default=1)
    p.add_argument('--kernel',default='wl')
    p.add_argument('--pooling',choices=('add','mean','max'),default='add')
    p.add_argument('--activation',default='relu')
    p.add_argument('--mlp_layers',type=int,choices=(1,2),default=2)
    p.add_argument('--max_cc',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--lr',type=float,default=0.005)
    p.add_argument('--lr_graph',type=float,default=0.01)
    p.add_argument('--batch_size',type=int,default=16)
    p.add_argument('--max_epochs',type=int,default=400)
    p.add_argument('--patience',type=int,default=75)
    p.add_argument('--cam_samples',type=int,default=4)
    p.add_argument('--cam_eval_samples',type=int,default=16)
    p.add_argument('--cam_initial_stop_prob',type=float,default=0.1)
    p.add_argument('--cam_eval_seed',type=int,default=1729)
    p.add_argument('--cam_input_temperature',type=float,default=1.0)
    p.add_argument('--cam_multilayer_inputs',choices=('gknn','categorical'),default='gknn',
                   help="CAM layers>=2: 'gknn' = argmax labels + original GKNN input-gradient "
                        "estimator (same as GKNN; only masks differ). 'categorical' = legacy "
                        "softmax-sampled labels")
    p.add_argument('--eval_batch_size',type=int,default=8)
    p.add_argument('--probe_graphs',type=int,default=4)
    p.add_argument('--snapshot_epochs',default='0,1,4,9,24,49,99,199')
    p.add_argument('--snapshot_every',type=int,default=50)
    p.add_argument('--cam_graph_samples',type=int,default=128)
    p.add_argument('--output',required=True,help='Unique per-run output directory')
    p.add_argument('--reuse_existing_folds',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--evaluate_train',action='store_true',
                   help='Costly: export exact training metrics at best checkpoint')
    p.add_argument('--evaluate_test',action='store_true',
                   help='Only final, post-selection test; NEVER tune on these results')
    p.add_argument('--benchmark_inference',action=argparse.BooleanOptionalAction,default=True,
                   help='Measure latency per graph at best checkpoint, separate from tracking')
    p.add_argument('--benchmark_split',choices=('val','test'),default='test',
                   help='Timing-only inference split; test labels are never used')
    p.add_argument('--benchmark_warmup',type=int,default=2)
    p.add_argument('--benchmark_repeats',type=int,default=5)
    p.add_argument('--disable_probes',action='store_true')
    p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True,
                   help='If --output holds an interrupted run with checkpoints/last.ckpt, '
                        'continue it (model, optimizer, early-stopping and best-checkpoint '
                        'state are restored; global RNG streams are not)')
    return p


def _truncate_epochs_csv(path, last_done_epoch):
    """Drop rows of epochs after the checkpoint we resume from (they will be redone)."""
    import csv
    if not path.exists():
        return
    with path.open(newline='',encoding='utf-8') as f:
        rows=list(csv.DictReader(f))
    keep=[r for r in rows if int(r['epoch'])<=last_done_epoch]
    if len(keep)!=len(rows):
        with path.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(keep)


def split_indices(dataset, labels, folds, fold, reuse=True):
    """Use existing paper-style fold file when valid; else deterministic split.

    Existing convention: 10-fold stratified CV (seed=1), then 10% of the
    training fold reserved for validation (seed=0). The same splits are used
    by both models. This is intentionally not nested CV.
    """
    n=len(labels)
    src=Path('data/folds')/f'{dataset}_folds_{folds}.txt'
    arrays=None
    source='generated_StratifedKFold_random_state1_train_test_split_random_state0'
    if reuse and src.is_file():
        raw=src.read_text(encoding='utf-8')
        try:
            data=json.loads(raw)
        except json.JSONDecodeError:
            data=ast.literal_eval(raw)
        if not isinstance(data,list) or len(data)!=folds:
            raise ValueError('Existing fold file has wrong count: '+str(src))
        if not isinstance(data[fold],(list,tuple)) or len(data[fold])!=3:
            raise ValueError('Existing fold item must contain train,val,test')
        arrays=[np.asarray(a,dtype=int) for a in data[fold]]
        source=str(src)
    else:
        skf=StratifiedKFold(n_splits=folds,shuffle=True,random_state=1)
        splits=list(skf.split(np.arange(n),labels))
        tr,te=splits[fold]
        val_size=int(len(tr)*0.1)
        if val_size<2:
            raise ValueError('Not enough training graphs for 10% validation')
        tr,va=train_test_split(tr,test_size=val_size,stratify=labels[tr],random_state=0)
        arrays=[np.asarray(a,dtype=int) for a in (tr,va,te)]
    for arr in arrays:
        if np.any(arr<0) or np.any(arr>=n) or len(np.unique(arr))!=len(arr):
            raise ValueError('Invalid indices in split')
    tr,va,te=map(set,map(tuple,arrays))
    if tr & va or tr & te or va & te or len(tr|va|te)!=n:
        raise ValueError('Train/val/test folds overlap or do not cover the dataset')
    return arrays,source


def main():
    args=arguments().parse_args()
    if args.fold<0 or args.fold>=args.folds:
        raise SystemExit('--fold must be in [0,folds)')
    if args.reg_type=='none' and args.jsd_weight!=0:
        raise SystemExit('For reg_type none, --jsd_weight must be zero')
    if args.model=='cam' and args.layers>1 and '+' in args.kernel:
        raise SystemExit('CAM multilayer currently supports a single graph kernel')
    out=Path(args.output).resolve()
    if out.exists() and (out/'DONE.json').exists():
        print('Already finished, refusing to overwrite:',out)
        return
    resume_ckpt=None
    if out.exists() and (out/'config.json').exists():
        # A previous attempt was interrupted (e.g. SLURM time limit).
        stored=json.loads((out/'config.json').read_text(encoding='utf-8'))['configuration']
        cli={k:v for k,v in vars(args).items() if k not in ('output','resume')}
        diff={k:(stored.get(k),v) for k,v in cli.items() if k in stored and stored.get(k)!=v}
        last=out/'checkpoints'/'last.ckpt'
        if not args.resume:
            raise SystemExit('Output directory already contains a run (use --resume): '+str(out))
        if diff:
            raise SystemExit('Cannot resume, configuration differs from the stored run: %s' % diff)
        if last.is_file():
            resume_ckpt=last
            print('RESUMING interrupted run from',last)
        elif (out/'epochs.csv').exists():
            raise SystemExit('Interrupted run has epochs but no last.ckpt; move it away: '+str(out))
        else:
            print('Restarting run interrupted before its first epoch:',out)
    out.mkdir(parents=True,exist_ok=True)
    (out/'FAILED.json').unlink(missing_ok=True)  # stale marker from an earlier attempt
    (out/'RUNNING').write_text('running\n',encoding='utf-8')
    t0=time.perf_counter()
    try:
        pl.seed_everything(args.seed,workers=True)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.set_num_threads(max(1,int(os.environ.get('OMP_NUM_THREADS','1'))))
        if args.synthetic_path:
            from synthetic_data import SyntheticGraphsDataset
            ds=SyntheticGraphsDataset(args.synthetic_path,hops=args.hops)
        else:
            ds=TUDataset('./data',args.dataset,hops=args.hops)
        y=np.array([int(ds[i].y.item()) for i in range(len(ds))],dtype=int)
        (tr,va,te),split_source=split_indices(args.dataset,y,args.folds,args.fold,args.reuse_existing_folds)
        if args.synthetic_path:
            # Save exact inserted motif locations for later node-response analysis.
            sys.path.insert(0,str(Path(__file__).resolve().parent/'scripts'))
            from generate_motifs import required_hops
            motif_names=sorted(ds.records[0]['motifs'])
            need=required_hops(motif_names)
            if args.hops<need:
                print(f'WARNING: hops={args.hops} < {need}: no egonet contains the whole motif')
            write_json(out/'synthetic_ground_truth.json',
                       dict(source=args.synthetic_path,
                            motif_visibility=dict(motifs=motif_names,required_hops=need,
                                                  hops=args.hops,whole_motif_visible=args.hops>=need),
                            train={str(i):ds.records[int(i)]['motifs'] for i in tr},
                            val={str(i):ds.records[int(i)]['motifs'] for i in va},
                            test={str(i):ds.records[int(i)]['motifs'] for i in te}))
        write_json(out/'split.json',dict(source=split_source,
                       train=tr.tolist(),val=va.tolist(),test=te.tolist(),
                       train_counts=np.bincount(y[tr]).tolist(),
                       val_counts=np.bincount(y[va]).tolist(),
                       test_counts=np.bincount(y[te]).tolist()))
        train_data=[ds[int(i)] for i in tr]
        val_data=[ds[int(i)] for i in va]
        test_data=[ds[int(i)] for i in te]
        train_loader=EgonetLoader(train_data,batch_size=args.batch_size,shuffle=True)
        val_loader=EgonetLoader(val_data,batch_size=args.eval_batch_size,shuffle=False)
        probe_loader=(EgonetLoader(val_data[:args.probe_graphs],batch_size=args.probe_graphs,shuffle=False)
                      if args.probe_graphs>0 and not args.disable_probes else None)
        args.in_features=int(train_data[0].x.shape[1])
        args.labels=args.in_features
        args.num_classes=int(np.max(y)+1)
        args.loss=nn.CrossEntropyLoss()
        args.optimize_masks=True
        args.mask_distribution=args.model
        args.sparsity=0.0
        args.normalize=True
        metadata=dict(configuration={k:v for k,v in vars(args).items() if k!='loss'},
                      environment=software_environment(),git=git_info(),
                      source_file_hashes={str(p):sha256_file(p) for p in [
                          Path('src/model.py'),Path('src/layers.py'),Path('src/dataset.py'),
                          Path('src/cam.py'),Path('run_tracked.py')]
                          if p.is_file()},
                      split_source=split_source)
        write_json(out/'config.json',metadata)
        model=Model(args)
        if resume_ckpt is None:
            # Initial distribution/mask state, before any training steps.
            save_mask_snapshot(model,out/'snapshots',-1,cam_graph_samples=args.cam_graph_samples)
        else:
            done_epoch=int(torch.load(resume_ckpt,map_location='cpu',weights_only=False)['epoch'])
            _truncate_epochs_csv(out/'epochs.csv',done_epoch)
            log=out/'resume_log.json'
            events=json.loads(log.read_text(encoding='utf-8')) if log.exists() else []
            events.append(dict(resumed_after_epoch=done_epoch,checkpoint=str(resume_ckpt),
                               time=time.strftime('%Y-%m-%d %H:%M:%S'),
                               slurm_job=os.environ.get('SLURM_JOB_ID')))
            write_json(log,events)
        if args.model=='cam' and not hasattr(model.conv_layers[0],'stop_logits'):
            raise RuntimeError('src/model.py not patched for CAM; see README')
        if args.model=='original' and hasattr(model.conv_layers[0],'stop_logits'):
            raise RuntimeError('Original baseline unexpectedly uses CAM')
        checkpoints=ModelCheckpoint(dirpath=str(out/'checkpoints'),monitor='val_loss',mode='min',
                                    save_top_k=1,save_last=True,filename='best-{epoch:03d}-{val_loss:.4f}')
        recorder=RichRecorder(out,probe_loader=probe_loader,
                              snapshot_epochs=tuple(int(s) for s in args.snapshot_epochs.split(',') if s),
                              snapshot_every=args.snapshot_every,
                              cam_graph_samples=args.cam_graph_samples)
        logger=CSVLogger(save_dir=str(out),name='lightning',version='metrics')
        trainer=pl.Trainer(accelerator='cpu',devices=1,max_epochs=args.max_epochs,
                           logger=logger,callbacks=[checkpoints,
                           EarlyStopping(monitor='val_loss',patience=args.patience,mode='min'),recorder],
                           num_sanity_val_steps=0,log_every_n_steps=1,
                           enable_progress_bar=False)
        fit_kwargs={}
        if resume_ckpt is not None:
            fit_kwargs['ckpt_path']=str(resume_ckpt)
            # Our own checkpoint stores hparams objects (e.g. the loss module), which
            # torch>=2.6's default weights_only=True refuses to unpickle.
            import inspect
            if 'weights_only' in inspect.signature(trainer.fit).parameters:
                fit_kwargs['weights_only']=False
        trainer.fit(model,train_dataloaders=train_loader,val_dataloaders=val_loader,**fit_kwargs)
        ckpt=checkpoints.best_model_path
        if not ckpt or not Path(ckpt).exists():
            raise RuntimeError('No best checkpoint saved')
        checkpoint_data=torch.load(ckpt,map_location='cpu',weights_only=False)
        model.load_state_dict(checkpoint_data['state_dict'],strict=True)
        best_epoch=int(checkpoint_data['epoch'])
        save_mask_snapshot(model,out/'best_masks',best_epoch,
                           cam_graph_samples=max(args.cam_graph_samples,256))
        summary=dict(best_epoch=best_epoch,best_val_loss=float(checkpoints.best_model_score),
                     best_checkpoint=str(ckpt),dataset=args.dataset,model=args.model,
                     fold=args.fold,seed=args.seed,reg_type=args.reg_type,
                     jsd_weight=args.jsd_weight,
                     total_train_seconds=float(time.perf_counter()-t0),
                     outputs={})
        if (out/'resume_log.json').exists():
            # total_train_seconds then covers only the last segment; per-epoch
            # times in epochs.csv are complete.
            summary['resume_events']=json.loads((out/'resume_log.json').read_text(encoding='utf-8'))
        if args.evaluate_train:
            loader=EgonetLoader(train_data,batch_size=args.eval_batch_size,shuffle=False)
            summary['outputs']['train']=export_predictions_and_responses(
                model,loader,out/'best_train.npz','train',tr)
        summary['outputs']['val']=export_predictions_and_responses(
            model,EgonetLoader(val_data,batch_size=args.eval_batch_size,shuffle=False),
            out/'best_val.npz','val',va)
        summary['val_ablation']=posthoc_filter_ablation(
            model,out/'best_val.npz',out/'best_val_ablation.json',args.pooling)
        if args.evaluate_test:
            summary['outputs']['test']=export_predictions_and_responses(
                model,EgonetLoader(test_data,batch_size=args.eval_batch_size,shuffle=False),
                out/'best_test.npz','test',te)
            summary['test_ablation']=posthoc_filter_ablation(
                model,out/'best_test.npz',out/'best_test_ablation.json',args.pooling)
        if args.benchmark_inference:
            # Fresh loader: EgonetLoader mutates the yielded batches.
            bench_data=test_data if args.benchmark_split=='test' else val_data
            bench_loader=EgonetLoader(bench_data,batch_size=args.eval_batch_size,shuffle=False)
            summary['inference_benchmark']=benchmark_inference(
                model,bench_loader,split=args.benchmark_split,
                warmup=args.benchmark_warmup,repeats=args.benchmark_repeats,
                seed=args.cam_eval_seed)
            write_json(out/'inference_benchmark.json',summary['inference_benchmark'])
        write_json(out/'summary.json',summary)
        write_json(out/'DONE.json',dict(success=True,seconds=time.perf_counter()-t0))
        (out/'RUNNING').unlink(missing_ok=True)
        print('Run completed, saved:',out)
    except BaseException as exc:
        write_json(out/'FAILED.json',dict(type=type(exc).__name__,message=str(exc),
                                          elapsed_seconds=time.perf_counter()-t0))
        (out/'RUNNING').unlink(missing_ok=True)
        raise


if __name__=='__main__':
    main()
