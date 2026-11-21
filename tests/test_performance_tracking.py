import sys
import json
import csv
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from performance_tracking import benchmark_inference
from aggregate_runtime import aggregate,collect,pair_model_cost,aggregate_ratios

class FakeBatch:
    def __init__(self,n_graphs):
        self.num_graphs=n_graphs
        self.features=torch.ones(n_graphs,3)

class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(3,2)
    def forward(self,data):
        return self.fc(data.features)


def test_inference_bench_counts_graphs_not_batches():
    mdl=FakeModel()
    assert mdl.training
    result=benchmark_inference(mdl,[FakeBatch(3),FakeBatch(2)],warmup=1,repeats=3)
    assert result['n_graphs']==5
    assert result['n_batches']==2
    assert result['repeats']==3
    assert result['per_graph_ms_mean']>0
    assert mdl.training


def test_aggregate_folds_not_independent_seeds():
    base=dict(dataset='MUTAG',model='cam',reg_type='none',**{'lambda':0.0},
              layers=1,hidden=8,nodes=6,hops=1,kernel='wl',pooling='add',batch_size=16,
              eval_batch_size=8,cam_samples=4,cam_eval_samples=16,cpu_threads='4',
              slurm_cpus='4',inference_split='test')
    rows=[]
    for fold,seed,t,inf in [(0,0,10.,2.),(0,1,14.,4.),(1,0,20.,6.)]:
        rows.append(dict(base,fold=fold,seed=seed,training_epoch_s_mean=t,
                         inference_per_graph_ms=inf))
    per_fold,summary=aggregate(rows)
    assert len(per_fold)==2
    assert summary[0]['n_folds']==2
    assert summary[0]['n_seed_runs']==3
    assert np.isclose(summary[0]['mean_training_epoch_s'],16.0)
    assert np.isclose(summary[0]['mean_inference_per_graph_ms'],4.5)


def test_collect_from_saved_run(tmp_path):
    folder=tmp_path/'MUTAG'/'cam'/'f0'/'s0'
    folder.mkdir(parents=True)
    (folder/'summary.json').write_text(json.dumps(dict(dataset='MUTAG',model='cam',fold=0,
             seed=0,reg_type='none',jsd_weight=0,
             inference_benchmark=dict(per_graph_ms_mean=5.2,n_graphs=19,split='test'))))
    (folder/'config.json').write_text(json.dumps(dict(configuration=dict(layers=1,hidden=8),
             environment=dict(threads=dict(OMP_NUM_THREADS='4'),slurm=dict(SLURM_CPUS_PER_TASK='4')))))
    with (folder/'epochs.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['epoch','train_epoch_batch_s'])
        w.writeheader()
        for ep,t in enumerate([22.,10.,12.]):
            w.writerow(dict(epoch=ep,train_epoch_batch_s=t))
    r=collect(tmp_path,skip_first_epoch=1)
    assert len(r)==1
    assert r[0]['training_epoch_s_mean']==11.
    assert r[0]['inference_per_graph_ms']==5.2


def test_paired_speedup_ratio():
    shared=dict(dataset='MUTAG',fold=0,seed=3,reg_type='none',**{'lambda':0.0},
                layers=1,hidden=8,nodes=6,hops=1,kernel='wl',pooling='add',
                batch_size=16,eval_batch_size=8,cpu_threads='4',slurm_cpus='4',
                inference_split='test')
    base=dict(shared,model='original',training_epoch_s_mean=10.0,inference_per_graph_ms=20.0,run='baseline')
    cam=dict(shared,model='cam',training_epoch_s_mean=30.0,inference_per_graph_ms=50.0,
             cam_samples=4,cam_eval_samples=16,run='cam')
    rows=pair_model_cost([base,cam])
    assert len(rows)==1
    assert rows[0]['ratio_training_time_cam_over_gknn']==3
    assert rows[0]['ratio_inference_time_cam_over_gknn']==2.5
    summary=aggregate_ratios(rows)
    assert summary[0]['n_folds']==1
