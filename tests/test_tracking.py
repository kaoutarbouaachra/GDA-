"""Fast checks for mathematical diagnostics and posthoc ablation.

From repo root after copying src/experiment_tracking.py:
  python -m pytest tests/test_tracking.py -q
"""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import json
import numpy as np
import torch
from torch import nn
from experiment_tracking import (response_stats, graph_level_metrics,
                                 posthoc_filter_ablation, save_mask_snapshot)


def test_identical_nonconstant_filters_rank_one():
    t=np.linspace(-2,2,9)
    x=np.stack([t,t,t],axis=1)
    stats,c=response_stats(x)
    assert stats['constant_filters']==0
    assert abs(stats['mean_abs_corr']-1.0)<1e-12
    assert abs(stats['effective_rank']-1)<1e-10
    assert abs(stats['participation_rank']-1)<1e-10
    assert np.allclose(c,1)


def test_orthogonal_centered_filters_have_full_effective_rank():
    rng=np.random.default_rng(0)
    x=rng.normal(size=(30,4))
    x-=x.mean(0)
    u,_=np.linalg.qr(x)
    stats,c=response_stats(u)
    assert np.max(np.abs(c-np.eye(4)))<1e-9
    assert abs(stats['effective_rank']-4)<1e-9
    assert abs(stats['participation_rank']-4)<1e-9


def test_constant_filter_is_counted_not_falsely_decorrelated():
    t=np.linspace(0,1,12)
    stats,c=response_stats(np.stack([t,t,0*t+5],axis=1))
    assert stats['active_filters']==2
    assert stats['constant_filters']==1
    assert np.isnan(c[2]).all()


def test_logits_can_be_informative_without_argmax_prediction():
    logits=np.array([[2,0.1],[2,0.2],[2,0.3],[0.9,0.8]])
    labels=np.array([0,0,0,1])
    d,probs=graph_level_metrics(logits,labels,[15,15,15,10])
    assert d['accuracy']==.75
    assert d['balanced_accuracy']==.5
    assert d['roc_auc']==1.0
    assert d['size_only_auc_on_this_split']==1.0
    assert probs.max()<1


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc=nn.Linear(2,2,bias=False)
        self.fc.weight.data=torch.tensor([[0.5,-0.5],[-0.5,0.5]])


def test_posthoc_ablations_match_direct_fc(tmp_path):
    model=FakeModel()
    pooled=np.array([[2.,1.],[1.,3.],[2.,0.]],dtype=np.float32)
    labels=np.array([0,1,0])
    with torch.no_grad():
        logits=model.fc(torch.from_numpy(pooled)).numpy()
    src=tmp_path/'r.npz'
    np.savez(src,labels=labels,logits=logits,layer1_scaled_add=pooled)
    result=posthoc_filter_ablation(model,src,tmp_path/'ablation.json',pooling='add')
    assert len(result['single_ablation_delta_ce'])==2
    assert result['n_pairwise_ablations']==1
    assert result['reconstruction_error']<1e-9
    assert (tmp_path/'ablation.json').exists()


def test_saving_mask_snapshot_without_cam(tmp_path):
    class FakeLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.Padd=nn.Parameter(torch.ones(1,3,3))
            self.Xp=nn.Parameter(torch.ones(1,3,2))
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_layers=nn.ModuleList([FakeLayer()])
    obj=Wrap()
    save_mask_snapshot(obj,tmp_path,0)
    info=json.loads((tmp_path/'masks_0000.json').read_text())
    assert len(info['layers'][0]['masks'])==1
    assert (tmp_path/'state_0000.pt').exists()
