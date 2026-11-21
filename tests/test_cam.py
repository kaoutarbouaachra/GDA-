"""Unit tests for CAM without GraKeL/PyG: run `python -m pytest tests/test_cam.py -q`."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))

import numpy as np
import pytest
import torch
from cam import CAMKernel


class FakeKernel:
    """Test-only deterministic graph kernel surrogate (not used in real training)."""
    def fit(self, graphs):
        self.inputs=graphs
        return self
    def transform(self, masks):
        def get_graph_features(g):
            adj,labels=g
            edges=set((min(int(u),int(v)),max(int(u),int(v)))
                      for u,nbs in adj.items() for v in nbs if int(u)!=int(v))
            return edges,labels
        out=[]
        for mask in masks:
            m_edges,m_labels=get_graph_features(mask)
            row=[]
            for ego in self.inputs:
                g_edges,g_labels=get_graph_features(ego)
                # Depends on topology AND labels of BOTH input and mask.
                val = 0.35+0.8*len(m_edges)+1.3*int((0,2) in m_edges)
                val += 0.2*sum(m_labels.values())+0.05*len(g_edges)
                val += 0.14*sum(g_labels.values())
                val += 0.5*int(m_labels[0]==g_labels[0])
                row.append(val)
            out.append(row)
        return np.asarray(out,dtype=float)


def kernel(p=4, filters=2, labels=3, B=8, mode='argmax', kernels='wl'):
    return CAMKernel(p,labels,filters,mc_samples=B,eval_samples=B,
                     input_label_mode=mode,kernels=kernels,
                     kernel_factories={'wl':FakeKernel,'gl':FakeKernel})


def small_data():
    x = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.],[1.,0.,0.]])
    edges=torch.tensor([[0,1,1,2,2,3],[1,0,2,1,3,2]])
    # 2-hop/1-hop synthetic egonets, local edge ids
    idx=[np.array([0,1]),np.array([0,1,2]),np.array([1,2,3]),np.array([2,3])]
    eg=[np.array([[0,1],[1,0]]),
        np.array([[0,1,1,2],[1,0,2,1]]),
        np.array([[0,1,1,2],[1,0,2,1]]),
        np.array([[0,1],[1,0]])]
    return x,edges,torch.zeros(4,dtype=torch.long),(idx,eg)


def connected(tr):
    adj={v:set() for v in range(tr.size)}
    for a,b in tr.edges:
        adj[a].add(b);adj[b].add(a)
    seen={0};queue=[0]
    while queue:
        for v in adj[queue.pop()]:
            if v not in seen:
                seen.add(v);queue.append(v)
    return len(seen)==tr.size


def test_size_law_and_stop_grad():
    c=kernel(p=5)
    p=c.size_probabilities()
    assert p.shape==(2,5)
    assert torch.allclose(p.sum(1),torch.ones(2))
    (p[:,0]+2*p[:,4]).sum().backward()
    assert torch.isfinite(c.stop_logits.grad).all()
    assert c.stop_logits.grad.abs().sum()>0


def test_sampling_connected_and_trace_probabilities():
    torch.manual_seed(4)
    c=kernel(p=6)
    for _ in range(60):
        tr=c.sample_trace(0)
        assert 1<=tr.size<=6
        assert len(tr.edges)>=tr.size-1
        assert connected(tr)
        assert len(tr.optional_edges)==sum(t-1 for t in range(1,tr.size))
        assert torch.allclose(tr.log_prob,c.trace_log_probability(tr),atol=1e-6)


def test_edge_dependence_and_marginals():
    torch.manual_seed(6)
    c=kernel(p=3,filters=1)
    with torch.no_grad():
        c.stop_logits.fill_(-20)
        c.optional_logits.fill_(-20)
        c.parent_logits[0,2,0]=np.log(.7)
        c.parent_logits[0,2,1]=np.log(.3)
    marg=c.edge_marginals(0,2).detach()
    assert torch.allclose(marg,torch.tensor([.7,.3]),atol=2e-5)
    samples=[c.sample_trace(0) for _ in range(500)]
    assert all(tr.size==3 for tr in samples)
    a=np.mean([(0,2) in tr.edges for tr in samples])
    b=np.mean([(1,2) in tr.edges for tr in samples])
    both=np.mean([((0,2) in tr.edges and (1,2) in tr.edges) for tr in samples])
    assert abs(a-.7)<.07 and abs(b-.3)<.07 and both==0


def test_mask_gradients_all_four_parameter_families():
    torch.manual_seed(11)
    c=kernel(p=4,filters=2,B=24)
    # Ensure meaningful chance of size >=3, optional edges and alternate labels.
    with torch.no_grad():
        c.stop_logits.fill_(-1)
    x,e,b,egos=small_data()
    c.train()
    z=c(x,e,b,egonets_data=egos)
    assert z.shape==(4,2) and torch.isfinite(z).all()
    loss=(z**2).mean()
    loss.backward()
    for group in ('stop_logits','parent_logits','optional_logits','label_logits'):
        grad=getattr(c,group).grad
        assert grad is not None and torch.isfinite(grad).all(),group
        assert grad.abs().sum()>0,group


def test_one_layer_mc_gradient_approximates_exact_expectation():
    """Toy analytic E[indicator(edge (0,2))] derivative wrt parent logit."""
    torch.manual_seed(21)
    c=kernel(p=3,filters=1)
    with torch.no_grad():
        c.stop_logits.fill_(-25)
        c.optional_logits.fill_(-25)
        c.parent_logits[0,2,0] = 0.7
        c.parent_logits[0,2,1] = -0.7
    prob=c.parent_logits[0,2,:2].softmax(0)[0]
    # When theta=0, edge 0-2 is present iff parent==0.
    true_grad=float((prob*(1-prob)).detach())
    vals=[]
    for _ in range(400):
        tr=c.sample_trace(0)
        vals.append((float((0,2) in tr.edges),tr.log_prob))
    # Independent zero baseline (valid but noisier) using simple indicator reward.
    estimated=torch.stack([lp*(1 if y else 0) for y,lp in vals]).mean()
    d=torch.autograd.grad(estimated,c.parent_logits)[0][0,2,0]
    assert abs(float(d)-true_grad)<0.07


def test_deterministic_eval_and_multiple_kernels():
    torch.manual_seed(7)
    c=kernel(p=4,filters=2,B=6,kernels='wl+gl')
    x,e,b,egos=small_data()
    c.eval()
    with torch.no_grad():
        one=c(x,e,b,egonets_data=egos)
        two=c(x,e,b,egonets_data=egos)
    assert one.shape==(4,4)
    assert torch.equal(one,two)
    assert torch.equal(one[:,:2],one[:,2:])
    assert c.last_diagnostics['input_label_mode']=='argmax'


def test_multilayer_gradient_reaches_first_layer_masks():
    torch.manual_seed(14)
    c1=kernel(p=4,filters=3,labels=3,B=18)
    c2=kernel(p=4,filters=2,labels=3,B=18,mode='categorical')
    with torch.no_grad():
        c1.stop_logits.fill_(-2)
        c2.stop_logits.fill_(-2)
    x,e,b,egos=small_data()
    c1.train();c2.train()
    h=c1(x,e,b,egonets_data=egos)
    assert h.requires_grad and h.shape==(4,3)
    y=c2(h,e,b,egonets_data=egos)
    assert y.shape==(4,2)
    # Mimic a downstream classifier + nonlinearity.
    y.pow(2).mean().backward()
    for c in (c1,c2):
        for group in ('stop_logits','parent_logits','optional_logits','label_logits'):
            grad=getattr(c,group).grad
            assert grad is not None and torch.isfinite(grad).all(),group
    assert c1.label_logits.grad.abs().sum()>0
    assert c2.label_logits.grad.abs().sum()>0


def test_incompatible_input_channels_rejected():
    c=kernel(filters=2,labels=3,B=2)
    x,e,b,egos=small_data()
    with pytest.raises(ValueError,match='expected input'):
        c(x[:,:2],e,b,egonets_data=egos)
