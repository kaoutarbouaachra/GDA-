"""Reproducible CPU forward-latency benchmark for GKNN/CAM.

Measurement excludes dataset loading/collation and records model.eval() forward
on pre-built batches. It includes stochastic CAM sampling, GraKeL, and other
forward computations. The denominator is NUMBER OF GRAPHS, not nodes or batches.
The test split may be used for timing only; labels are not used for selection.
"""
from __future__ import annotations

import time
from contextlib import nullcontext

import numpy as np
import torch


def _sync_if_cuda(model):
    if any(p.is_cuda for p in model.parameters()):
        torch.cuda.synchronize()


def _graph_count(batch):
    if hasattr(batch, 'num_graphs'):
        return int(batch.num_graphs)
    if hasattr(batch, 'y'):
        return int(batch.y.numel())
    raise TypeError('Each batch must provide num_graphs or graph-level y')


def benchmark_inference(model, loader, *, warmup=2, repeats=5,
                        split='test', seed=1729):
    """Benchmark forward inference, after the best checkpoint is loaded.

    Loader is materialized BEFORE the timer, so disk access and batch collation
    are not included; it can be too costly to cache very large datasets.
    No gradients, but uses torch.no_grad() (compatible with custom autograd).
    Stochastic RNG is RESET per repeat so timing comparisons see the same
    random streams; RNG state is restored on exit.
    """
    warmup=int(warmup)
    repeats=int(repeats)
    if repeats < 1 or warmup < 0:
        raise ValueError('repeats must be >=1 and warmup >=0')
    batches=list(loader)
    n_graphs=sum(_graph_count(b) for b in batches)
    if n_graphs == 0:
        raise ValueError('Cannot benchmark an empty split')
    old_training=bool(model.training)
    old_torch_rng=torch.random.get_rng_state()
    old_np_rng=np.random.get_state()
    old_cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    times=[]
    try:
        model.eval()
        with torch.no_grad():
            for iteration in range(warmup+repeats):
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                _sync_if_cuda(model)
                t0=time.perf_counter()
                for batch in batches:
                    model(batch)
                _sync_if_cuda(model)
                seconds=float(time.perf_counter()-t0)
                if iteration >= warmup:
                    times.append(seconds)
    finally:
        model.train(old_training)
        torch.random.set_rng_state(old_torch_rng)
        np.random.set_state(old_np_rng)
        if old_cuda_rng is not None:
            torch.cuda.set_rng_state_all(old_cuda_rng)
    per_graph=1000.0*np.asarray(times,dtype=float)/n_graphs
    return dict(split=str(split),n_graphs=n_graphs,n_batches=len(batches),
                warmup_passes=warmup,repeats=repeats,
                batch_size_graphs=[_graph_count(b) for b in batches],
                measured_total_seconds=times,
                per_graph_ms_mean=float(np.mean(per_graph)),
                per_graph_ms_sd=float(np.std(per_graph,ddof=1)) if repeats>1 else None,
                per_graph_ms_median=float(np.median(per_graph)),
                per_graph_ms_each_repeat=per_graph.tolist(),
                measurement='eval_no_grad_full_forward_excluding_loader_and_collation',
                note='Includes sampling and GraKeL forward; CPU allocation and sample budget affect latency.')
