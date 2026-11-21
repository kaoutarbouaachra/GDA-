"""Connected Adaptive Masks (CAM) for the Cosmo et al. GKNN architecture.

A CAM mask is a learned probability distribution over CONNECTED labeled graphs:
    rho: stopping decisions (learnable size, bounded by nodes),
    pi: compulsory categorical parent for each new vertex,
    theta: optional Bernoulli edges conditioned on the chosen parent,
    phi: categorical node labels.

For each filter, forward computes an iid Monte Carlo estimate of
    z_i(v) = E_T[graph_kernel(H(T), N_G^r(v))].
The score-function straight-through surrogate gives an unbiased estimate of
    d z_i(v) / d psi_i
(not, for finite samples, an unbiased estimate of d CE(E[z]) / d psi).

If input_label_mode='categorical', node input labels are sampled from
softmax(x / input_temperature) and differentiated with an additional score
estimator. This supports multi-layer CAM *but differs from the original
GKNN's argmax handling of intermediate node labels*. For comparisons with
published GKNN, begin with a single layer and input_label_mode='argmax'.

CPU/GraKeL backend. No PyG import is needed when precomputed egonets are given.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

Edge = Tuple[int, int]


@dataclass
class CAMTrace:
    filter_index: int
    size: int
    parents: Tuple[int, ...]  # index t=0 is -1
    optional_edges: Tuple[Tuple[int, int, int], ...]  # (t,j,0/1), j!=parent[t]
    edges: Tuple[Edge, ...]  # sorted undirected edges (u,v) with u<v
    labels: Tuple[int, ...]
    log_prob: torch.Tensor  # differentiable log q_psi(trace)

    def graph_key(self):
        return (self.size, self.edges, self.labels)


class CAMKernel(nn.Module):
    """A GKNN graph-kernel convolution with a CAM distribution per filter.

    The signature matches layers.GKernel so Model.forward remains unchanged.
    Args:
      nodes: p_max, the maximum number of mask vertices
      labels: number of allowable vertex labels in this layer
      filters: number of learned CAM distributions
      input_label_mode: 'argmax' for fixed/discrete input node labels,
        or 'categorical' for a differentiable multi-layer probabilistic bridge
      kernel_factories: optional test-only mapping name -> kernel constructor
    """
    def __init__(self, nodes: int, labels: int, filters: int = 8,
                 max_cc=None, hops: int = 1, kernels: str = 'wl',
                 normalize: bool = True, store_fit: bool = False,
                 egonets_data=None, mc_samples: int = 8,
                 eval_samples: int = 32, initial_stop_prob: float = 0.1,
                 eval_seed: int = 1729,
                 input_label_mode: str = 'argmax',
                 input_temperature: float = 1.0,
                 kernel_factories: Optional[Dict[str, Callable]] = None):
        super().__init__()
        if min(nodes, labels, filters, mc_samples, eval_samples) < 1:
            raise ValueError('nodes, labels, filters and sample counts must be >= 1')
        if not 0 < initial_stop_prob < 1:
            raise ValueError('initial_stop_prob must be in (0,1)')
        if input_label_mode not in ('argmax', 'categorical'):
            raise ValueError('input_label_mode must be argmax or categorical')
        if input_temperature <= 0:
            raise ValueError('input_temperature must be positive')
        self.nodes, self.labels, self.filters = int(nodes), int(labels), int(filters)
        self.hops = int(hops)
        self.kernels = str(kernels)
        self.kernel_names = self.kernels.split('+')
        self.normalize = bool(normalize)
        self.mc_samples = int(mc_samples)
        self.eval_samples = int(eval_samples)
        self.eval_seed = int(eval_seed)
        self.input_label_mode = input_label_mode
        self.input_temperature = float(input_temperature)
        self.kernel_factories = kernel_factories or {}
        self.store_fit = False  # sampled masks cannot use original static cache
        self.stored = False
        stop_logit = float(np.log(initial_stop_prob / (1.0 - initial_stop_prob)))
        # Each learned filter has its own four groups of logits.
        self.stop_logits = nn.Parameter(torch.full((filters, max(0, nodes-1)), stop_logit))
        self.parent_logits = nn.Parameter(torch.zeros(filters, nodes, nodes))
        self.optional_logits = nn.Parameter(torch.zeros(filters, nodes, nodes))
        self.label_logits = nn.Parameter(torch.zeros(filters, nodes, labels))
        with torch.no_grad():
            self.parent_logits.normal_(std=0.1)
            self.optional_logits.normal_(mean=-1.0, std=0.2)
            self.label_logits.normal_(std=0.1)
        self.last_diagnostics: Dict[str, object] = {}

    def prefit(self, x, edge_index):
        """Retained for compatibility; CAM uses fresh kernel fits per batch."""
        return None

    def size_probabilities(self):
        """Size distribution [filters,p_max], normalized including forced stop."""
        if self.nodes == 1:
            return self.label_logits.new_ones((self.filters, 1))
        rho = self.stop_logits.sigmoid()
        survive = rho.new_ones(self.filters)
        out = []
        for t in range(self.nodes - 1):
            out.append(survive * rho[:, t])
            survive = survive * (1-rho[:, t])
        out.append(survive)
        return torch.stack(out, dim=1)

    def edge_marginals(self, filter_index: int, t: int):
        """P((j,t) is present | P >= t+1), j=0,...,t-1 (zero indexed)."""
        if not 0 <= filter_index < self.filters or not 1 <= t < self.nodes:
            raise ValueError('invalid filter or vertex index')
        pi = self.parent_logits[filter_index, t, :t].softmax(dim=0)
        theta = self.optional_logits[filter_index, t, :t].sigmoid()
        return pi + (1-pi)*theta

    def sample_trace(self, filter_index: int) -> CAMTrace:
        if not 0 <= filter_index < self.filters:
            raise IndexError(filter_index)
        if self.label_logits.device.type != 'cpu':
            raise RuntimeError('CAM with GraKeL runs on CPU')
        # Stop after at least one vertex, force stopping at p_max.
        logp = self.label_logits[filter_index, 0, 0] * 0.0
        p = self.nodes
        for size in range(1, self.nodes):
            logits = self.stop_logits[filter_index, size-1]
            stop = bool(torch.bernoulli(torch.sigmoid(logits.detach())).item())
            logp = logp + (F.logsigmoid(logits) if stop else F.logsigmoid(-logits))
            if stop:
                p = size
                break
        parent = [-1] * p
        optional = []
        edges = []
        for t in range(1, p):
            parent_lps = F.log_softmax(self.parent_logits[filter_index, t, :t], dim=0)
            a = int(torch.multinomial(parent_lps.detach().exp(), 1).item())
            parent[t] = a
            logp = logp + parent_lps[a]
            edges.append((a, t))
            for j in range(t):
                if j == a:
                    continue  # the mandatory edge is not drawn twice
                logits = self.optional_logits[filter_index, t, j]
                b = bool(torch.bernoulli(torch.sigmoid(logits.detach())).item())
                logp = logp + (F.logsigmoid(logits) if b else F.logsigmoid(-logits))
                optional.append((t, j, int(b)))
                if b:
                    edges.append((j, t))
        labs = []
        for u in range(p):
            label_lps = F.log_softmax(self.label_logits[filter_index, u], dim=0)
            l = int(torch.multinomial(label_lps.detach().exp(), 1).item())
            logp = logp + label_lps[l]
            labs.append(l)
        return CAMTrace(filter_index, p, tuple(parent), tuple(optional),
                        tuple(sorted(set(edges))), tuple(labs), logp)

    def trace_log_probability(self, tr: CAMTrace):
        """Re-evaluate the EXACT probability of a supplied generation trace."""
        f, p = tr.filter_index, tr.size
        if not 0 <= f < self.filters or not 1 <= p <= self.nodes:
            raise ValueError('invalid filter or mask size')
        logp = self.label_logits[f, 0, 0] * 0.0
        for size in range(1, self.nodes):
            stop = (size == p)
            logit = self.stop_logits[f, size-1]
            logp = logp + (F.logsigmoid(logit) if stop else F.logsigmoid(-logit))
            if stop:
                break
        if len(tr.parents) != p or tr.parents[0] != -1:
            raise ValueError('invalid parent vector')
        opts = {(t,j): b for t,j,b in tr.optional_edges}
        expected = set()
        expected_opt_count = 0
        for t in range(1,p):
            a = tr.parents[t]
            if not 0 <= a < t:
                raise ValueError('invalid parent')
            logp = logp + F.log_softmax(self.parent_logits[f,t,:t],dim=0)[a]
            expected.add((a,t))
            for j in range(t):
                if j == a:
                    continue
                expected_opt_count += 1
                if (t,j) not in opts or opts[(t,j)] not in (0,1):
                    raise ValueError('invalid optional edge')
                b = opts[(t,j)]
                theta = self.optional_logits[f,t,j]
                logp = logp + (F.logsigmoid(theta) if b else F.logsigmoid(-theta))
                if b:
                    expected.add((j,t))
        if len(opts) != expected_opt_count or tuple(sorted(expected)) != tr.edges:
            raise ValueError('inconsistent trace edges')
        if len(tr.labels) != p or any(not 0 <= l < self.labels for l in tr.labels):
            raise ValueError('invalid vertex labels')
        for u,l in enumerate(tr.labels):
            logp = logp + F.log_softmax(self.label_logits[f,u],dim=0)[l]
        return logp

    @staticmethod
    def as_grakel_graph(tr: CAMTrace):
        """GraKeL labeled undirected adjacency dictionary, preserving singletons."""
        adj = {u: {} for u in range(tr.size)}
        for u,v in tr.edges:
            adj[u][v] = 1.0
            adj[v][u] = 1.0
        return (adj, {u: int(l) for u,l in enumerate(tr.labels)})

    def _new_kernel(self, name: str):
        if name in self.kernel_factories:
            return self.kernel_factories[name]()
        from grakel.kernels import (WeisfeilerLehman, WeisfeilerLehmanOptimalAssignment,
                                    Propagation, GraphletSampling, RandomWalkLabeled,
                                    PyramidMatch)
        supported = {
            'wl': lambda: WeisfeilerLehman(n_iter=3, normalize=self.normalize),
            'wloa': lambda: WeisfeilerLehmanOptimalAssignment(n_iter=3, normalize=self.normalize),
            'prop': lambda: Propagation(normalize=self.normalize),
            'gl': lambda: GraphletSampling(normalize=self.normalize),
            'rw': lambda: RandomWalkLabeled(normalize=self.normalize),
            'py': lambda: PyramidMatch(normalize=self.normalize),
        }
        if name not in supported:
            raise ValueError('unsupported graph kernel: ' + name)
        return supported[name]()

    @staticmethod
    def _egonet_lists(x, edge_index, egonets_data, hops):
        """Return per-node global indices and local edges, not graph labels."""
        if egonets_data is None:
            from torch_geometric.utils import k_hop_subgraph
            node_lists, edge_lists = [], []
            for v in range(x.shape[0]):
                nodes, edges, _, _ = k_hop_subgraph(
                    v, hops, edge_index, relabel_nodes=True, num_nodes=x.shape[0])
                node_lists.append(nodes.detach().cpu().numpy())
                edge_lists.append(edges.detach().cpu().numpy())
        else:
            node_lists, edge_lists = egonets_data
        if len(node_lists) != x.shape[0] or len(edge_lists) != x.shape[0]:
            raise ValueError('expect one precomputed egonet per input node')
        node_lists = [np.asarray(n, dtype=int).reshape(-1) for n in node_lists]
        edge_lists = [np.asarray(e, dtype=int) for e in edge_lists]
        for nodes, edges in zip(node_lists,edge_lists):
            if len(nodes) == 0 or (nodes < 0).any() or (nodes >= len(x)).any():
                raise ValueError('egonet contains invalid global node index')
            if edges.size and (edges.ndim != 2 or (edges.shape[0] != 2 and edges.shape[1] != 2)):
                raise ValueError('egonet edges must be (2,E) or (E,2)')
        return node_lists, edge_lists

    @staticmethod
    def _grakel_inputs(labels, node_lists, edge_lists):
        """Make GraKeL graphs from labels and egonet topology."""
        graphs = []
        for nidx, edge in zip(node_lists,edge_lists):
            lab = labels[nidx]
            if edge.size == 0:
                pairs = []
            elif edge.shape[0] == 2:
                pairs = list(map(tuple, edge.T.tolist()))
            else:
                pairs = list(map(tuple, edge.tolist()))
            adj = {j: {} for j in range(len(nidx))}
            for u,v in pairs:
                u,v = int(u),int(v)
                if not (0<=u<len(nidx) and 0<=v<len(nidx)):
                    raise ValueError('egonet edges contain nonlocal indices')
                adj[u][v] = adj[v][u] = 1.0
            graphs.append((adj, {j:int(y) for j,y in enumerate(lab)}))
        return graphs

    def _evaluate(self, mask_traces, graphs):
        """Evaluate B masks per filter against one collection of input egonets.

        mask_traces are indexed [B][filters]. Returns array [B,N,filters*kernels]
        with kernel-major ordering, matching the original GKernel.
        """
        B = len(mask_traces)
        unique, unique_traces, gather = {}, [], []
        for row in mask_traces:
            for tr in row:
                key = tr.graph_key()
                if key not in unique:
                    unique[key] = len(unique_traces)
                    unique_traces.append(tr)
                gather.append(unique[key])
        masks = [self.as_grakel_graph(tr) for tr in unique_traces]
        sims_by_kernel = []
        for name in self.kernel_names:
            kern = self._new_kernel(name)
            kern.fit(graphs)
            sim = np.asarray(kern.transform(masks),dtype=np.float64)
            if sim.shape != (len(masks),len(graphs)):
                raise RuntimeError('unexpected GraKeL kernel output shape: '+str(sim.shape))
            # Normalized kernels give 0/0 = NaN when a graph has an empty feature
            # vector (e.g. 'gl' on graphs smaller than its graphlet size). That
            # means "no shared features": 0, as original GKNN's np.nan_to_num.
            sim = np.where(np.isnan(sim), 0.0, sim)
            if not np.isfinite(sim).all():
                raise FloatingPointError('nonfinite GraKeL kernel evaluation for '+name)
            sims_by_kernel.append(sim[gather].reshape(B,self.filters,len(graphs)))
        # B, kernels, filters, N -> B, N, kernels, filters -> B, N, K*m
        return np.stack(sims_by_kernel,axis=1).transpose(0,3,1,2).reshape(B,len(graphs),-1)

    def forward(self, x, edge_index, batch=None, not_used=None, fixedges=None,
                node_indexes=None, egonets_data=None, optimize=True):
        if x.device.type != 'cpu':
            raise RuntimeError('CAM/GraKeL currently runs on CPU')
        if x.ndim != 2 or x.shape[1] != self.labels:
            raise ValueError(f'expected input [N,{self.labels}], got {tuple(x.shape)}')
        B = self.mc_samples if self.training else self.eval_samples
        node_lists, edge_lists = self._egonet_lists(x, edge_index, egonets_data, self.hops)
        # Fixed sampling seed for reproducible checkpoint comparison on validation.
        rng_guard = (contextlib.nullcontext() if self.training else torch.random.fork_rng(devices=[]))
        with rng_guard:
            if not self.training:
                torch.manual_seed(self.eval_seed)
            mask_traces = [[self.sample_trace(i) for i in range(self.filters)] for _ in range(B)]
            if self.input_label_mode == 'argmax':
                input_y = x.detach().argmax(-1).numpy()
                if (input_y >= self.labels).any():
                    raise ValueError('input labels not in CAM alphabet')
                graph_egos = self._grakel_inputs(input_y,node_lists,edge_lists)
                kvals_np = self._evaluate(mask_traces, graph_egos)
                input_scores = None
            else:
                # Explicit probabilistic alternative to original GKNN argmax at
                # intermediate layers: Y_u ~ Cat(softmax(x_u / temperature)).
                logp = F.log_softmax(x/self.input_temperature,dim=1)
                kvals_np = np.empty((B,len(x),self.filters*len(self.kernel_names)),dtype=np.float64)
                score_egos = []
                for b in range(B):
                    y = torch.multinomial(logp.detach().exp(),1).flatten()
                    lp_nodes = logp.gather(1,y[:,None]).flatten()
                    score_egos.append(torch.stack([lp_nodes[inds.tolist()].sum() for inds in node_lists]))
                    egos = self._grakel_inputs(y.detach().numpy(),node_lists,edge_lists)
                    kvals_np[b:b+1] = self._evaluate([mask_traces[b]], egos)
                input_scores = torch.stack(score_egos)  # [B,N]
        values = torch.as_tensor(kvals_np,dtype=x.dtype,device=x.device)
        mean_values = values.mean(dim=0)
        if self.training and optimize and torch.is_grad_enabled():
            if B > 1:
                loo = (values.sum(dim=0,keepdim=True)-values)/(B-1)
            else:
                loo = torch.zeros_like(values)
            centered_values = (values-loo).detach()
            trace_scores = torch.stack([torch.stack([tr.log_prob for tr in row])
                                        for row in mask_traces])
            trace_scores = trace_scores.repeat(1,len(self.kernel_names))
            # For each output (node,filter), differentiate only its mask trace.
            mask_surrogate = (centered_values*trace_scores[:,None,:]).mean(dim=0)
            out = mean_values + mask_surrogate - mask_surrogate.detach()
            if input_scores is not None and x.requires_grad:
                # Local ego score, excluding node labels outside that neighborhood.
                # For each v, K(T,Y_ego_v) has expectation differentiated wrt x.
                input_surrogate = (centered_values*input_scores[:,:,None]).mean(dim=0)
                out = out + input_surrogate - input_surrogate.detach()
        else:
            out = mean_values.detach()
        with torch.no_grad():
            sz = self.size_probabilities()
            sizes = torch.as_tensor([tr.size for row in mask_traces for tr in row],dtype=x.dtype)
            self.last_diagnostics = {
                'expected_size_per_filter': (sz * torch.arange(1,self.nodes+1,device=sz.device)[None,:]).sum(1).tolist(),
                'mean_sampled_size': float(sizes.mean()),
                'unique_mask_graphs': len(set(tr.graph_key() for row in mask_traces for tr in row)),
                'total_mask_traces': B*self.filters,
                'requested_kernel_values': B*self.filters*len(self.kernel_names)*len(x),
                'input_label_mode': self.input_label_mode,
            }
        return out
