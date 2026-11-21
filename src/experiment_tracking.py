"""Non-invasive experiment tracking for original GKNN and connected CAM.

Requires NumPy, PyTorch, PyTorch Lightning; does not import PyG or GraKeL.
It attaches to an existing Model with conv_layers, and records rich, replayable
outputs for independent post-hoc figures. All code runs on CPU.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import subprocess
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl


def json_safe(obj):
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        if obj.numel() == 1:
            return obj.detach().cpu().item()
        return obj.detach().cpu().tolist()
    if isinstance(obj, (float, int, str, bool)) or obj is None:
        if isinstance(obj, float) and not math.isfinite(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [json_safe(v) for v in obj]
    return str(obj)


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(json_safe(data), indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def append_csv(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open('a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow({k: json_safe(v) for k, v in row.items()})


def git_info(root='.'):  # a HEAD hash is not sufficient for dirty worktrees
    result = {}
    for name, args in [('commit', ['git', 'rev-parse', 'HEAD']),
                       ('status', ['git', 'status', '--porcelain'])]:
        try:
            p = subprocess.run(args, cwd=root, capture_output=True, text=True, timeout=5)
            result[name] = p.stdout.strip() if p.returncode == 0 else None
        except (OSError, subprocess.TimeoutExpired):
            result[name] = None
    return result


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def software_environment():
    modules = ['numpy', 'torch', 'pytorch_lightning', 'torch_geometric', 'grakel',
               'scipy', 'sklearn', 'networkx']
    versions = {}
    for name in modules:
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, '__version__', 'installed')
        except Exception as exc:
            versions[name] = 'unavailable: ' + type(exc).__name__
    return dict(python=platform.python_version(), host=platform.node(),
                platform=platform.platform(), versions=versions,
                slurm={k: os.environ.get(k) for k in (
                    'SLURM_JOB_ID', 'SLURM_ARRAY_TASK_ID', 'SLURM_CPUS_PER_TASK',
                    'SLURM_MEM_PER_NODE', 'SLURM_JOB_NODELIST')},
                threads={k: os.environ.get(k) for k in (
                    'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')})


def response_stats(matrix, const_std=1e-6):
    """All calculations are after concatenating the COMPLETE split, not batches.

    Active columns only are used for Pearson correlations. Constant columns are
    counted separately; they are not spuriously counted as decorrelated.
    Entropic and participation effective ranks are different definitions.
    """
    x = np.asarray(matrix, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError('Expected [number_of_observations, number_of_filters]')
    n, d = x.shape
    std = x.std(axis=0) if n else np.zeros(d)
    active = np.isfinite(std) & (std > const_std)
    d_active = int(active.sum())
    result = dict(n_observations=n, n_filters=d, active_filters=d_active,
                  constant_filters=int(d-d_active), fraction_active=d_active / max(d, 1))
    if n < 2 or d_active < 2:
        result.update(mean_abs_corr=None, mean_sq_corr=None, effective_rank=None,
                      participation_rank=None, eigenvalues=[], top_eigen_fraction=None,
                      abs_corr_gt_09=None)
        return result, np.full((d, d), np.nan, dtype=float)
    xc = x[:, active] - x[:, active].mean(0)
    norms = np.linalg.norm(xc, axis=0)
    u = xc / norms
    c = np.clip(u.T @ u, -1.0, 1.0)
    c = (c + c.T) / 2
    eig = np.maximum(np.linalg.eigvalsh(c)[::-1], 0.0)
    pe = eig / max(float(eig.sum()), 1e-15)
    keep = pe > 0
    erank = float(np.exp(-(pe[keep] * np.log(pe[keep])).sum()))
    prank = float(eig.sum()**2 / max(np.square(eig).sum(), 1e-15))
    off = np.abs(c[np.triu_indices(d_active, k=1)])
    result.update(mean_abs_corr=float(off.mean()), mean_sq_corr=float((off**2).mean()),
                  effective_rank=erank, participation_rank=prank,
                  eigenvalues=eig.tolist(), top_eigen_fraction=float(pe[0]),
                  abs_corr_gt_09=float((off > 0.9).mean()))
    full_c = np.full((d, d), np.nan, dtype=float)
    ix = np.flatnonzero(active)
    full_c[np.ix_(ix, ix)] = c
    return result, full_c


def graph_level_metrics(logits, true_labels, n_nodes):
    """Exact pooled graph-level metrics at a fixed checkpoint and split."""
    from sklearn.metrics import (accuracy_score, balanced_accuracy_score, confusion_matrix,
                                 log_loss, roc_auc_score, average_precision_score)
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(true_labels, dtype=int).ravel()
    if z.ndim != 2 or len(z) != len(y):
        raise ValueError('logits must be [G,C], labels [G]')
    if not np.isfinite(z).all():
        raise FloatingPointError('nonfinite logits')
    s = z - z.max(axis=1, keepdims=True)
    p = np.exp(s) / np.exp(s).sum(axis=1, keepdims=True)
    pred = p.argmax(1)
    labels = np.arange(z.shape[1])
    row = dict(n_graphs=len(y), accuracy=float(accuracy_score(y,pred)),
               balanced_accuracy=float(balanced_accuracy_score(y,pred)),
               cross_entropy=float(log_loss(y,p,labels=labels)),
               confusion_matrix=confusion_matrix(y,pred,labels=labels).tolist(),
               true_class_counts=np.bincount(y,minlength=z.shape[1]).tolist(),
               pred_class_counts=np.bincount(pred,minlength=z.shape[1]).tolist(),
               mean_nodes=float(np.asarray(n_nodes).mean()))
    if z.shape[1] == 2 and len(np.unique(y)) == 2:
        row['roc_auc'] = float(roc_auc_score(y,p[:,1]))
        row['average_precision'] = float(average_precision_score(y,p[:,1]))
        nodes = np.asarray(n_nodes)
        row['size_only_auc_on_this_split'] = float(roc_auc_score(y,-nodes))
        row['size_only_auc_not_training_baseline'] = True
    else:
        row['roc_auc'] = None
        row['average_precision'] = None
    return row, p


def _original_mask(layer):
    """Decode the hard mask *as used by the legacy GKernel*, then keep largest CC.

    Legacy max_comp has a special case for a mask without edges; rather than
    invent a dominant singleton, all isolated labeled vertices are retained.
    """
    import networkx as nx
    padd = layer.Padd.detach().cpu().numpy()
    xp = layer.Xp.detach().cpu().numpy()
    masks = []
    for i in range(len(padd)):
        n = padd.shape[-1]
        adj = (padd[i] > 0)
        g = nx.Graph()
        g.add_nodes_from(range(n))
        g.add_edges_from((j, k) for j in range(n) for k in range(j+1,n) if adj[j,k])
        all_edges = g.number_of_edges()
        if not all_edges:
            # Original max_comp returns singleton loops for all vertices.
            g.add_edges_from((j,j) for j in range(n))
        if all_edges:
            comps = list(nx.connected_components(g))
            largest = max(comps,key=len)
            g = g.subgraph(largest).copy()
        mapping = {u:j for j,u in enumerate(sorted(g.nodes()))}
        g = nx.relabel_nodes(g,mapping)
        chosen = sorted(mapping.keys())
        labels = [int(xp[i,u].argmax()) for u in chosen]
        masks.append(dict(filter=i, n_nodes=len(chosen), labels=labels,
                          edges=[list(map(int,e)) for e in sorted(g.edges())],
                          original_n_nodes=n, pre_component_edges=all_edges,
                          largest_component_fraction=len(chosen)/max(n,1)))
    return masks


def _cam_parameters_and_samples(layer, samples=256, seed=1729):
    """Snapshot exact logits plus interpretable derived CAM statistics.

    Frequencies of sampled graph keys are NOT exact marginal graph probabilities.
    Analytic edge covariance is conditional on the vertex t existing.
    """
    out = {}
    size_prob=layer.size_probabilities().detach().cpu()
    out['size_probabilities'] = size_prob.tolist()
    out['expected_size_per_filter'] = (size_prob *
                                      torch.arange(1,layer.nodes+1)[None,:]).sum(1).tolist()
    # Exact entropy of the GENERATION TRACE, not of the marginal graph law.
    # Distinct traces can construct the same graph: H(graph) <= H(trace).
    trace_entropy=[]
    with torch.no_grad():
        for f in range(layer.filters):
            q=size_prob[f].double()
            ent=-(q*q.clamp_min(1e-15).log()).sum()
            for t in range(layer.nodes):
                survival=q[t:].sum()
                lp=torch.softmax(layer.label_logits[f,t].double(),dim=-1)
                ent=ent + survival*(-(lp*lp.clamp_min(1e-15).log()).sum())
                if t>=1:
                    pi=torch.softmax(layer.parent_logits[f,t,:t].double(),dim=-1)
                    ent=ent + survival*(-(pi*pi.clamp_min(1e-15).log()).sum())
                    theta=torch.sigmoid(layer.optional_logits[f,t,:t].double())
                    bern=-(theta*theta.clamp_min(1e-15).log() +
                           (1-theta)*(1-theta).clamp_min(1e-15).log())
                    ent=ent + survival*((1-pi)*bern).sum()
            trace_entropy.append(float(ent))
    out['exact_trace_entropy_per_filter'] = trace_entropy
    edges = []
    for f in range(layer.filters):
        marg = []
        covs = []
        for t in range(1,layer.nodes):
            pi = torch.softmax(layer.parent_logits[f,t,:t],dim=0).detach().cpu().numpy()
            th = torch.sigmoid(layer.optional_logits[f,t,:t]).detach().cpu().numpy()
            p = pi + (1-pi)*th
            # P(X_j=X_k=1)=pi_j*th_k + pi_k*th_j + (1-pi_j-pi_k)*th_j*th_k
            pp = np.zeros((t,t),dtype=float)
            for j in range(t):
                for k in range(t):
                    if j==k:
                        pp[j,k] = p[j]
                    else:
                        pp[j,k] = (pi[j]*th[k] + pi[k]*th[j]
                                   + (1-pi[j]-pi[k])*th[j]*th[k])
            cov = pp - p[:,None]*p[None,:]
            marg.append(p.tolist())
            covs.append(cov.tolist())
        edges.append(dict(filter=f, edge_marginal_conditional=marg,
                          edge_covariance_conditional=covs))
    out['edge_statistics'] = edges
    from collections import Counter
    freqs = []
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(seed))
        for f in range(layer.filters):
            traces = [layer.sample_trace(f) for _ in range(samples)]
            cnt = Counter(t.graph_key() for t in traces)
            top = []
            for key, number in cnt.most_common(5):
                p, ed, labs = key
                top.append(dict(n_nodes=p, edges=[list(e) for e in ed],
                                labels=list(labs), frequency=number/samples))
            freqs.append(dict(filter=f, sample_count=samples,
                              unique_labelled_graphs=len(cnt), top_graphs=top,
                              empirical_mode_mass=cnt.most_common(1)[0][1]/samples))
    out['graph_samples'] = freqs
    return out


def save_mask_snapshot(model, path, epoch, cam_graph_samples=256, sample_seed=1729):
    """Store exact tensor state for later rendering, plus decoded sidecars."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save({k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
               path / ('state_%04d.pt' % epoch))
    out = {'epoch':int(epoch),'layers':[]}
    for j, layer in enumerate(model.conv_layers):
        family = type(layer).__name__
        detail = {'layer':j+1,'family':family}
        with torch.no_grad():
            if hasattr(layer,'stop_logits') and hasattr(layer,'sample_trace'):
                detail.update(_cam_parameters_and_samples(layer,cam_graph_samples,sample_seed+j))
            elif hasattr(layer,'Padd') and hasattr(layer,'Xp'):
                detail['masks'] = _original_mask(layer)
                edge_p=torch.sigmoid(layer.Padd.detach().double())
                lab_p=torch.softmax(layer.Xp.detach().double(),dim=-1)
                h_edge=-(edge_p*edge_p.clamp_min(1e-15).log()+
                         (1-edge_p)*(1-edge_p).clamp_min(1e-15).log())
                a,b=torch.triu_indices(edge_p.shape[1],edge_p.shape[2],offset=1)
                detail['mean_bernoulli_edge_entropy_per_filter']=h_edge[:,a,b].mean(1).tolist()
                detail['mean_label_entropy_per_filter']=(-(lab_p*lab_p.clamp_min(1e-15).log())
                                                           .sum(-1).mean(-1)).tolist()
                detail['entropy_note']='Parameter-wise entropies, NOT entropy of the decoded/connected graph distribution'
            else:
                detail['unsupported_mask_layer'] = True
        out['layers'].append(detail)
    if hasattr(model,'mask') and model.mask is not None:
        out['global_filter_weights'] = model.mask.detach().cpu().tolist()
    write_json(path / ('masks_%04d.json' % epoch),out)
    return out


@contextmanager
def _capture_gkernel_outputs(model):
    """Capture *pre-learned-mask-weight* kernel responses, not scaled responses."""
    features = []
    hooks = []
    for layer in model.conv_layers:
        hooks.append(layer.register_forward_hook(
            lambda module, args, output: features.append(output.detach().cpu().numpy().copy())))
    try:
        yield features
    finally:
        for hook in hooks:
            hook.remove()


def export_predictions_and_responses(model, data_loader, output_path, split_name,
                                     original_indices=None, const_std=1e-6):
    """Export a COMPLETE split, including node and pooled representations.

    `data_loader` must have shuffle=False; graph IDs are saved in exactly that order.
    No test split is used for model selection. Saving all node features per split
    can take substantial disk space. Use only at the best checkpoint.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True,exist_ok=True)
    was_training = model.training
    model.eval()
    logits_parts, y_parts, n_parts, batch_parts = [],[],[],[]
    edge_parts, node_label_parts = [], []
    node_offset = 0
    raws, scales, pools = defaultdict(list),defaultdict(list),defaultdict(list)
    graph_offset = 0
    with torch.no_grad(), _capture_gkernel_outputs(model) as raw_outputs:
        for batch in data_loader:
            raw_outputs.clear()
            logits, responses, _ = model(batch)
            ids = batch.batch.detach().cpu().numpy().astype(int)
            if len(raw_outputs) != len(responses):
                raise RuntimeError('Cannot capture all GKernel layer outputs')
            n_graphs = int(batch.num_graphs) if hasattr(batch,'num_graphs') else int(ids.max()+1)
            n_nodes = np.bincount(ids, minlength=n_graphs)
            logits_parts.append(logits.detach().cpu().numpy())
            y_parts.append(batch.y.detach().cpu().numpy().reshape(-1))
            n_parts.append(n_nodes)
            batch_parts.append(ids+graph_offset)
            edge_parts.append(batch.edge_index.detach().cpu().numpy().astype(int) + node_offset)
            node_label_parts.append(batch.x.detach().cpu().numpy().argmax(axis=1).astype(int))
            node_offset += len(ids)
            for j,(raw,scaled) in enumerate(zip(raw_outputs,responses)):
                name = f'layer{j+1}'
                sc = scaled.detach().cpu().numpy()
                raws[name].append(raw)
                scales[name].append(sc)
                for kind,arr in [('raw',raw),('scaled',sc)]:
                    psum = np.zeros((n_graphs,arr.shape[1]),dtype=np.float64)
                    np.add.at(psum,ids,arr)
                    pools[f'{name}_{kind}_add'].append(psum)
                    pools[f'{name}_{kind}_mean'].append(psum/np.maximum(n_nodes[:,None],1))
                    pmax=np.full((n_graphs,arr.shape[1]),-np.inf,dtype=np.float64)
                    np.maximum.at(pmax,ids,arr)
                    pools[f'{name}_{kind}_max'].append(pmax)
            graph_offset += n_graphs
    model.train(was_training)
    logits = np.concatenate(logits_parts,axis=0)
    labels = np.concatenate(y_parts,axis=0).astype(int)
    nodes = np.concatenate(n_parts,axis=0).astype(int)
    node_graph = np.concatenate(batch_parts,axis=0).astype(int)
    if original_indices is None:
        original_indices = np.arange(len(labels))
    original_indices=np.asarray(original_indices,dtype=int)
    if len(original_indices)!=len(labels):
        raise ValueError('Saved graph indices length does not match evaluation loader')
    summary, probs = graph_level_metrics(logits, labels, nodes)
    content = dict(logits=logits,probs=probs,labels=labels,n_nodes=nodes,
                   node_graph=node_graph, original_graph_index=original_indices,
                   edge_index=np.concatenate(edge_parts,axis=1),
                   node_input_label=np.concatenate(node_label_parts,axis=0))
    all_stats = {}
    for name,parts in raws.items():
        x = np.concatenate(parts,axis=0)
        content[f'{name}_raw'] = x
        stat, corr=response_stats(x,const_std)
        all_stats[f'{name}_raw'] = stat
        content[f'{name}_raw_correlation']=corr
    for name,parts in scales.items():
        x = np.concatenate(parts,axis=0)
        content[f'{name}_scaled'] = x
        stat,corr=response_stats(x,const_std)
        all_stats[f'{name}_scaled'] = stat
        content[f'{name}_scaled_correlation']=corr
    for name,parts in pools.items():
        x = np.concatenate(parts,axis=0)
        content[name] = x
        stat,corr=response_stats(x,const_std)
        all_stats[name] = stat
        content[f'{name}_correlation']=corr
    np.savez_compressed(out_path,**content)
    summary['split']=split_name
    summary['response_stats']=all_stats
    summary['n_nodes_by_class']={str(c): [int(v) for v in nodes[labels==c]]
                                for c in sorted(set(labels.tolist()))}
    write_json(out_path.with_suffix('.json'),summary)
    return summary


def posthoc_filter_ablation(model, response_file, output_path, pooling='add',
                           maximum_pairs=2048):
    """No graph-kernel evaluations. Ablates pooled filter features using same FC.

    Exactly reproduces zeroing an entire filter response (then pool) for
    add/mean/max pooling, assuming the stored scaled responses are identical
    to the inputs used by Model.fc. For max, zeroing a whole feature before
    max-pooling also gives zero. No classifier retraining is performed.
    Delta is CE(ablated) - CE(original) on the SAVED split, not proof of
    redundancy or of causal feature importance.
    """
    import torch.nn.functional as F
    with np.load(response_file,allow_pickle=False) as npz:
        keys=sorted([k for k in npz.files if k.startswith('layer') and
                     k.endswith('_scaled_'+pooling)],
                    key=lambda k:int(k.split('_')[0][5:]))
        if not keys:
            raise ValueError('No pooled scaled features for '+pooling)
        pooled=np.concatenate([npz[k] for k in keys],axis=1)
        labels=npz['labels'].astype(np.int64)
        logits_saved=npz['logits']
    old=model.training
    model.eval()
    x=torch.tensor(pooled,dtype=next(model.fc.parameters()).dtype,
                   device=next(model.fc.parameters()).device)
    y=torch.tensor(labels,dtype=torch.long,device=x.device)
    with torch.no_grad():
        original=model.fc(x).detach().cpu().numpy()
        reconstructed_max_error=float(np.max(np.abs(original-logits_saved)))
        if reconstructed_max_error>1e-3:
            raise RuntimeError('Pooled FC reconstruction mismatch: '+str(reconstructed_max_error))
        base=float(F.cross_entropy(model.fc(x),y).item())
        m=x.shape[1]
        # A structural filter may contribute multiple features (WL + other
        # kernels). Group the kernel-major columns of the SAME mask together.
        hp=getattr(model,'hparams',None)
        hidden=int(getattr(hp,'hidden',m)) if hp is not None else m
        kernel_names=str(getattr(hp,'kernel','wl')).split('+') if hp is not None else ['wl']
        n_layers=int(getattr(hp,'layers',1)) if hp is not None else 1
        if hidden*len(kernel_names)*n_layers==m:
            groups=[[layer*hidden*len(kernel_names)+k*hidden+f
                     for k in range(len(kernel_names))]
                    for layer in range(n_layers) for f in range(hidden)]
            group_names=[f'layer{layer+1}_filter{f+1}'
                         for layer in range(n_layers) for f in range(hidden)]
        else:
            groups=[[j] for j in range(m)]
            group_names=[f'feature{j+1}' for j in range(m)]
        ng=len(groups)
        single=np.zeros(ng)
        for i,indices in enumerate(groups):
            alt=x.clone();alt[:,indices]=0
            single[i]=float(F.cross_entropy(model.fc(alt),y))-base
        interactions=np.full((ng,ng),np.nan)
        pairs=0
        for i in range(ng):
            for j in range(i+1,ng):
                if pairs>=maximum_pairs:
                    break
                alt=x.clone();alt[:,groups[i]]=0;alt[:,groups[j]]=0
                both=float(F.cross_entropy(model.fc(alt),y))-base
                interactions[i,j]=interactions[j,i]=both-single[i]-single[j]
                pairs+=1
        model.train(old)
    result=dict(base_ce=base,reconstruction_error=reconstructed_max_error,
                pooling=pooling,filter_group_names=group_names,
                filter_feature_groups=groups,single_ablation_delta_ce=single.tolist(),
                pairwise_interaction_delta_ce=interactions.tolist(),
                n_pairwise_ablations=pairs,
                meaning='positive delta means higher CE after removing that filter; pair interaction is nonadditivity, not a redundancy test')
    write_json(output_path,result)
    return result


class RichRecorder(pl.Callback):
    """Lightning callback for epoch traces, mask states, gradients and probe data.

    Param hooks work with the legacy manual `.backward()` implementation.
    Do not rely on Lightning on_after_backward for manual optimization.
    """
    def __init__(self, out_dir, probe_loader=None, snapshot_epochs=(0,1,4,9,24,49,99,199),
                 snapshot_every=50, cam_graph_samples=128):
        super().__init__()
        self.out_dir=Path(out_dir)
        self.probe_loader=probe_loader
        self.snapshot_epochs=set(int(i) for i in snapshot_epochs)
        self.snapshot_every=int(snapshot_every)
        self.cam_graph_samples=int(cam_graph_samples)
        self._grads=defaultdict(list)
        self._hooks=[]
        self._time=0.0
        self._batch_t0=None
        self._train_batch_seconds=0.0
        self._train_batches=0
        self._validation_t0=None
        self._validation_seconds=0.0
        self._latest_epoch_stats={}
        self._kernel_stats=defaultdict(lambda: {'calls':0,'requested_pairs':0,
                                                 'evaluated_pairs':0,'seconds':0.0})
        self._restore_layers=None
        self._restore_cam=[]
        self._in_probe=False

    def _record_kernel_call(self, stage, requested, evaluated, duration):
        d=self._kernel_stats[stage]
        d['calls']+=1
        d['requested_pairs']+=int(requested)
        d['evaluated_pairs']+=int(evaluated)
        d['seconds']+=float(duration)

    def _install_kernel_counters(self, pl_module):
        # The original GKNN computes additional GraKeL similarities inside
        # custom backward(). Intercept the exact eval_kernel function so these
        # are counted, not silently omitted from runtime comparison.
        try:
            import layers
            old=layers.GKernelConv.eval_kernel
            def instrumented(*args,**kwargs):
                start=time.perf_counter()
                out=old(*args,**kwargs)
                stage='probe' if self._in_probe else ('train' if pl_module.training else 'validation')
                n=int(out.numel())
                self._record_kernel_call(stage,n,n,time.perf_counter()-start)
                return out
            layers.GKernelConv.eval_kernel=staticmethod(instrumented)
            self._restore_layers=(layers,old)
        except (ImportError,AttributeError):
            self._restore_layers=None
        # CAM passes an entire batch of sampled traces to _evaluate and
        # removes duplicate sampled graph keys before GraKeL transform.
        for j,layer in enumerate(pl_module.conv_layers):
            if not hasattr(layer,'_evaluate'):
                continue
            old_bound=layer._evaluate
            def wrapped(mask_traces, graphs, prev=old_bound, module=layer):
                start=time.perf_counter()
                out=prev(mask_traces,graphs)
                stage='probe' if self._in_probe else ('train' if pl_module.training else 'validation')
                n=int(np.asarray(out).size)
                k=len(module.kernel_names)
                unique=len({t.graph_key() for row in mask_traces for t in row})
                computed=unique*len(graphs)*k
                self._record_kernel_call(stage,n,computed,time.perf_counter()-start)
                return out
            layer._evaluate=wrapped
            self._restore_cam.append((layer,old_bound))

    def on_fit_start(self, trainer, pl_module):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._install_kernel_counters(pl_module)
        for name,param in pl_module.named_parameters():
            if not param.requires_grad:
                continue
            def f(grad, n=name):
                if grad is not None:
                    self._grads[n].append(float(grad.detach().norm().cpu()))
                return grad
            self._hooks.append(param.register_hook(f))

    def on_train_epoch_start(self, trainer, pl_module):
        self._time=time.perf_counter()
        self._batch_t0=None
        self._train_batch_seconds=0.0
        self._train_batches=0
        self._validation_seconds=0.0
        self._grads.clear()
        self._kernel_stats.clear()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        # Starts AFTER dataloader collation; covers forward, backward and step.
        self._batch_t0=time.perf_counter()

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch_t0 is not None:
            self._train_batch_seconds+=time.perf_counter()-self._batch_t0
            self._train_batches+=1
            self._batch_t0=None

    def on_validation_epoch_start(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._validation_t0=time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module):
        # Preserve the train metrics when Lightning finishes aggregating them,
        # even on versions that aggregate *after* validation callbacks.
        metrics={}
        for name in ('acc','loss','loss_reg','loss_ce','frac_neg_resp',
                     'frac_const_filters','corr_min_eig'):
            value=trainer.callback_metrics.get(name)
            if torch.is_tensor(value) and value.numel()==1:
                metrics[name]=float(value.detach().cpu())
        write_json(self.out_dir/'train_epoch_end'/f'epoch_{trainer.current_epoch:04d}.json',
                   dict(epoch=int(trainer.current_epoch),metrics=metrics))

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self._validation_t0 is not None:
            self._validation_seconds=time.perf_counter()-self._validation_t0
            self._validation_t0=None
        epoch=int(trainer.current_epoch)
        gradrow={name:{'mean_norm':float(np.mean(v)), 'max_norm':float(np.max(v)),
                       'nonzero_batches':int(sum(x > 0 for x in v)),'batches':len(v)}
                 for name,v in self._grads.items() if v}
        self._latest_epoch_stats=dict(epoch=epoch,wall_epoch_s_including_validation=time.perf_counter()-self._time,
                                       gradients=gradrow)
        row = dict(epoch=epoch,
                   wall_epoch_s_including_validation=self._latest_epoch_stats['wall_epoch_s_including_validation'],
                   # Closest to paper's 'training time per epoch': includes loader
                   # and framework overhead, excludes validation (and subsequent
                   # snapshot/probe exports). Use with consistent instrumentation.
                   train_epoch_wall_excluding_validation_s=max(0.0,
                       self._latest_epoch_stats['wall_epoch_s_including_validation']
                       - self._validation_seconds),
                   train_epoch_batch_s=float(self._train_batch_seconds),
                   train_batches=int(self._train_batches),
                   mean_train_batch_s=float(self._train_batch_seconds/max(self._train_batches,1)),
                   validation_epoch_s=float(self._validation_seconds))
        for key in ('acc','loss','loss_reg','loss_ce','val_loss','val_acc',
                    'frac_neg_resp','frac_const_filters','corr_min_eig'):
            val=trainer.callback_metrics.get(key)
            if torch.is_tensor(val) and val.numel()==1:
                row[key]=float(val.detach().cpu())
            elif isinstance(val,(float,int)):
                row[key]=float(val)
            else:
                row[key]=''
        for name in ('train','validation'):
            d=self._kernel_stats.get(name, {})
            row[f'{name}_kernel_calls']=int(d.get('calls',0))
            row[f'{name}_kernel_requested_pairs']=int(d.get('requested_pairs',0))
            row[f'{name}_kernel_evaluated_pairs']=int(d.get('evaluated_pairs',0))
            row[f'{name}_kernel_seconds']=float(d.get('seconds',0))
        append_csv(self.out_dir/'epochs.csv',row)
        write_json(self.out_dir/'gradient_epochs'/f'epoch_{epoch:04d}.json',
                   self._latest_epoch_stats)
        for j,layer in enumerate(pl_module.conv_layers):
            if hasattr(layer,'last_diagnostics') and layer.last_diagnostics:
                diagnostics={k:json.dumps(json_safe(v)) for k,v in layer.last_diagnostics.items()}
                write_json(self.out_dir/'cam_forward_diagnostics'/f'epoch_{epoch:04d}_layer_{j+1}.json',
                           dict(epoch=epoch,layer=j+1,diagnostics=diagnostics))
        if epoch in self.snapshot_epochs or (self.snapshot_every>0 and
                                             (epoch+1)%self.snapshot_every==0):
            save_mask_snapshot(pl_module,self.out_dir/'snapshots',epoch,
                               cam_graph_samples=self.cam_graph_samples,
                               sample_seed=1729+epoch)
            if self.probe_loader is not None:
                self._in_probe=True
                try:
                    export_predictions_and_responses(
                        pl_module,self.probe_loader,
                        self.out_dir/'probe'/f'probe_epoch_{epoch:04d}.npz',
                        split_name=f'probe_epoch_{epoch}')
                finally:
                    self._in_probe=False

    def on_fit_end(self, trainer, pl_module):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        if self._restore_layers is not None:
            module,old=self._restore_layers
            module.GKernelConv.eval_kernel=staticmethod(old)
            self._restore_layers=None
        for layer,bound in self._restore_cam:
            layer._evaluate=bound
        self._restore_cam.clear()
