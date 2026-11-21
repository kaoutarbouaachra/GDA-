"""Ground-truth node localization on synthetic motif graphs, no retraining.

Requires one run produced by run_tracked.py --synthetic_path ...
Produces the per-mask localization ROC-AUC matrix, optionally dilated by r.
This does NOT measure exact motif recovery by graph edit distance.

Usage:
  python scripts/motif_analysis.py --run results/synthetic/SYNTH_RING/cam/... --split val
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze(run_dir,split,layer,dilation=0):
    from sklearn.metrics import roc_auc_score
    import networkx as nx
    run_dir=Path(run_dir)
    with np.load(run_dir/f'best_{split}.npz',allow_pickle=False) as z:
        labels=z['labels'].astype(int)
        graph_indices=z['original_graph_index'].astype(int)
        mapping=z['node_graph'].astype(int)
        all_edges=z['edge_index'].astype(int)
        features=z[f'layer{layer}_raw']
    meta=json.loads((run_dir/'synthetic_ground_truth.json').read_text(encoding='utf-8'))[split]
    motif_names=sorted({name for graph in meta.values() for name in graph})
    if not motif_names:
        raise ValueError('No motif annotations')
    results={k:[] for k in motif_names}
    for gidx,original_idx in enumerate(graph_indices):
        # IMPORTANT: in a two-motif AND task, a graph can contain this motif
        # while its graph-level class is 0. Analyze every graph with the motif.
        nodes=np.flatnonzero(mapping==gidx)
        if not len(nodes):
            continue
        graph_meta=meta[str(int(original_idx))]
        present_names=[name for name in motif_names if graph_meta[name]['present']]
        g=nx.Graph()
        g.add_nodes_from(nodes.tolist())
        sub_edges=all_edges[:,(np.isin(all_edges[0],nodes)&np.isin(all_edges[1],nodes))]
        g.add_edges_from(sub_edges.T.tolist())
        for name in present_names:
            target=set(int(nodes[int(v)]) for v in graph_meta[name]['nodes'])
            if dilation:
                expanded=set(target)
                for v in list(target):
                    expanded.update(nx.single_source_shortest_path_length(g,v,cutoff=dilation))
                target=expanded
            y=np.array([int(int(v) in target) for v in nodes])
            if min(y.sum(),len(y)-y.sum())==0:
                continue
            aucs=np.array([roc_auc_score(y,features[nodes,j]) for j in range(features.shape[1])])
            results[name].append(aucs)
    rows=[]
    for name,vals in results.items():
        if not vals:
            continue
        arr=np.stack(vals)
        for i in range(arr.shape[1]):
            rows.append(dict(motif=name,filter=i+1,n_graphs_with_motif=len(arr),
                             mean_node_localization_auc=float(arr[:,i].mean()),
                             sd_node_localization_auc=float(arr[:,i].std(ddof=1))
                                  if len(arr)>1 else None))
    out=run_dir/'motif_analysis'
    out.mkdir(exist_ok=True)
    if rows:
        with (out/f'localization_{split}_layer{layer}_d{dilation}.csv').open('w',newline='',encoding='utf-8') as f:
            wr=csv.DictWriter(f,fieldnames=list(rows[0]))
            wr.writeheader();wr.writerows(rows)
        heat=np.full((len(motif_names),features.shape[1]),np.nan)
        for j,name in enumerate(motif_names):
            for row in rows:
                if row['motif']==name:
                    heat[j,int(row['filter'])-1]=row['mean_node_localization_auc']
        fig,ax=plt.subplots(figsize=(max(7,features.shape[1]*0.5),2+len(motif_names)*0.6))
        im=ax.imshow(np.ma.masked_invalid(heat),vmin=0,vmax=1,cmap='viridis',aspect='auto')
        ax.set_xticks(range(features.shape[1]),[str(i+1) for i in range(features.shape[1])])
        ax.set_yticks(range(len(motif_names)),motif_names)
        ax.set(xlabel='Filter index',ylabel='Known inserted motif',
               title=f'Node localization AUC, {split} (dilation r={dilation})')
        fig.colorbar(im,ax=ax,label='Mean ROC-AUC on positive graphs')
        fig.tight_layout()
        fig.savefig(out/f'localization_{split}_layer{layer}_d{dilation}.pdf')
        fig.savefig(out/f'localization_{split}_layer{layer}_d{dilation}.png',dpi=200)
        plt.close(fig)
    print('Annotated motifs:',motif_names,' graphs used:',{k:len(v) for k,v in results.items()})
    return rows


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--split',choices=['train','val','test'],default='val')
    ap.add_argument('--layer',type=int,default=1)
    ap.add_argument('--dilation',type=int,default=0)
    args=ap.parse_args()
    analyze(args.run,args.split,args.layer,args.dilation)


if __name__=='__main__':
    main()
