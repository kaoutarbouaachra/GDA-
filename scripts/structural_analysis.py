"""Normalized Weisfeiler-Lehman similarities of saved decoded GKNN/CAM masks.

Usage:
  python scripts/structural_analysis.py --run results/.../MUTAG_original... 
  python scripts/structural_analysis.py --run results/.../seed0 --other_run results/.../seed1

For CAM, uses each filter's most frequently sampled graph (finite MC), not an
exact mode of the marginal graph law. WL=1 does not prove isomorphism.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_graphs(run):
    path=Path(run)
    files=sorted((path/'best_masks').glob('masks_*.json'))
    if not files:
        raise FileNotFoundError('Missing best_masks/masks_*.json in '+str(path))
    layers=json.loads(files[-1].read_text(encoding='utf-8'))['layers']
    if not layers:
        raise ValueError('No learned mask layers')
    layer=layers[0]
    if 'masks' in layer:
        graphs=layer['masks']
    elif 'graph_samples' in layer:
        graphs=[f['top_graphs'][0] for f in layer['graph_samples']]
    else:
        raise ValueError('Unknown mask representation')
    return graphs


def grakel_graph(mask):
    p=int(mask['n_nodes'])
    labels={i:int(v) for i,v in enumerate(mask['labels'])}
    edges=set()
    for u,v in mask['edges']:
        edges.add((int(u),int(v)))
        edges.add((int(v),int(u)))
    if not edges:
        edges={(i,i) for i in range(p)}
    return [edges,labels]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--run',required=True)
    ap.add_argument('--other_run',default=None)
    ap.add_argument('--out',default=None)
    args=ap.parse_args()
    from grakel.kernels import WeisfeilerLehman
    g1=read_graphs(args.run)
    g2=read_graphs(args.other_run) if args.other_run else None
    merged=g1+g2 if g2 is not None else g1
    K=WeisfeilerLehman(n_iter=3,normalize=True).fit_transform([grakel_graph(g) for g in merged])
    n=len(g1)
    sim=K[:n,n:] if g2 is not None else K
    out=Path(args.out) if args.out else Path(args.run)/'structural_wl'
    out.mkdir(parents=True,exist_ok=True)
    np.save(out/'matrix.npy',sim)
    vals=sim.ravel() if g2 is not None else sim[np.triu_indices_from(sim,k=1)]
    report=dict(n_masks_A=n,n_masks_B=len(g2) if g2 else n,
                mean_wl=float(vals.mean()) if len(vals) else None,
                std_wl=float(vals.std()) if len(vals) else None,
                min_wl=float(vals.min()) if len(vals) else None,
                max_wl=float(vals.max()) if len(vals) else None,
                score_definition='normalized 3-iteration Weisfeiler-Lehman kernel',
                caution='WL=1 does not establish graph isomorphism; CAM modes estimated from finite samples')
    (out/'summary.json').write_text(json.dumps(report,indent=2))
    fig,ax=plt.subplots(figsize=(6,5))
    im=ax.imshow(sim,vmin=0,vmax=1,cmap='Blues')
    fig.colorbar(im,ax=ax,label='Normalized WL similarity')
    ax.set(xlabel='Filters in run B' if g2 else 'Mask index',
           ylabel='Filters in run A' if g2 else 'Mask index',
           title='Cross-initialization WL' if g2 else 'Within-run mask WL')
    fig.tight_layout()
    fig.savefig(out/'similarity.png',dpi=200)
    fig.savefig(out/'similarity.pdf')
    plt.close(fig)
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
