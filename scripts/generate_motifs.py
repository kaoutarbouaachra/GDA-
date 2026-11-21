"""Generate balanced controlled synthetic motif classification tasks.

This is a NEW controlled protocol inspired by, NOT an exact reproduction of,
Cosmo et al. Figure 8. Saves exact motif vertex memberships for localization.
Input file is JSONL; each record has n_nodes, edges, label and motif metadata.

Usage:
  python scripts/generate_motifs.py --motif ring --n_graphs 2000 --seed 7 --output synthetic/ring.jsonl
  python scripts/generate_motifs.py --motif ring+wheel --n_graphs 2000 --seed 7 --output synthetic/ring_wheel.jsonl
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import networkx as nx


def motif_graph(name):
    if name=='ring':
        g=nx.cycle_graph(8)
    elif name=='wheel':
        g=nx.wheel_graph(8)
    elif name=='grid':
        g=nx.convert_node_labels_to_integers(nx.grid_2d_graph(3,3))
    elif name=='ladder':
        g=nx.ladder_graph(5)
    elif name=='cliques':
        g=nx.disjoint_union(nx.complete_graph(4),nx.complete_graph(4))
        g.add_edge(3,4)
    else:
        raise ValueError('Choose motif from ring, wheel, grid, ladder, cliques')
    return nx.convert_node_labels_to_integers(g)


def graph_with_same_nodes_edges_but_different(target,rng):
    n,m=target.number_of_nodes(),target.number_of_edges()
    for _ in range(2000):
        graph=nx.gnm_random_graph(n,m,seed=rng.randrange(2**30))
        if nx.is_connected(graph) and not nx.is_isomorphic(graph,target):
            return graph
    raise RuntimeError('Could not generate a matched random connected replacement')


def generate_one(names, label, rng, p_base, p_attach, base_min, base_max):
    base_n=rng.randint(base_min,base_max)
    g=nx.erdos_renyi_graph(base_n,p_base,seed=rng.randrange(2**30))
    motifs={}
    # If one motif: label 1 iff target is inserted; if two: AND of both.
    bits=([int(label)] if len(names)==1 else
          [1,1] if label else rng.choice([(0,0),(0,1),(1,0)]))
    for name,present in zip(names,bits):
        target=motif_graph(name)
        block=target if present else graph_with_same_nodes_edges_but_different(target,rng)
        off=g.number_of_nodes()
        g.add_nodes_from(range(off,off+block.number_of_nodes()))
        g.add_edges_from((off+u,off+v) for u,v in block.edges())
        nodes=list(range(off,off+block.number_of_nodes()))
        # Reuse the same attachment scheme for positive/negative structure.
        attach=[(u,v) for u in range(base_n) for v in nodes if rng.random()<p_attach]
        if not attach:
            attach=[(rng.randrange(base_n),rng.choice(nodes))]
        g.add_edges_from(attach)
        motifs[name]=dict(present=bool(present),nodes=nodes if present else [],
                          block_nodes=nodes,template_edges=target.number_of_edges(),
                          template_nodes=target.number_of_nodes())
    return dict(n_nodes=g.number_of_nodes(),edges=[list(e) for e in sorted(g.edges())],
                label=label,n_base_nodes=base_n,motifs=motifs,
                n_edges=g.number_of_edges())


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--motif',default='ring')
    p.add_argument('--n_graphs',type=int,default=2000)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--base_min',type=int,default=30)
    p.add_argument('--base_max',type=int,default=50)
    p.add_argument('--p_base',type=float,default=0.10)
    p.add_argument('--p_attach',type=float,default=0.02)
    p.add_argument('--output',required=True)
    opts=p.parse_args()
    names=opts.motif.split('+')
    if len(names)>2:
        raise SystemExit('Up to two motifs are supported')
    for name in names:
        motif_graph(name)
    if opts.n_graphs<4 or opts.n_graphs%2:
        raise SystemExit('n_graphs must be even and >= 4')
    rng=random.Random(opts.seed)
    records=[]
    for i in range(opts.n_graphs):
        y=int(i%2==0)
        d=generate_one(names,y,rng,opts.p_base,opts.p_attach,opts.base_min,opts.base_max)
        d['graph_id']=i
        records.append(d)
    rng.shuffle(records)
    out=Path(opts.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',encoding='utf-8') as f:
        for record in records:
            f.write(json.dumps(record,sort_keys=True)+'\n')
    print(f'Saved {len(records)} labelled graphs in {out}. This is a NEW synthetic protocol.')


if __name__=='__main__':
    main()
