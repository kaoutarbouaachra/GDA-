"""Extended Figure 8 analysis for an annotated synthetic GKNN/CAM run.

Unlike the paper's selected strongest filter, this script displays EVERY
learned mask and EVERY corresponding response map on a common input graph.
It also reports structure-vs-target normalized WL similarity, exact
isomorphism of the representative decoded graphs, and per-filter motif
localization AUC across annotated graphs. No model retraining.

Requires a run_tracked.py synthetic run containing best_{split}.npz,
synthetic_ground_truth.json, and best_masks/masks_*.json.

From the repository root:
    python scripts/fig8_extended.py --run results/synthetic/ring_cam --split val
    python scripts/fig8_extended.py --root results/synthetic --split val

Requires GraKeL for WL scores. CAM's most frequent graph among finite Monte
Carlo samples is a REPRESENTATIVE, not the exact marginal mode.
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
import networkx as nx
from sklearn.metrics import roc_auc_score
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

from generate_motifs import motif_graph


def selected_graphs(run_dir, layer=1):
    root = Path(run_dir)
    snaps = sorted((root / 'best_masks').glob('masks_*.json'))
    if not snaps:
        raise FileNotFoundError(f'No best_masks/masks_*.json under {root}')
    data = json.loads(snaps[-1].read_text(encoding='utf-8'))
    if not 1 <= layer <= len(data.get('layers', [])):
        raise ValueError(f'Layer {layer} unavailable')
    spec = data['layers'][layer - 1]
    if 'masks' in spec:
        return spec['masks'], 'GKNN hard graph after largest-component extraction'
    if 'graph_samples' in spec:
        graphs = []
        for result in spec['graph_samples']:
            if not result['top_graphs']:
                raise ValueError('No sampled graphs for a CAM filter')
            g = dict(result['top_graphs'][0])
            g['filter'] = int(result['filter'])
            g['sample_count'] = int(result['sample_count'])
            graphs.append(g)
        return graphs, 'CAM most frequent sampled graph (finite sample, not exact mode)'
    raise ValueError('Unknown mask family in snapshot')


def nx_mask(mask):
    g = nx.Graph()
    n = int(mask['n_nodes'])
    labs = [int(v) for v in mask['labels']]
    if n != len(labs):
        raise ValueError('Number of mask labels differs from node count')
    g.add_nodes_from((i, {'label': labs[i]}) for i in range(n))
    g.add_edges_from((int(a), int(b)) for a, b in mask['edges'])
    return g


def nx_target(name):
    g = motif_graph(name)
    nx.set_node_attributes(g, 0, 'label')
    return g


def _grakel_representation(g):
    edges = {(int(a), int(b)) for a, b in g.edges()}
    if not edges:
        edges = {(int(v), int(v)) for v in g.nodes()}
    return [edges, {int(v): int(g.nodes[v].get('label', 0)) for v in g.nodes()}]


def structural_scores(target_graph, masks):
    """Return WL similarity and exact labeled-graph isomorphism for each mask.

    Structural scores refer ONLY to representative graphs for CAM. They are
    not measurements of the whole learned probabilistic mask distribution.
    """
    from grakel.kernels import WeisfeilerLehman
    graphs = [nx_mask(mask) for mask in masks]
    converted = [_grakel_representation(target_graph)] + [
        _grakel_representation(g) for g in graphs]
    kmat = np.asarray(WeisfeilerLehman(n_iter=3, normalize=True)
                      .fit_transform(converted), dtype=float)
    wl = np.clip(kmat[0, 1:], 0.0, 1.0)
    exact = [bool(nx.is_isomorphic(target_graph, g,
                      node_match=lambda a, b: a.get('label') == b.get('label')))
             for g in graphs]
    return wl, exact


def _local_graph(nodes, edge_index):
    """Graph with nodes 0..n-1, reindexed from the full split node IDs."""
    pairs = {int(global_id): pos for pos, global_id in enumerate(nodes)}
    g = nx.Graph()
    g.add_nodes_from(range(len(nodes)))
    for u, v in edge_index.T:
        u = int(u)
        v = int(v)
        if u in pairs and v in pairs:
            g.add_edge(pairs[u], pairs[v])
    return g


def localization_scores(features, node_graph, indices, all_edges, gt_for_split,
                        motif, dilation=0):
    """Compute AUC over graphs where THIS motif exists (including AND-task negatives).

    Returns (mean_AUC_for_each_filter, number_of_graphs). This makes the
    two-motif AND task meaningful: a graph can contain A but have label 0.
    """
    m = features.shape[1]
    all_auc = []
    for split_id, original_id in enumerate(indices):
        info = gt_for_split.get(str(int(original_id)), {}).get(motif)
        if not info or not info.get('present'):
            continue
        nodes = np.flatnonzero(node_graph == split_id)
        if not len(nodes):
            continue
        graph = _local_graph(nodes, all_edges)
        target = set(int(v) for v in info['nodes'])
        if dilation > 0:
            expanded = set(target)
            for node in target:
                if node not in graph:
                    raise ValueError('Invalid annotated motif node')
                expanded.update(nx.single_source_shortest_path_length(
                    graph, node, cutoff=dilation).keys())
            target = expanded
        if any(v < 0 or v >= len(nodes) for v in target):
            raise ValueError('Motif metadata references invalid local vertices')
        y = np.asarray([int(i in target) for i in range(len(nodes))])
        if y.min() == y.max():
            continue
        one = features[nodes]
        all_auc.append([roc_auc_score(y, one[:, fi]) for fi in range(m)])
    if not all_auc:
        return np.full(m, np.nan), 0
    return np.asarray(all_auc, dtype=float).mean(axis=0), len(all_auc)


def _save_figure(fig, directory, name):
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(directory / f'{name}.png', bbox_inches='tight', dpi=180)
    plt.close(fig)


def plot_all_masks(masks, target, scores, exact, out, title, note):
    count = len(masks)
    # One panel for the target, plus EVERY learned mask (even when m > 16).
    columns = 4
    rows = int(np.ceil((count + 1) / columns))
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(3.2 * columns, 3.05 * rows))
    for ax in axes.flat:
        ax.axis('off')
    ax0 = axes.flat[0]
    tgraph = nx_target(target)
    tpos = (nx.circular_layout(tgraph) if target == 'ring'
            else nx.spring_layout(tgraph, seed=5))
    nx.draw_networkx(tgraph, pos=tpos, ax=ax0, with_labels=False,
                     node_size=280, width=1.6, node_color='#e4ac58')
    ax0.set_title(f'TARGET: {target} ({tgraph.number_of_nodes()} nodes)',
                  fontsize=10, fontweight='bold')
    for j, mask in enumerate(masks):
        ax = axes.flat[j + 1]
        graph = nx_mask(mask)
        pos = nx.spring_layout(graph, seed=5)
        nx.draw_networkx(graph, pos=pos, ax=ax, node_size=280,
                         labels={v: str(graph.nodes[v]['label']) for v in graph},
                         font_size=8, width=1.2)
        label = f'Filter {j+1} | WL={scores[j]:.2f}'
        if exact[j]:
            label += ' | exact'
        if 'frequency' in mask:
            label += f" | freq={mask['frequency']:.2f}"
        ax.set_title(label, fontsize=9)
    fig.suptitle(f'{title}\nALL {count} learned masks | target: {target}\n{note}',
                 fontsize=11, y=1.02)
    fig.tight_layout()
    _save_figure(fig, out, f'all_masks_{target}')


def plot_all_responses(masks, features, nodes, edges, graph_id, target_nodes,
                       out, title, motif):
    """Same topology, layout and response color scale for EVERY filter in a run."""
    scores = np.asarray(features[nodes], dtype=float)
    count = len(masks)
    graph = _local_graph(nodes, edges)
    positions = nx.spring_layout(graph, seed=9)
    all_nodes = list(graph.nodes())
    vx = float(np.min(scores))
    ux = float(np.max(scores))
    if not vx < ux:
        ux = vx + 1.0
    columns = min(4, max(1, count))
    rows = int(np.ceil(count / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.8*columns, 3.35*rows+0.65),
                             squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for f in range(count):
        ax = axes.flat[f]
        nx.draw_networkx_edges(graph, positions, ax=ax, alpha=0.45, width=1)
        scatter = nx.draw_networkx_nodes(graph, positions, ax=ax,
                     nodelist=all_nodes, node_size=95, cmap='viridis',
                     node_color=scores[:, f], vmin=vx, vmax=ux)
        if target_nodes:
            nx.draw_networkx_nodes(graph, positions, ax=ax,
                    nodelist=sorted(target_nodes), node_size=156,
                    node_color='none', edgecolors='#d14932', linewidths=1.6)
        ax.set_title(f'Filter {f+1}', fontsize=10)
    fig.suptitle(f'{title}\nAll filters on graph {graph_id} ({motif}); '
                 'red outlines mark motif vertices', fontsize=10, y=0.985)
    fig.subplots_adjust(top=0.81, bottom=0.21, right=0.96,
                        left=0.04, wspace=0.08, hspace=0.25)
    bar_ax = fig.add_axes([0.25, 0.08, 0.50, 0.023])
    fig.colorbar(ScalarMappable(norm=Normalize(vmin=vx, vmax=ux), cmap='viridis'),
                 cax=bar_ax, orientation='horizontal',
                 label='Raw kernel response (common scale across all filters)')
    # color scale is shared within run; different runs may have different scales
    _save_figure(fig, out, f'all_node_responses_{motif}')


def plot_quantitative(rows, motif, directory, title):
    subset = [row for row in rows if row['motif'] == motif]
    x = np.arange(1, len(subset) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    wl = np.asarray([r['wl_similarity'] for r in subset], dtype=float)
    auc = np.asarray([r['localization_auc'] for r in subset], dtype=float)
    axes[0].bar(x, wl)
    exact_idx = [i for i, r in enumerate(subset) if r['representative_exact_match']]
    if exact_idx:
        axes[0].scatter(x[exact_idx], wl[exact_idx], marker='*', s=120,
                        label='Exact labeled isomorphism')
        axes[0].legend(fontsize=8)
    axes[0].set(title='Mask vs target (WL kernel)', xlabel='Filter index',
                ylabel='Normalized WL similarity', ylim=(0, 1.03))
    axes[1].bar(x, auc)
    axes[1].axhline(0.5, color='gray', linestyle='--', label='Random ranking')
    axes[1].set(title='Node localization of inserted motif', xlabel='Filter index',
                ylabel='ROC-AUC across annotated graphs', ylim=(0, 1.03))
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.set_xticks(x)
        ax.grid(alpha=0.2, axis='y')
    fig.suptitle(f'{title}: {motif}; structural recovery and localization '
                 'are DIFFERENT quantities', fontsize=11)
    fig.tight_layout()
    _save_figure(fig, directory, f'quantitative_{motif}')


def write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row.keys()))
    with path.open('w', newline='', encoding='utf-8') as fp:
        w = csv.DictWriter(fp, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def analyze_one(run, split='val', layer=1, motif=None, graph_id=None, dilation=0,
                out_dir=None, make_figures=True):
    run = Path(run)
    info = json.loads((run / 'synthetic_ground_truth.json').read_text(encoding='utf-8'))
    gt = info.get(split)
    if gt is None:
        raise ValueError(f'Synthetic split {split} unavailable')
    with np.load(run / f'best_{split}.npz', allow_pickle=False) as data:
        original_indices = data['original_graph_index'].astype(int)
        node_graph = data['node_graph'].astype(int)
        edges = data['edge_index'].astype(int)
        features = data[f'layer{layer}_raw'].astype(float)
        graph_labels = data['labels'].astype(int)
    masks, note = selected_graphs(run, layer=layer)
    if features.shape[1] != len(masks):
        raise ValueError(f'{features.shape[1]} response channels but {len(masks)} masks. '
                         'Use a single graph kernel for this experiment.')
    names = sorted({name for ann in gt.values() for name in ann.keys()})
    if motif is not None:
        if motif not in names:
            raise ValueError(f'Unknown motif {motif}; choose from {names}')
        names = [motif]
    title = run.name
    dest = Path(out_dir) if out_dir else run / 'fig8_extended' / split
    rows = []
    summary = []
    for name in names:
        target = nx_target(name)
        wl, exact = structural_scores(target, masks)
        aucs, count = localization_scores(features, node_graph,
                                original_indices, edges, gt, name, dilation=dilation)
        if graph_id is None:
            eligible = [int(original_id) for original_id in original_indices
                        if gt.get(str(int(original_id)), {}).get(name, {}).get('present')]
            positive = [orig for orig in eligible
                        if graph_labels[np.flatnonzero(original_indices == orig)[0]] == 1]
            pick = (positive or eligible)[0] if eligible else None
        else:
            pick = int(graph_id)
        for f, mask in enumerate(masks):
            rows.append(dict(run=str(run), split=split, layer=layer,
                       motif=name, filter=f+1,
                       wl_similarity=float(wl[f]),
                       representative_exact_match=int(exact[f]),
                       localization_auc=float(aucs[f]) if np.isfinite(aucs[f]) else None,
                       n_graphs_localization=count,
                       representative_frequency=mask.get('frequency', None),
                       cam_graph_samples=mask.get('sample_count', None)))
        vals = np.asarray(aucs, dtype=float)
        summary.append(dict(run=str(run), split=split, layer=layer, motif=name,
                     n_filters=len(masks), n_graphs_localization=count,
                     max_wl_similarity=float(np.max(wl)),
                     n_exact_representative_matches=int(sum(exact)),
                     max_localization_auc=float(np.nanmax(vals))
                                          if np.isfinite(vals).any() else None,
                     mean_localization_auc=float(np.nanmean(vals))
                                          if np.isfinite(vals).any() else None,
                     representative_note=note))
        if not make_figures:
            continue
        plot_all_masks(masks, name, wl, exact, dest, title, note)
        if pick is not None:
            matches = np.flatnonzero(original_indices == pick)
            if len(matches) != 1:
                raise ValueError('Requested original graph ID does not belong to split')
            sid = int(matches[0])
            nodes = np.flatnonzero(node_graph == sid)
            target_nodes = set(gt[str(pick)][name]['nodes'])
            plot_all_responses(masks, features, nodes, edges, pick, target_nodes,
                               dest, title, name)
        plot_quantitative(rows, name, dest, title)
    write_rows(dest / 'per_mask_metrics.csv', rows)
    write_rows(dest / 'per_motif_summary.csv', summary)
    (dest / 'README.txt').write_text(
        'All masks shown; no selection of a single strongest filter.\n'
        'Normalized WL similarity is NOT an isomorphism test.\n'
        'Exact means labeled isomorphism of the one representative hard graph.\n'
        'For CAM that graph is the most frequent of a finite number of samples.\n'
        'Localization AUC uses all graphs where the named motif is present, even\n'
        'when the overall AND-task class label is zero.\n'
        'Red outlines denote the true inserted motif vertices. The response\n'
        'color scale is shared only within an individual run.\n'
        'No number of "recovered masks" is claimed from an arbitrary WL cutoff.\n',
        encoding='utf-8')
    return rows, summary


def analyze_many(root, split='val', layer=1, motif=None, dilation=0, out=None):
    root = Path(root)
    runs = sorted({path.parent for path in root.rglob('synthetic_ground_truth.json')})
    if not runs:
        raise ValueError('No annotated synthetic runs found under ' + str(root))
    all_rows, all_summary = [], []
    for run in runs:
        if not (run / f'best_{split}.npz').exists():
            continue
        try:
            rows, summary = analyze_one(run, split=split, layer=layer,
                            motif=motif, dilation=dilation)
            all_rows.extend(rows)
            all_summary.extend(summary)
        except Exception as e:
            print(f'Error with {run}: {e}')
            raise
    out = Path(out) if out else root / 'fig8_comparison'
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / 'all_masks_all_runs.csv', all_rows)
    write_rows(out / 'all_runs_motif_summary.csv', all_summary)
    # Points show each independent training run; do NOT silently average
    # different motifs or label nominal plots as independent folds.
    if all_summary:
        configs = []
        for item in all_summary:
            try:
                config = json.loads((Path(item['run'])/'summary.json').read_text(encoding='utf-8'))
                label = f"{config.get('model','?')}/{config.get('reg_type','?')}"
            except FileNotFoundError:
                label = Path(item['run']).name
            configs.append(label)
        uniques = list(dict.fromkeys(configs))
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, key, desc in zip(axes,
                       ('max_wl_similarity', 'max_localization_auc'),
                       ('Best representative mask similarity', 'Best motif-localizing filter')):
            for i, label in enumerate(uniques):
                vals = [row[key] for row, lab in zip(all_summary, configs)
                        if lab == label and row[key] is not None]
                if vals:
                    ax.scatter(i+np.linspace(-0.1,0.1,len(vals)),vals,
                               label=label if ax is axes[0] else None,alpha=0.75)
                    ax.scatter(i,np.mean(vals),marker='D',s=65,color='black')
            ax.set_xticks(range(len(uniques)),uniques,rotation=30,ha='right')
            ax.set_ylim(0,1.05)
            ax.set_ylabel(key)
            ax.set_title(desc)
            ax.grid(alpha=0.2,axis='y')
        fig.suptitle(f'Synthetic Fig. 8 extension ({split}): points are per run and motif')
        fig.tight_layout()
        _save_figure(fig,out,'across_runs_structural_vs_localization')
    print(f'Processed {len(runs)} synthetic runs; results saved to {out}')
    return all_rows,all_summary


def main():
    ap = argparse.ArgumentParser(description='Extended Figure 8 - all masks and all responses')
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument('--run',type=Path,help='A run_tracked.py synthetic result directory')
    group.add_argument('--root',type=Path,help='Process every annotated synthetic run under a root')
    ap.add_argument('--split',choices=('train','val','test'),default='val')
    ap.add_argument('--layer',type=int,default=1)
    ap.add_argument('--motif',choices=('ring','wheel','grid','ladder','cliques'),default=None)
    ap.add_argument('--graph_id',type=int,default=None,
                    help='Original graph index of an annotated graph (single-run mode)')
    ap.add_argument('--dilation',type=int,default=0,
                    help='Grow target motif nodes by graph distance before localization AUC')
    ap.add_argument('--out',type=Path,default=None)
    args=ap.parse_args()
    if args.dilation<0:
        ap.error('--dilation must be >=0')
    if args.run:
        analyze_one(args.run,args.split,args.layer,args.motif,args.graph_id,
                    args.dilation,args.out)
        print('Saved full Figure 8 extension for:',args.run)
    else:
        if args.graph_id is not None:
            ap.error('--graph_id requires --run')
        analyze_many(args.root,args.split,args.layer,args.motif,args.dilation,args.out)


if __name__=='__main__':
    main()
