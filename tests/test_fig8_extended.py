"""Extended Figure 8: all masks and motif-localization ground truth."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from fig8_extended import (
    localization_scores, nx_mask, nx_target, selected_graphs, analyze_one
)


def _fixture_run(tmp_path):
    run = tmp_path / 'ring_original_test'
    (run / 'best_masks').mkdir(parents=True)
    # Two independent graphs; both contain a ring motif despite different
    # graph-level labels (as happens with a two-motif AND classification task).
    n = 10
    mask0 = dict(filter=0, n_nodes=8, labels=[0]*8,
                 edges=[[i,(i+1)%8] for i in range(8)])
    mask1 = dict(filter=1, n_nodes=8, labels=[0]*8,
                 edges=[[i,i+1] for i in range(7)])
    (run / 'best_masks' / 'masks_0001.json').write_text(json.dumps(dict(
        layers=[dict(masks=[mask0,mask1])]
    )))
    target = {'present':True,'nodes':list(range(8))}
    (run / 'synthetic_ground_truth.json').write_text(json.dumps(dict(
        val={'2':{'ring':target},'5':{'ring':target}}
    )))
    edges = [(i,(i+1)%8) for i in range(8)] + [(7,8),(8,9)]
    rows=[]
    for graph_id in range(2):
        offset=graph_id*n
        rows.extend((offset+a, offset+b) for a,b in edges)
    matrix = np.asarray(rows,dtype=int).T
    feats = np.concatenate([
        np.column_stack([np.r_[np.full(8,2.), np.full(2,.1)],
                         np.r_[np.full(8,.1), np.full(2,2.)]]) for _ in range(2)
    ],axis=0)
    np.savez_compressed(run/'best_val.npz',
        original_graph_index=np.array([2,5]),
        labels=np.array([1,0]),
        node_graph=np.r_[np.zeros(n,dtype=int),np.ones(n,dtype=int)],
        edge_index=matrix,
        layer1_raw=feats)
    return run


def test_localization_includes_motif_present_when_graph_label_is_zero(tmp_path):
    run=_fixture_run(tmp_path)
    gt=json.loads((run/'synthetic_ground_truth.json').read_text())['val']
    with np.load(run/'best_val.npz',allow_pickle=False) as z:
        auc,count=localization_scores(z['layer1_raw'],z['node_graph'],
            z['original_graph_index'],z['edge_index'],gt,'ring')
    assert count==2  # not just the one graph whose AND-task label is 1
    assert np.allclose(auc,[1.0,0.0])


def test_mask_and_target_metadata(tmp_path):
    run=_fixture_run(tmp_path)
    masks,note=selected_graphs(run)
    assert len(masks)==2
    assert nx_mask(masks[0]).number_of_edges()==8
    assert nx_target('ring').number_of_nodes()==8
    assert 'GKNN' in note


def test_every_filter_has_figure_and_row(tmp_path,monkeypatch):
    import fig8_extended as e
    run=_fixture_run(tmp_path)
    # GraKeL is used for the real science output; deterministic stub here
    # tests export/layout without depending on optional GraKeL installation.
    monkeypatch.setattr(e,'structural_scores',lambda target,masks: (
        np.array([1.0,0.4]),[True,False]))
    rows,summary=e.analyze_one(run,split='val',layer=1,
                              out_dir=run/'fig8_extended')
    assert len(rows)==2
    assert summary[0]['n_filters']==2
    assert summary[0]['n_graphs_localization']==2
    assert summary[0]['n_exact_representative_matches']==1
    folder=run/'fig8_extended'
    for name in ['all_masks_ring','all_node_responses_ring','quantitative_ring']:
        assert (folder/f'{name}.pdf').is_file()
        assert (folder/f'{name}.png').is_file()
    assert (folder/'per_mask_metrics.csv').is_file()
