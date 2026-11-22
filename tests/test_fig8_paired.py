"""Paired Figure-8 data: pairs share size/edges, folds are grouped, files load."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))


def _generate(tmp_path, *extra):
    cmd = [sys.executable, str(ROOT / 'scripts' / 'generate_fig8_paired.py'), '--n_graphs', '40',
           '--n_folds', '10', '--out_dir', str(tmp_path / 'syn'), '--fold_dir', str(tmp_path / 'folds'),
           *extra]
    return subprocess.run(cmd, capture_output=True, text=True)


def test_pairs_and_grouped_folds(tmp_path):
    r = _generate(tmp_path)
    assert r.returncode == 0, r.stderr
    mapping = json.loads((tmp_path / 'syn' / 'fig8_datasets.json').read_text())
    assert sorted(mapping) == ['FIG8_CLIQUES', 'FIG8_GRID', 'FIG8_LADDER', 'FIG8_RING', 'FIG8_WHEEL']
    radius = {'ring': 4, 'wheel': 1, 'grid': 2, 'ladder': 3, 'cliques': 2}
    for ds, path in mapping.items():
        rows = [json.loads(x) for x in Path(path).read_text().splitlines()]
        motif = ds.split('_')[1].lower()
        assert rows[0]['motifs'][motif]['radius'] == radius[motif]
        pairs = {}
        for i, r_ in enumerate(rows):
            pairs.setdefault(r_['base_pair_id'], []).append(i)
        for a, b in pairs.values():
            ra, rb = rows[a], rows[b]
            assert {ra['label'], rb['label']} == {0, 1}
            assert (ra['n_nodes'], ra['n_edges']) == (rb['n_nodes'], rb['n_edges'])
        folds = json.loads((tmp_path / 'folds' / f'{ds}_folds_10.txt').read_text())
        assert len(folds) == 10
        pair_of = {i: r_['base_pair_id'] for i, r_ in enumerate(rows)}
        for tr, va, te in folds:
            sides = [{pair_of[i] for i in part} for part in (tr, va, te)]
            assert not (sides[0] & sides[1] or sides[0] & sides[2] or sides[1] & sides[2])
            assert sorted(tr + va + te) == list(range(len(rows)))


def test_refuses_to_overwrite(tmp_path):
    assert _generate(tmp_path).returncode == 0
    second = _generate(tmp_path)
    assert second.returncode != 0 and 'refusing to overwrite' in second.stderr


def test_loads_in_training_pipeline(tmp_path):
    pytest.importorskip('torch_geometric')
    assert _generate(tmp_path, '--motifs', 'ring').returncode == 0
    from synthetic_data import SyntheticGraphsDataset
    ds = SyntheticGraphsDataset(tmp_path / 'syn' / 'fig8_ring.jsonl', hops=4)
    assert len(ds) == 40 and ds.num_classes == 2
    from generate_motifs import required_hops
    assert required_hops(sorted(ds.records[0]['motifs'])) == 4
