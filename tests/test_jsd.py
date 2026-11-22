"""JSD regularization: official formula, sign, gradient flow, validation-only selection."""
import json
import sys
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'cluster'))


def official_jsd_term(R, weight):
    """Verbatim from lcosmo/GKNN src/model.py (Model.__init__._regularizers)."""
    H = lambda x: -torch.sum(x * x.log(), -1)
    JSD = lambda x: H(x.mean(0)) - H(x).mean(0)
    return -(weight * JSD(R.softmax(-1)))


def test_jsd_code_is_the_official_implementation():
    pytest.importorskip('torch_geometric')
    from model import jsd_code
    torch.manual_seed(0)
    for scale in (0.1, 1.0, 10.0):
        R = torch.rand(57, 8) * scale
        assert torch.allclose(-1000 * jsd_code(R), official_jsd_term(R, 1000), atol=1e-6)


def test_sign_rewards_diverse_filters():
    pytest.importorskip('torch_geometric')
    from model import jsd_code
    m = 8
    redundant = torch.ones(40, m) * 3.0                    # every filter identical
    diverse = torch.zeros(40, m)
    diverse[torch.arange(40), torch.arange(40) % m] = 3.0  # each node claimed by one filter
    assert jsd_code(redundant).abs() < 1e-6
    assert jsd_code(diverse) > 0.5
    reg = lambda R: -1000 * jsd_code(R)                    # term added to the loss
    assert reg(diverse) < reg(redundant)                   # minimizing loss -> diversity
    # One gradient step on the loss increases JSD.
    R = (torch.rand(40, m) * 0.1).requires_grad_()
    before = jsd_code(R).item()
    reg(R).backward()
    with torch.no_grad():
        R -= 1e-3 * R.grad
    assert jsd_code(R).item() > before


def _tiny_batch(tmp_path):
    rng = np.random.default_rng(0)
    path = tmp_path / 'tiny.jsonl'
    with path.open('w') as f:
        for i in range(8):
            n = 12
            edges = sorted({tuple(sorted(map(int, rng.choice(n, 2, replace=False)))) for _ in range(20)})
            edges += [(j, j + 1) for j in range(n - 1) if (j, j + 1) not in edges]
            f.write(json.dumps(dict(n_nodes=n, edges=[list(e) for e in edges], label=i % 2,
                                    motifs={})) + '\n')
    from synthetic_data import SyntheticGraphsDataset
    from dataset import EgonetLoader
    ds = SyntheticGraphsDataset(path, hops=1)
    return next(iter(EgonetLoader([ds[i] for i in range(8)], batch_size=8, shuffle=False)))


def _model(kind, layers=1):
    from model import Model
    return Model(Namespace(
        in_features=1, labels=1, hidden=4, num_classes=2, nodes=4, layers=layers, hops=1,
        kernel='wl', pooling='add', activation='relu', mlp_layers=2, max_cc=True,
        optimize_masks=True, mask_distribution=kind, reg_type='jsd_code', jsd_weight=1000.0,
        lr=0.005, lr_graph=0.01, loss=torch.nn.CrossEntropyLoss(), cam_samples=4,
        cam_eval_samples=4, cam_initial_stop_prob=0.1, cam_eval_seed=1729,
        cam_input_temperature=1.0, cam_multilayer_inputs='gknn'))


MASK_PARAMS = {'original': ['Padd'], 'cam': ['stop_logits', 'parent_logits', 'optional_logits']}


@pytest.mark.parametrize('kind', ['original', 'cam'])
def test_multilayer_same_readout_and_inter_layer_jsd_gradient(tmp_path, kind):
    """L=2: readout concatenates both layers; layer-2 JSD alone reaches layer-1 masks."""
    pytest.importorskip('grakel')
    torch.manual_seed(1)
    np.random.seed(1)
    batch = _tiny_batch(tmp_path)
    model = _model(kind, layers=2)
    model.train()
    assert model.fc[0].in_features == 4 * 2  # hidden x layers: all layers concatenated
    if kind == 'cam':  # deep CAM layer reads argmax labels, like GKNN
        deep = model.conv_layers[1]
        assert deep.input_label_mode == 'argmax' and deep.input_gradient == 'gknn'
    first = model.conv_layers[0]
    seen = {p: 0.0 for p in MASK_PARAMS[kind]}
    for _ in range(6):
        model.zero_grad()
        logits, responses, _ = model(batch)
        assert len(responses) == 2 and logits.shape == (8, 2)
        model.regularizers(responses[1], batch.batch).backward()  # layer-2 JSD ONLY
        for p in seen:
            g = getattr(first, p).grad
            seen[p] = max(seen[p], 0.0 if g is None else float(g.abs().sum()))
    assert all(v > 0 for v in seen.values()), seen


@pytest.mark.parametrize('kind,params', [
    ('original', ['Padd']),                                        # GKNN edit-probability logits
    ('cam', ['stop_logits', 'parent_logits', 'optional_logits']),  # CAM distribution logits
])
def test_jsd_alone_updates_mask_parameters(tmp_path, kind, params):
    """Backpropagate ONLY the JSD term; the mask-generation parameters must move."""
    pytest.importorskip('grakel')
    torch.manual_seed(0)
    np.random.seed(0)
    batch = _tiny_batch(tmp_path)
    model = _model(kind)
    model.train()
    layer = model.conv_layers[0]
    seen = {p: 0.0 for p in params}
    for _ in range(5):  # GKNN's gradient is a random-edit estimator: allow a few draws
        model.zero_grad()
        _, responses, _ = model(batch)
        loss_reg = torch.stack([model.regularizers(x, batch.batch) for x in responses]).mean()
        assert loss_reg.requires_grad and loss_reg.item() <= 0
        loss_reg.backward()
        for p in params:
            g = getattr(layer, p).grad
            seen[p] = max(seen[p], 0.0 if g is None else float(g.abs().sum()))
    assert all(v > 0 for v in seen.values()), seen


def _fake_run(root, lam, val_ce, val_acc, done=True, test=False):
    d = root / f'run_w{lam:g}'
    d.mkdir(parents=True)
    (d / 'config.json').write_text(json.dumps(dict(configuration=dict(evaluate_test=False))))
    outputs = dict(val=dict(cross_entropy=val_ce, accuracy=val_acc))
    if test:
        outputs['test'] = dict(accuracy=1.0)
    (d / 'summary.json').write_text(json.dumps(dict(outputs=outputs, best_val_loss=val_ce)))
    if done:
        (d / 'DONE.json').write_text('{}')
    return dict(dataset='D', model='cam', reg_type='jsd_code', fold=0, seed=0,
                jsd_weight=lam, output=str(d), hidden=8)


def test_selection_uses_validation_only(tmp_path):
    import select_jsd as s
    members = [_fake_run(tmp_path, 100, 0.40, 0.80), _fake_run(tmp_path, 1000, 0.30, 0.70),
               _fake_run(tmp_path, 10000, 0.30, 0.75)]
    groups = s.groups_from_manifest([dict(args=a) for a in members])
    assert len(groups) == 1
    cands, problems = s.candidate_scores(members)
    assert not problems
    best = s.choose(cands)  # lowest val CE, tie broken by val accuracy
    assert best['jsd_weight'] == 10000
    assert set(best) == {'jsd_weight', 'layers', 'run', 'val_cross_entropy', 'val_accuracy',
                         'val_loss_checkpoint_monitor', 'best_epoch'}  # no test fields


def test_depth_and_lambda_grouping(tmp_path):
    """Fixed-depth groups vary lambda only; joint groups vary depth and lambda."""
    import select_jsd as s
    rows = []
    for L in (1, 2, 3):
        for lam in (100, 1000, 10000):
            a = _fake_run(tmp_path / f'L{L}', lam, 0.5 - 0.01 * L, 0.7)
            a['layers'] = L
            rows.append(dict(args=a))
    assert len(s.groups_from_manifest(rows, ('jsd_weight',))) == 3            # one per depth
    joint = s.groups_from_manifest(rows, ('layers', 'jsd_weight'))
    assert len(joint) == 1
    cands, _ = s.candidate_scores(next(iter(joint.values())))
    best = s.choose(cands)
    assert best['layers'] == 3 and best['jsd_weight'] == 100  # lowest val CE, then smaller lambda


def test_selection_refuses_test_touched_or_incomplete_grid(tmp_path):
    import select_jsd as s
    touched = [_fake_run(tmp_path / 'a', 100, 0.4, 0.8, test=True)]
    with pytest.raises(SystemExit):
        s.candidate_scores(touched)
    _, problems = s.candidate_scores([_fake_run(tmp_path / 'b', 100, 0.4, 0.8, done=False)])
    assert problems
    # Runs differing in anything but the selected hyperparameters are never pooled.
    mixed = [_fake_run(tmp_path / 'c', 100, 0.4, 0.8), _fake_run(tmp_path / 'd', 1000, 0.4, 0.8)]
    mixed[1]['hidden'] = 16
    assert len(s.groups_from_manifest([dict(args=a) for a in mixed])) == 2
    dup = [_fake_run(tmp_path / 'e', 100, 0.4, 0.8), _fake_run(tmp_path / 'f', 100, 0.3, 0.8)]
    with pytest.raises(SystemExit):  # same candidate twice in one group
        s.groups_from_manifest([dict(args=a) for a in dup])
