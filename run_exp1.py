#!/usr/bin/env python3
"""Experiment 1 (MUTAG): diversity regularizers for GKNN structural masks.

Four conditions, identical splits/seeds/architecture/protocol:
  none       L = CE
  jsd_code   L = CE - lambda/L_layers * sum_l JSD(softmax over filters)   (released code)
  jsd_paper  L = CE - lambda/L_layers * sum_l JSD(p_i(v), per graph)      (Cosmo et al.)
  logdet     L = CE + lambda/L_layers * sum_l -(1/m) log det(C^(l) + eps I) (ours)
             C^(l) = correlation matrix of the m filter responses of layer l

The losses are the ones in src/model.py (jsd_code, jsd_paper, logdet_penalty);
pairwise correlations are only an evaluation metric, never a training loss;
this script only checks them against independent reference formulas.
Lives at the GKNN repository root (next to src/). No test data are evaluated.
Run one condition, weight, fold and seed per invocation.
"""
from argparse import ArgumentParser, BooleanOptionalAction, Namespace
from hashlib import sha256
from importlib import metadata as package_metadata
from pathlib import Path
import inspect
import json
import math
import os
import sys
import time


METHODS = ('none', 'jsd_code', 'jsd_paper', 'logdet')
# Starting points only; the scales differ, so tune each on validation.
DEFAULT_WEIGHTS = {'none': 0.0, 'jsd_code': 1000.0, 'jsd_paper': 1.0, 'logdet': 1.0}


def build_parser():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--reg_type', choices=METHODS, required=True)
    p.add_argument('--reg_weight', '--jsd_weight', dest='jsd_weight', type=float, default=None,
                   help='lambda of the selected regularizer (stored as hparams.jsd_weight). '
                        'Default: %s.' % DEFAULT_WEIGHTS)
    p.add_argument('--dataset', default='MUTAG', choices=['MUTAG'])
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--folds', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--nodes', type=int, default=6)
    p.add_argument('--hidden', type=int, default=16)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--hops', type=int, default=1)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--max_epochs', type=int, default=1000)
    p.add_argument('--patience', type=int, default=300)
    p.add_argument('--lr', type=float, default=5e-3)
    p.add_argument('--lr_graph', type=float, default=1e-2)
    p.add_argument('--data_root', type=Path, default=Path('data'))
    p.add_argument('--output', type=Path, default=Path('results/exp1'))
    p.add_argument('--logdet_eps', type=float, default=1e-4,
                   help='epsilon in -(1/m) log det(C + eps I) (reg_type logdet; recorded for all).')
    p.add_argument('--std_tol', type=float, default=1e-8)
    p.add_argument('--deterministic', action=BooleanOptionalAction, default=True,
                   help='torch.use_deterministic_algorithms (default on; measured no CPU slowdown).')
    # Cluster / bookkeeping options (do not change the experiment itself).
    p.add_argument('--threads', type=int, default=1,
                   help='CPU threads for torch (1 per job is best when running many jobs).')
    p.add_argument('--skip_existing', action='store_true',
                   help='Exit 0 if this run already finished; move an unfinished '
                        'run directory aside (never deleted) and restart it.')
    p.add_argument('--prepare_only', action='store_true',
                   help='Only build the processed dataset and split file, then exit. '
                        'Run once before launching parallel jobs.')
    p.add_argument('--progress_bar', action='store_true',
                   help='Show the Lightning progress bar (off: one line per epoch).')
    return p


def run_directory(args):
    return (args.output / args.dataset /
            ('%s_w%g' % (args.reg_type, args.jsd_weight)) /
            ('fold%d_seed%d' % (args.fold, args.seed)))


def now():
    return time.strftime('%Y-%m-%d %H:%M:%S')


def update_json(path, **fields):
    """Merge fields into a small JSON file, atomically (safe to read concurrently)."""
    data = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding='utf-8'))
        except ValueError:
            data = {}
    data.update(fields)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding='utf-8')
    os.replace(tmp, path)


class Tee:
    """Duplicate stdout/stderr into run.log, flushing every write."""
    def __init__(self, stream, handle):
        self.stream, self.handle = stream, handle

    def write(self, text):
        self.stream.write(text)
        self.handle.write(text)
        self.handle.flush()
        return len(text)

    def flush(self):
        self.stream.flush()
        self.handle.flush()

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.stream, name)


def make_epoch_logger(run_dir, max_epochs):
    """Lightning callback: one log line + progress.json per epoch."""
    import pytorch_lightning as pl

    class EpochLogger(pl.Callback):
        def on_train_start(self, trainer, pl_module):
            self.t0 = self.t_epoch = time.perf_counter()
            self.best = float('inf')

        def on_train_epoch_start(self, trainer, pl_module):
            self.t_epoch = time.perf_counter()

        def on_train_epoch_end(self, trainer, pl_module):
            m = {k: float(v) for k, v in trainer.callback_metrics.items()
                 if getattr(v, 'numel', lambda: 1)() == 1}
            epoch = trainer.current_epoch + 1
            dt = time.perf_counter() - self.t_epoch
            elapsed = time.perf_counter() - self.t0
            val = m.get('val_loss', float('nan'))
            if val == val:
                self.best = min(self.best, val)
            get = lambda k: m.get(k, float('nan'))
            print('[%s] epoch %4d/%d  %.1fs  loss_ce=%.4f loss_reg=%.4f acc=%.1f  '
                  'val_loss=%.4f val_acc=%.1f  best_val=%.4f  const=%.2f min_eig=%.2e neg=%.2f'
                  % (now(), epoch, max_epochs, dt, get('loss_ce'), get('loss_reg'),
                     get('acc'), val, get('val_acc'), self.best,
                     get('frac_const_filters'), get('corr_min_eig'),
                     get('frac_neg_resp')), flush=True)
            update_json(run_dir / 'progress.json', epoch=epoch, max_epochs=max_epochs,
                        last_epoch_seconds=round(dt, 2), elapsed_seconds=round(elapsed, 1),
                        mean_epoch_seconds=round(elapsed / epoch, 2),
                        val_loss=val, best_val_loss=self.best, val_acc=get('val_acc'),
                        loss_ce=get('loss_ce'), loss_reg=get('loss_reg'),
                        frac_const_filters=get('frac_const_filters'),
                        corr_min_eig=get('corr_min_eig'), updated=now())

    return EpochLogger()


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                             allow_nan=False).encode()).hexdigest()


def state_hash(model):
    """Hash tensor values, not torch.save bytes (serialization can vary)."""
    h = sha256()
    for name, tensor in sorted(model.state_dict().items()):
        array = tensor.detach().cpu().contiguous().numpy()
        h.update(name.encode())
        h.update(str(array.shape).encode())
        h.update(str(array.dtype).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def load_splits(labels, data_root, n_folds):
    """Reuse the repository split cache; otherwise reproduce its split recipe."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold, train_test_split

    path = data_root / 'folds' / ('MUTAG_folds_%d.txt' % n_folds)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        folds = json.loads(path.read_text(encoding='utf-8'))
    else:
        labels = np.asarray(labels)
        outer = StratifiedKFold(n_splits=n_folds, random_state=1, shuffle=True)
        folds = []
        for train_val, test in outer.split(np.arange(len(labels)), labels):
            train, val = train_test_split(
                [int(i) for i in train_val], stratify=labels[train_val],
                test_size=int(len(train_val) * 0.1), random_state=0,
            )
            folds.append([train, val, [int(i) for i in test]])
        # Never silently overwrite an existing split file.
        with path.open('x', encoding='utf-8') as f:
            json.dump(folds, f)
    if len(folds) != n_folds:
        raise ValueError('Wrong number of folds in %s' % path)
    expected = set(range(len(labels)))
    for i, split in enumerate(folds):
        if len(split) != 3 or any(len(part) == 0 for part in split):
            raise ValueError('Malformed/empty train, validation or test split at fold %d' % i)
        flat = [int(j) for part in split for j in part]
        if set(flat) != expected or len(flat) != len(expected):
            raise ValueError('Overlapping, missing or repeated graph IDs in fold %d' % i)
    # Ensures this is an outer cross-validation partition, not arbitrary subsets.
    test_ids = [int(j) for split in folds for j in split[2]]
    if set(test_ids) != expected or len(test_ids) != len(expected):
        raise ValueError('Outer test folds do not partition the dataset.')
    return folds, path


def reference_regularizer(R, batch, reg_type, eps=1e-4):
    """Independent (loop-based) formulas, used only to verify src/model.py."""
    import torch
    entropy = lambda z: -(z * z.clamp_min(1e-12).log()).sum(-1)
    if reg_type == 'none':
        return R.sum() * 0
    if reg_type == 'jsd_code':
        q = R.softmax(-1)                              # q_v(i): over filters
        return entropy(q.mean(0)) - entropy(q).mean()
    if reg_type == 'jsd_paper':
        values = []
        for g in torch.unique(batch):                  # p_i(v): over vertices of G
            Rg = R[batch == g].clamp_min(0) + 1e-8
            P = Rg / Rg.sum(0, keepdim=True)           # column i = p_i
            values.append(entropy(P.mean(1)) - torch.stack(
                [entropy(P[:, i]) for i in range(P.shape[1])]).mean())
        return torch.stack(values).mean()
    if reg_type == 'logdet':
        m = R.shape[1]
        cols = []
        for i in range(m):                             # collapsed column -> zero vector
            c = R[:, i] - R[:, i].mean()
            active = float(c.detach().pow(2).mean().sqrt()) > 1e-6
            cols.append(c / c.norm() if active else torch.zeros_like(c))
        C = torch.stack([torch.stack([cols[i] @ cols[j] for j in range(m)]) for i in range(m)])
        eig = torch.linalg.eigvalsh(C.double())        # eigenvalues, not Cholesky
        return -(torch.log(eig + eps).sum() / m).to(R.dtype)
    raise ValueError(reg_type)


def check_regularizer(model, reg_type, weight, hidden, eps):
    """Verify model.regularizers == sign * lambda * reference formula, with gradients.

    Uses a private generator, so the global RNG (and hence training) is untouched.
    """
    import torch
    g = torch.Generator().manual_seed(12345)
    R = (torch.rand(7, hidden, generator=g) + 0.1).requires_grad_(True)
    R.data[:, -1] = 0.5                                # include one constant filter
    batch = torch.tensor([0, 0, 0, 1, 1, 1, 1], dtype=torch.long)
    try:
        actual = model.regularizers(R, batch)
    except (TypeError, AttributeError) as exc:
        raise RuntimeError(
            'src/model.py must be your edited version with reg_type and '
            'regularizers(x, batch).'
        ) from exc
    sign = {'none': 0.0, 'jsd_code': -1.0, 'jsd_paper': -1.0, 'logdet': 1.0}[reg_type]
    expected = sign * weight * reference_regularizer(R, batch, reg_type, eps)
    if actual.ndim != 0 or not torch.isfinite(actual):
        raise RuntimeError('The regularizer must return one finite scalar.')
    if not torch.allclose(actual, torch.as_tensor(expected, dtype=actual.dtype),
                          rtol=1e-4, atol=1e-6):
        raise RuntimeError('model.regularizers (%g) does not match the reference %s '
                           'formula (%g).' % (float(actual), reg_type, float(expected)))
    grad = torch.autograd.grad(actual, R)[0]
    if not torch.isfinite(grad).all():
        raise RuntimeError('Nonfinite regularizer gradient in the preflight check.')
    if reg_type != 'none' and not torch.any(grad != 0):
        raise RuntimeError('The %s preflight produced no response gradient.' % reg_type)


def main():
    args = build_parser().parse_args()
    if args.jsd_weight is None:
        args.jsd_weight = DEFAULT_WEIGHTS[args.reg_type]
    if args.reg_type == 'none' and args.jsd_weight != 0:
        raise SystemExit('Use --reg_weight 0 with --reg_type none.')
    if args.reg_type != 'none' and args.jsd_weight <= 0:
        raise SystemExit('Use a positive --reg_weight with --reg_type %s.' % args.reg_type)
    if not math.isfinite(args.jsd_weight):
        raise SystemExit('--jsd_weight must be finite.')
    if args.folds < 2 or not 0 <= args.fold < args.folds:
        raise SystemExit('Require folds >= 2 and 0 <= fold < folds.')
    if args.hidden < 2 or min(args.nodes, args.layers, args.hops, args.batch_size,
                              args.max_epochs, args.patience) < 1:
        raise SystemExit('Use at least 2 masks and positive integer hyperparameters.')
    if not (math.isfinite(args.logdet_eps) and args.logdet_eps > 0):
        raise SystemExit('--logdet_eps must be finite and positive.')
    if not math.isfinite(args.std_tol) or args.std_tol < 0:
        raise SystemExit('--std_tol must be finite and nonnegative.')
    if args.threads < 1:
        raise SystemExit('--threads must be >= 1.')
    os.environ.setdefault('OMP_NUM_THREADS', str(args.threads))
    os.environ.setdefault('MKL_NUM_THREADS', str(args.threads))

    if args.prepare_only:
        _run(args, None)
        return

    run_dir = run_directory(args)
    if (run_dir / 'diagnostics.json').exists() and args.skip_existing:
        print('Already finished, skipping:', run_dir)
        return
    if run_dir.exists():
        if not args.skip_existing:
            raise SystemExit('Output already exists: %s. Use --skip_existing to resume, '
                             'or another --output.' % run_dir)
        aside = run_dir.with_name(run_dir.name + '_incomplete_' + time.strftime('%Y%m%d-%H%M%S'))
        os.replace(run_dir, aside)
        print('Moved unfinished run aside:', aside)
    run_dir.mkdir(parents=True)

    import platform
    import traceback
    log_handle = (run_dir / 'run.log').open('a', encoding='utf-8')
    sys.stdout = Tee(sys.__stdout__, log_handle)
    sys.stderr = Tee(sys.__stderr__, log_handle)
    status = run_dir / 'status.json'
    update_json(status, state='running', started=now(), host=platform.node(), pid=os.getpid(),
                command=' '.join(sys.argv),
                slurm_job=os.environ.get('SLURM_JOB_ID'),
                slurm_array_task=os.environ.get('SLURM_ARRAY_TASK_ID'))
    print('[%s] start %s on %s (pid %d)' % (now(), run_dir, platform.node(), os.getpid()))
    t0 = time.perf_counter()
    try:
        result = _run(args, run_dir)
    except BaseException as exc:
        update_json(status, state='failed', finished=now(),
                    wall_seconds=round(time.perf_counter() - t0, 1),
                    error='%s: %s' % (type(exc).__name__, exc),
                    traceback=traceback.format_exc()[-4000:])
        print('[%s] FAILED: %s' % (now(), exc))
        raise
    update_json(status, state='done', finished=now(),
                wall_seconds=round(time.perf_counter() - t0, 1), **result)
    print('[%s] done' % now())


def _run(args, run_dir):
    """The experiment proper. run_dir is None for --prepare_only."""
    repo = Path(__file__).resolve().parent
    src = repo / 'src'
    if not (src / 'model.py').exists() or not (src / 'dataset.py').exists():
        raise SystemExit('Copy run_exp1.py and exp1_diagnostics.py next to the '
                         'src/ directory in your existing GKNN repository.')
    os.chdir(repo)  # The existing repository uses relative paths internally.
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # This supplied GKNN code uses CPU/GraKeL.
    sys.path.insert(0, str(src))

    import numpy as np
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from dataset import TUDataset, EgonetLoader
    from model import Model
    from layers import max_comp
    from exp1_diagnostics import export_response_diagnostics
    from exp1_structure import export_structural_diagnostics
    torch.set_num_threads(args.threads)

    # The supplied max_comp used zip(*E) directly as sparse coordinates.
    # Recent SciPy requires the explicit (rows, cols) tuple; fail before training.
    try:
        max_comp({(0, 1), (1, 0)}, {0: 0, 1: 0})
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            'The supplied layers.max_comp failed a two-node check. '
            'See README section 2: use rows, cols = zip(*E), then '
            'csr_matrix((np.ones(len(E)), (rows, cols)), '
            'shape=(np.max(E)+1, np.max(E)+1)).'
        ) from exc

    pl.seed_everything(args.seed, workers=True)
    dataset = TUDataset(str(args.data_root), args.dataset, hops=args.hops)
    labels = [int(d.y) for d in dataset]
    folds, split_path = load_splits(labels, args.data_root, args.folds)
    if run_dir is None:
        print('Prepared %s (%d graphs, processed under %s) and splits %s.'
              % (args.dataset, len(dataset), args.data_root, split_path))
        return None
    train_ids, val_ids, test_ids = folds[args.fold]
    train_dataset, val_dataset = dataset[train_ids], dataset[val_ids]
    # Deliberately no test loader: this is a development/diagnostic experiment.
    train_loader = EgonetLoader(train_dataset[:], batch_size=args.batch_size, shuffle=True)
    val_loader = EgonetLoader(val_dataset, batch_size=2000, shuffle=False)

    versions = {}
    for package in ['torch', 'pytorch-lightning', 'torch-geometric', 'GraKeL',
                    'numpy', 'scipy', 'scikit-learn']:
        try:
            versions[package] = package_metadata.version(package)
        except package_metadata.PackageNotFoundError:
            versions[package] = 'unknown'
    source_hashes = {
        str(p.relative_to(src)): sha256(p.read_bytes()).hexdigest()
        for p in sorted(src.rglob('*.py'))
    }
    protocol = {
        'dataset': args.dataset, 'dataset_labels_sha256': canonical_hash(labels),
        'folds': args.folds, 'splits_sha256': canonical_hash(folds),
        'nodes': args.nodes, 'hidden': args.hidden, 'layers': args.layers,
        'hops': args.hops, 'kernel': 'wl', 'pooling': 'add', 'mlp_layers': 2,
        'activation': 'relu', 'normalize': True, 'max_cc': True,
        'optimize_masks': True, 'batch_size': args.batch_size,
        'max_epochs': args.max_epochs, 'patience': args.patience,
        'lr': args.lr, 'lr_graph': args.lr_graph,
        'val_batch_size': 2000, 'std_tol': args.std_tol, 'logdet_eps': args.logdet_eps,
        'deterministic': args.deterministic,
        'weighting': 'equal_per_node', 'source_sha256': source_hashes,
        'versions': versions,
        'experiment_source_sha256': {
            name: sha256((repo / name).read_bytes()).hexdigest()
            for name in ['run_exp1.py', 'exp1_diagnostics.py', 'exp1_structure.py']
        },
    }
    hparams = Namespace(
        **vars(args), in_features=train_dataset.num_features,
        labels=train_dataset.num_features, num_classes=train_dataset.num_classes,
        mlp_layers=2, activation='relu', kernel='wl', normalize=True,
        pooling='add', sparsity=0.0, max_cc=True, mask=False,
        optimize_masks=True, project='GKNN_EXP1', run=args.seed, debug=False,
        loss=torch.nn.CrossEntropyLoss(),
    )
    # Match initialization independently of dataset preprocessing/cache activity.
    pl.seed_everything(args.seed, workers=True)
    model = Model(hparams)
    check_regularizer(model, args.reg_type, args.jsd_weight, args.hidden, args.logdet_eps)
    initial_digest = state_hash(model)

    config = {
        'dataset': args.dataset, 'reg_type': args.reg_type,
        'jsd_weight': args.jsd_weight, 'fold': args.fold, 'seed': args.seed,
        'objective': {
            'none': 'CE',
            'jsd_code': 'CE - lambda * mean_l JSD_code(R^l)',
            'jsd_paper': 'CE - lambda * mean_l JSD_paper(R^l)',
            'logdet': 'CE + lambda * mean_l -(1/m) log det(C^l + %g I)' % args.logdet_eps,
        }[args.reg_type],
        'protocol_id': canonical_hash(protocol), 'protocol': protocol,
        'initial_state_sha256': initial_digest,
        'split_file': str(split_path.resolve()),
        'train_graph_ids': train_ids, 'validation_graph_ids': val_ids,
        'test_graph_ids_not_evaluated': test_ids,
    }
    (run_dir / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')

    checkpoint = ModelCheckpoint(
        dirpath=str(run_dir / 'checkpoints'), filename='best-{epoch:04d}',
        monitor='val_loss', mode='min', save_top_k=1, save_last=True,
        save_weights_only=True,
    )
    early_stopping = EarlyStopping(monitor='val_loss', mode='min',
                                  min_delta=0.0, patience=args.patience)
    logger = CSVLogger(save_dir=str(run_dir), name='training', version=0)
    trainer_args = dict(
        max_epochs=args.max_epochs, accelerator='cpu', devices=1,
        logger=logger,
        callbacks=[checkpoint, early_stopping, make_epoch_logger(run_dir, args.max_epochs)],
        num_sanity_val_steps=0, log_every_n_steps=1,
        enable_progress_bar=args.progress_bar, enable_model_summary=False,
        default_root_dir=str(run_dir), deterministic=args.deterministic,
    )
    if 'inference_mode' in inspect.signature(pl.Trainer).parameters:
        trainer_args['inference_mode'] = False
    trainer = pl.Trainer(**trainer_args)
    start = time.perf_counter()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    train_seconds = time.perf_counter() - start
    if getattr(trainer, 'interrupted', False):
        raise SystemExit('Training was interrupted; no completed-run diagnostics exported.')
    if not checkpoint.best_model_path:
        raise RuntimeError('No best validation checkpoint was saved.')

    # This is the checkpoint just written by this process. Explicit loading avoids
    # relying on the final-epoch model or on Lightning load_from_checkpoint's
    # constructor conventions for the legacy Model(hparams) class.
    load_args = {'map_location': 'cpu'}
    if 'weights_only' in inspect.signature(torch.load).parameters:
        load_args['weights_only'] = False
    saved = torch.load(checkpoint.best_model_path, **load_args)
    model.load_state_dict(saved['state_dict'], strict=True)
    model.cpu()
    metadata = dict(config)
    metadata.update({
        'checkpoint_path': str(Path(checkpoint.best_model_path).resolve()),
        'checkpoint_epoch': int(saved['epoch']),  # zero-based, as in Lightning
        'checkpoint_val_loss': float(checkpoint.best_model_score),
        'train_seconds': train_seconds,
        'trainer_current_epoch': int(trainer.current_epoch),
    })
    summary = export_response_diagnostics(
        model, val_loader, run_dir, graph_ids=val_ids,
        metadata=metadata, std_tol=args.std_tol,
    )
    # Decode exactly the masks used by the supplied kernel layer, including
    # its largest-component rule. This also saves their actual graphs.
    structure = export_structural_diagnostics(model, run_dir)
    for layer_summary, structure_summary in zip(summary['layers'], structure):
        if layer_summary['layer'] != structure_summary['layer']:
            raise RuntimeError('Structural and response layer indices disagree.')
        layer_summary['structural'] = structure_summary['structural']
    (run_dir / 'diagnostics.json').write_text(
        json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8'
    )
    print('\nCompleted validation diagnostics:', run_dir.resolve())
    print('Training time: %.1f minutes' % (train_seconds / 60))
    print('Validation accuracy: %.2f%%' % (100 * summary['classification']['accuracy']))
    print('Test data have not been evaluated.')
    return {
        'train_seconds': round(train_seconds, 1),
        'epochs_run': int(trainer.current_epoch),
        'stopped_early': bool(early_stopping.stopped_epoch),
        'best_epoch': int(saved['epoch']) + 1,
        'best_val_loss': float(checkpoint.best_model_score),
        'val_accuracy': summary['classification']['accuracy'],
    }


if __name__ == '__main__':
    main()
