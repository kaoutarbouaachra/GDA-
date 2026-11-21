"""Opt-in CAM integration into the user's CURRENT GDA- GKNN repository.

Dry run: python scripts/apply_cam_patch.py
Apply:   python scripts/apply_cam_patch.py --apply

Only patches src/model.py and run_training.py. Never modifies src/layers.py.
Creates *.before_cam_final backups before applying changes.
"""
from __future__ import annotations
import argparse
import ast
import difflib
from pathlib import Path
import re
import shutil

FACTORY = '''\n\ndef _cam_or_gkernel(hparams, layer_index, nodes, labels, filters, **kwargs):
    """Preserve original GKNN; choose connected CAM on request."""
    mode = getattr(hparams, "mask_distribution", "original")
    if mode == "original":
        return GKernel(nodes, labels, filters, **kwargs)
    if mode != "cam":
        raise ValueError("Invalid mask_distribution: " + str(mode))
    # For layer>=2, soft categorical node labels let gradients propagate
    # to previous layers; this is a documented extension of original GKNN.
    layer_index = int(layer_index)
    kernel_names = str(getattr(hparams, "kernel", "wl")).split("+")
    if layer_index > 0 and len(kernel_names) != 1:
        raise ValueError("Multilayer CAM currently requires exactly one graph kernel")
    return CAMKernel(
        nodes, labels, filters,
        mc_samples=int(getattr(hparams, "cam_samples", 8)),
        eval_samples=int(getattr(hparams, "cam_eval_samples", 32)),
        initial_stop_prob=float(getattr(hparams, "cam_initial_stop_prob", 0.1)),
        eval_seed=int(getattr(hparams, "cam_eval_seed", 1729)) + layer_index,
        input_label_mode="argmax" if layer_index == 0 else "categorical",
        input_temperature=float(getattr(hparams, "cam_input_temperature", 1.0)),
        **kwargs,
    )
\n\n'''

ARGS = {
    'mask_distribution': '    parser.add_argument("--mask_distribution", choices=["original", "cam"], default="original")\n',
    'cam_samples': '    parser.add_argument("--cam_samples", type=int, default=8)\n',
    'cam_eval_samples': '    parser.add_argument("--cam_eval_samples", type=int, default=32)\n',
    'cam_initial_stop_prob': '    parser.add_argument("--cam_initial_stop_prob", type=float, default=0.1)\n',
    'cam_eval_seed': '    parser.add_argument("--cam_eval_seed", type=int, default=1729)\n',
    'cam_input_temperature': '    parser.add_argument("--cam_input_temperature", type=float, default=1.0)\n',
}


def patch_model(src: str):
    # Replace either the ORIGINAL constructors or those patched by old CAM v1.
    if 'def _new_mask_layer(' in src:
        # Old CAM v1 factory appears just before Model.
        begin=src.index('\ndef _new_mask_layer(')
        end=src.find('\nclass Model(pl.LightningModule):',begin)
        if end < 0:
            raise ValueError('Old CAM factory cannot be located safely')
        src=src[:begin]+src[end:]
    original = r'(?:GKernel\(|_new_mask_layer\(hparams,\s*)'
    first = r'self\.conv_layers\.append\(' + original + r'hparams\.nodes\s*,\s*labels\s*,\s*hidden\s*,'
    higher = r'self\.conv_layers\.append\(' + original + r'hparams\.nodes\s*,\s*hidden\s*,\s*hidden\s*,'
    src, c1 = re.subn(first, 'self.conv_layers.append(_cam_or_gkernel(hparams, 0, hparams.nodes, labels, hidden,',src)
    src, c2 = re.subn(higher,'self.conv_layers.append(_cam_or_gkernel(hparams, i, hparams.nodes, hidden, hidden,',src)
    already = src.count('self.conv_layers.append(_cam_or_gkernel(')
    if (c1,c2) != (1,1) and not (already==2 and c1==0 and c2==0):
        raise ValueError('Unexpected layer factory calls. Do not patch automatically: '
                         f'first={c1}, higher={c2}, current={already}')
    if 'from cam import CAMKernel' not in src:
        needle='from layers import *'
        if needle not in src:
            raise ValueError('Missing legacy import from layers')
        src=src.replace(needle,needle+'\nfrom cam import CAMKernel',1)
    if 'def _cam_or_gkernel(' not in src:
        marker='class Model(pl.LightningModule):'
        if marker not in src:
            raise ValueError('Missing GKNN Model class')
        src=src.replace(marker,FACTORY+marker,1)
    return src


def patch_training(src: str):
    additions=[]
    for arg,line in ARGS.items():
        if re.search(r'add_argument\(\s*[\'\"]--'+re.escape(arg)+r'[\'\"]',src) is None:
            additions.append(line)
    if additions:
        positions=list(re.finditer(r'^\s{4}return parser\s*$',src,re.MULTILINE))
        if len(positions)!=1:
            raise ValueError('Could not find unique get_arg_parser() return parser')
        at=positions[0].start()
        src=src[:at]+''.join(additions)+src[at:]
    return src


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--repo',type=Path,default=Path('.'))
    ap.add_argument('--apply',action='store_true')
    opts=ap.parse_args()
    root=opts.repo.resolve()
    model_path=root/'src'/'model.py'
    train_path=root/'run_training.py'
    cam_path=root/'src'/'cam.py'
    if not cam_path.is_file():
        raise SystemExit('Copy src/cam.py to the repository before patching')
    changes=[]
    for path, func in ((model_path,patch_model),(train_path,patch_training)):
        if not path.exists():
            raise SystemExit('Missing '+str(path))
        old=path.read_text(encoding='utf-8')
        new=func(old)
        try:
            ast.parse(new,filename=str(path))
        except SyntaxError as exc:
            raise SystemExit('Patched Python would have invalid syntax: '+str(exc))
        changes.append((path,old,new))
    for path,old,new in changes:
        print('\n###',path,'###')
        if old==new:
            print('No changes')
        else:
            diff=list(difflib.unified_diff(old.splitlines(True),new.splitlines(True),
                                           fromfile=str(path)+' (before)',tofile=str(path)+' (proposed)'))
            print(''.join(diff[:160]))
            if len(diff)>160: print(f'... {len(diff)-160} additional diff lines omitted')
    if not opts.apply:
        print('\nDRY RUN: No files changed. Run with --apply after reviewing.')
        return
    # Verify backups can be created BEFORE modifying any source file.
    for path,old,new in changes:
        if old!=new and path.with_suffix(path.suffix+'.before_cam_final').exists():
            raise SystemExit('Backup already exists; refusing overwrite: '+str(path))
    for path,old,new in changes:
        if old!=new:
            backup=path.with_suffix(path.suffix+'.before_cam_final')
            shutil.copy2(path,backup)
            path.write_text(new,encoding='utf-8')
            print('Updated:',path,'Backup:',backup)
    print('CAM installed. Inspect git diff, run tests and smoke-test before long training.')

if __name__=='__main__':
    main()
