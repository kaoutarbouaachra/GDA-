"""Read-only preflight for GDA- CAM / GKNN repository integration.

Run from the GDA- repository root: python scripts/check_setup.py
Never modifies repository files and never launches training.
"""
from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys


def main():
    root = Path.cwd()
    required = [
        'src/model.py', 'src/layers.py', 'src/dataset.py',
        'src/cam.py', 'run_training.py', 'run_tracked.py',
        'src/experiment_tracking.py', 'src/performance_tracking.py',
        'src/synthetic_data.py', 'scripts/generate_motifs.py',
        'scripts/fig8_extended.py', 'cluster/array.slurm',
        'tests/test_cam.py',
    ]
    missing = []
    print('Checking:', root)
    for rel in required:
        path = root / rel
        exists = path.is_file()
        print(('OK ' if exists else 'MISSING '), rel)
        if not exists:
            missing.append(rel)
        elif path.suffix == '.py':
            try:
                ast.parse(path.read_text(encoding='utf-8'), filename=rel)
            except SyntaxError as e:
                print('  SYNTAX ERROR:', e)
                missing.append(rel + ' (syntax)')
    model = root / 'src/model.py'
    if model.exists():
        src = model.read_text(encoding='utf-8')
        cam_factory = '_cam_or_gkernel(' in src or 'CAMKernel(' in src
        cam_import = 'from cam import CAMKernel' in src or 'import cam' in src
        if cam_factory and cam_import:
            print('OK: model.py appears to expose a CAM layer factory')
        else:
            print('WARNING: model.py may not be integrated with CAM.')
            print('  Review scripts/apply_cam_patch.py (dry run only)')
    for lib in ('torch','pytorch_lightning','torch_geometric','grakel',
                'networkx','scipy','sklearn','matplotlib','pandas','numpy'):
        installed = importlib.util.find_spec(lib) is not None
        print(('OK ' if installed else 'NOT INSTALLED '),lib)
    if missing:
        print('\nSome required files are missing or invalid:', ', '.join(missing))
        return 1
    print('\nFile preflight passed. NEXT: pytest, then two 2-epoch smoke tests.')
    print('This check does NOT prove full GraKeL / PyG model compatibility.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
