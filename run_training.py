"""Compatibility entry point for the assembled GDA- package.

The exact local run_training.py from the user's machine was not available.
This launcher accepts the main legacy flags and delegates to run_tracked.py.
For reproducible jobs, prefer `python run_tracked.py --output ...`.
"""
import sys
import time
from pathlib import Path


def _translate(argv):
    args = list(argv)
    for old, new in (("--mask_distribution", "--model"),):
        args = [new if x == old else x for x in args]
    # Optional legacy logger flag, not used by the tracked runner.
    if "--logger" in args:
        i = args.index("--logger")
        del args[i:i+2]
    if "--output" not in args:
        args.extend(("--output", str(Path("results") / ("legacy_"+str(int(time.time()))))))
    return args


if __name__ == "__main__":
    sys.argv[1:] = _translate(sys.argv[1:])
    from run_tracked import main
    main()
