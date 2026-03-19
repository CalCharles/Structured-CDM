"""Console-script entry point for ``struct-sdfm-train``.

Delegates to scripts/run_train.py so the CLI logic lives in one place.
"""

import importlib.util
from pathlib import Path


def main():
    script = Path(__file__).resolve().parent.parent.parent / 'scripts' / 'run_train.py'
    spec   = importlib.util.spec_from_file_location('run_train', script)
    mod    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main()
