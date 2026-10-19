"""Shared pieces of the FCE replication of the D24 split.

Named ``fce_common`` (not ``common``) so that the pilot_corrected modules, which
do ``from common import ...``, still get their own ``common.py`` when loaded
here. All the reused code is loaded from pilot_corrected and 2_select.py, not
copied.
"""

import importlib.util
import sys
from pathlib import Path

from util.paths import DATA_DIR, FCE_DIR, FCE_DOCBINS_DIR

HERE = Path(__file__).resolve().parent
S3_SRC = HERE.parent
PC = S3_SRC / "pilot_corrected"
FCE_M2_DIR = FCE_DIR / "raw" / "fce" / "m2"
OUT = DATA_DIR / "study3" / "pilot_fce"
TAB = OUT / "tables"
SEED = 42
N_BOOT = 2000

for p in (str(PC), str(S3_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)


def load(name: str, path: Path):
    """Import a module from a file whose name is not a valid identifier."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def pc(stem: str):
    """A pilot_corrected module (common, 2_score, 3_analyze, 5_decompose)."""
    return load(f"pc_{stem}", PC / f"{stem}.py")


def select():
    """src/3-surprisal-decomposition/2_select.py (for gold_for_doc)."""
    return load("s3_select", S3_SRC / "2_select.py")
