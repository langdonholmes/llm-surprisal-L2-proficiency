"""Shared pieces of the association-measure pilot: paths, loaders, constants.

Named ``assoc_common`` (not ``common``) so the pilot_corrected modules, which do
``from common import ...``, still find their own ``common.py``.
"""

import importlib.util
import sys
from pathlib import Path

from util.paths import DATA_DIR, FCE_DOCBINS_DIR, RESULTS_DIR, WI_DOCBINS_DIR

HERE = Path(__file__).resolve().parent
S3_SRC = HERE.parent
OUT = DATA_DIR / "study3" / "pilot_assoc"
TAB = OUT / "tables"
FREQ = DATA_DIR / "dolma" / "frequency_tables"
SEED = 42
N_BOOT = 2000

DOCBINS = {
    ("wi", "orig"): WI_DOCBINS_DIR,
    ("wi", "corr"): DATA_DIR / "study3" / "pilot_corrected" / "docbins_corrected",
    ("fce", "orig"): FCE_DOCBINS_DIR,
    ("fce", "corr"): DATA_DIR / "study3" / "pilot_fce" / "docbins_corrected",
}
TOKDIR = {"wi": DATA_DIR / "study3" / "pilot_corrected", "fce": DATA_DIR / "study3" / "pilot_fce"}
CONTENT = {"NOUN", "VERB", "ADJ", "ADV"}


def load(name: str, path: Path):
    """Import a module from a file whose name is not a valid identifier."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def tau() -> float:
    import json
    return json.loads((RESULTS_DIR / "study3" / "threshold.json").read_text())["tau_bits"]
