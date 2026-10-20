"""Shared pieces of the correction-bucket pilot: buckets, paths, loaders.

Four fixed correction buckets over ERRANT categories (operation prefix
stripped), plus "unclassified" (UNK, or an edit ERRANT could not type):

    mechanical           SPELL ORTH PUNCT VERB:FORM VERB:SVA VERB:INFL VERB:TENSE
                         NOUN:NUM NOUN:INFL MORPH ADJ:FORM CONTR NOUN:POSS
    function word        DET PREP PRON CONJ PART
    content-word choice  NOUN VERB ADJ ADV
    multi-word / order   OTHER WO

Reuses, without copying: ``pilot_types/3_decompose.py`` (the D28 credit table
and covariance shares), ``pilot_assoc/2_classic.py`` (the classic measures) and
``pilot_corrected/common.py`` (``apply_edits``).
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
S3_SRC = HERE.parent
for p in (str(HERE), str(S3_SRC)):
    if p not in sys.path:
        sys.path.insert(0, p)

from util.paths import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "study3" / "pilot_buckets"
TAB = OUT / "tables"
SEED = 42
N_BOOT = 2000

BUCKETS = ["mechanical", "function word", "content-word choice", "multi-word / order"]
UNCL = "unclassified"
PLAYERS = BUCKETS + [UNCL]
_MAP = {}
for c in ["SPELL", "ORTH", "PUNCT", "VERB:FORM", "VERB:SVA", "VERB:INFL", "VERB:TENSE", "NOUN:NUM",
          "NOUN:INFL", "MORPH", "ADJ:FORM", "CONTR", "NOUN:POSS"]:
    _MAP[c] = "mechanical"
for c in ["DET", "PREP", "PRON", "CONJ", "PART"]:
    _MAP[c] = "function word"
for c in ["NOUN", "VERB", "ADJ", "ADV"]:
    _MAP[c] = "content-word choice"
for c in ["OTHER", "WO"]:
    _MAP[c] = "multi-word / order"


def bucket_of(errant_type) -> str:
    if errant_type is None or (isinstance(errant_type, float) and np.isnan(errant_type)):
        return UNCL
    t = str(errant_type)
    cat = t[2:] if t[:2] in ("M:", "R:", "U:") else t
    return _MAP.get(cat, UNCL)


def load(name: str, path: Path):
    """Import a module from a file whose name is not a valid identifier."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def cov_shares(X, y, cols, ref=0, n_boot=N_BOOT, seed=SEED):
    """Share of cov(X[:, ref], y) carried by every column of X, unit bootstrap CI.

    Returns {col: (share, lo, hi)}. Units with any NaN must be dropped before.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    rng = np.random.default_rng(seed)

    def sh(i):
        c = ((X[i] - X[i].mean(0)) * (y[i] - y[i].mean())[:, None]).mean(0)
        return c / c[ref]

    pt = sh(np.arange(len(y)))
    bs = np.array([sh(rng.integers(0, len(y), len(y))) for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5], axis=0)
    return {c: (float(p), float(a), float(b)) for c, p, a, b in zip(cols, pt, lo, hi)}


def fmt(t):
    return f"{100 * t[0]:.0f} [{100 * t[1]:.0f}, {100 * t[2]:.0f}]"
