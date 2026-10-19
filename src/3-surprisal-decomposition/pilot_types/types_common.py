"""Shared pieces of the error-type split: paths, corpora, groups, edit tables.

Reuses, without copying: the ERRANT grouping (``pilot_shapley/common.py``), the
M2 walk-back (``pilot_shapley/1_errant.py``), the edit-level Shapley helpers
(``pilot_shapley/engine.py``), and the corrections and alignments of
``pilot_corrected`` (W&I+LOCNESS) and ``pilot_fce`` (FCE).
"""

import importlib.util
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
S3_SRC = HERE.parent
SH = S3_SRC / "pilot_shapley"
for p in (str(S3_SRC), str(SH)):
    if p not in sys.path:
        sys.path.insert(0, p)

from util.paths import DATA_DIR, FCE_DIR  # noqa: E402

OUT = DATA_DIR / "study3" / "pilot_types"
TAB = OUT / "tables"
SEED = 42
N_BOOT = 2000
WIN = 8
FCE_RAW = FCE_DIR / "raw" / "fce"

CORPORA = {
    "wi": {"data": DATA_DIR / "study3" / "pilot_corrected",
           "edit_types": DATA_DIR / "study3" / "pilot_shapley" / "edit_types.parquet"},
    "fce": {"data": DATA_DIR / "study3" / "pilot_fce",
            "edit_types": OUT / "edit_types_fce.parquet"},
}
OPS = {"M": "missing", "R": "replacement", "U": "unnecessary"}


def load(name: str, path: Path):
    """Import a module from a file whose name is not a valid identifier."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


# pilot_shapley scripts do `from common import ...`, so its common is registered
# under that name; this module is named types_common to avoid the clash.
sh_common = load("common", SH / "common.py")
GROUPS = sh_common.GROUPS
group_of = sh_common.group_of


def op_of(errant_type: str) -> str:
    return OPS.get(str(errant_type)[:1], "unmapped") if str(errant_type)[1:2] == ":" else "unmapped"


def edit_table(corpus: str) -> pd.DataFrame:
    """Non-flagged first-annotator edits with ERRANT type, group and operation."""
    et = pd.read_parquet(CORPORA[corpus]["edit_types"])
    et = et[et.kind != "flagged"].copy()
    et["op"] = et.errant.map(op_of)
    return et
