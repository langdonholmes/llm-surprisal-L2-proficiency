"""Shared pieces of the RQ3 pilot: paths, feature names, the B / S feature sets.

Reads the D28 per-unit split (``data/study3/pilot_types/units_{wi,fce}.parquet``)
and the edit tables of ``pilot_types``. No model is run.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
S3_SRC = HERE.parent
for p in (str(S3_SRC), str(S3_SRC / "pilot_types")):
    if p not in sys.path:
        sys.path.insert(0, p)

from types_common import CORPORA, GROUPS, OUT as TYPES_OUT  # noqa: E402,F401
from util.paths import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "study3" / "pilot_rq3"
TAB = OUT / "tables"
SEED = 42
N_BOOT = 2000
N_REPEATS = 10
N_FOLDS = 10
ALPHAS = [10.0 ** k for k in range(-3, 6)]  # inner CV grid (log-spaced)


def slug(g: str) -> str:
    return g.replace("/", "_").replace(" ", "_")


RATE = [f"rate_{slug(g)}" for g in GROUPS]
COST = [f"cost_{slug(g)}" for g in GROUPS]
SPILL = [f"spill_{slug(g)}" for g in GROUPS]

B = RATE + ["rate_flagged", "log_n"]
S_FULL = ["clean"] + COST + SPILL + ["cost_flagged"]
S_COMPACT = ["clean", "cost_mean", "spill_mean"]
# features that are undefined when the essay has no error of that kind (imputed in-fold)
MAY_BE_MISSING = COST + SPILL + ["cost_flagged", "cost_mean", "spill_mean"]

MODELS = {
    "B": B,
    "B+S_compact": B + S_COMPACT,
    "B+S_full": B + S_FULL,
    "S_compact": S_COMPACT,
    "S_full": S_FULL,
    "mean_surprisal": ["total"],
    "B+mean_surprisal": B + ["total"],
}
