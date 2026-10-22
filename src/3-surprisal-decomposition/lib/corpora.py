"""Paths, corpora and analysis units shared by the stages.

Two gold-corrected corpora. W&I+LOCNESS: the unit is the essay and the
criterion is CEFR level (A = 1, B = 2, C = 3; LOCNESS, level N, is described
but has no criterion). FCE: the unit is the script (both answers pooled) and
the criterion is the script score (1-40; the one script scored 0 is left out).
ELLIPSE: the unit is the essay and the criterion the overall score (1-5).

A corpus id names the texts and where their edits come from. ``wi`` and
``fce`` take the first annotator's json edits (the main analysis). The other
ids take ERRANT's edits from ``corrector/c5_edits.py`` (VARIANTS: id -> base
corpus, c5 source):
    wi_model, fce_model   the cross-fitted corrector's corrections
    wi_errant, fce_errant the annotator's corrected texts, cut into edits by
                          ERRANT like the model's, so that model and gold differ
                          only in the corrections
    ellipse               the final corrector's corrections
"""

import json

from util.paths import (
    DATA_DIR,
    ELLIPSE_DOCBINS_DIR,
    FCE_DIR,
    FCE_DOCBINS_DIR,
    RESULTS_DIR,
    WI_DIR,
    WI_DOCBINS_DIR,
)

DATA = DATA_DIR / "study3"
RESULTS = RESULTS_DIR / "study3"
# The ELLIPSE token surprisal stored by the appendix's GPU pass (appendix/A0_extract.py)
ELLIPSE_APPENDIX = DATA / "appendix" / "ellipse"
CORPORA = ["wi", "fce"]
VARIANTS = {"wi_model": ("wi", "model"), "fce_model": ("fce", "model"),
            "wi_errant": ("wi", "gold"), "fce_errant": ("fce", "gold"),
            "ellipse": ("ellipse", "ellipse")}
ALL_IDS = CORPORA + list(VARIANTS)
VALIDATION_IDS = ["wi_model", "fce_model", "wi_errant", "fce_errant"]  # only the corrector check uses them
DOCBINS = {"wi": WI_DOCBINS_DIR, "fce": FCE_DOCBINS_DIR, "ellipse": ELLIPSE_DOCBINS_DIR}
RAW_DIRS = {"wi": WI_DIR / "raw" / "wi+locness", "fce": FCE_DIR / "raw" / "fce"}
CEFR_SCORE = {"A": 1.0, "B": 2.0, "C": 3.0}
FCE_BANDS = [(1, 23), (24, 27), (28, 31), (32, 40)]
# overall score 1-5 in halves; the closest the nine values allow to quartiles (DECISIONS D41)
ELLIPSE_BANDS = [(1.0, 2.5), (3.0, 3.0), (3.5, 3.5), (4.0, 5.0)]
SEED = 42
N_BOOT = 2000

# RQ3 feature sets (columns of units_<corpus>.parquet, written by 5_decompose.py)
GROUP_SLUGS = ["spelling_orth", "verb_form_SVA_tense", "determiner", "preposition", "punctuation",
               "noun_number", "word_order", "other_lexical", "other"]
ERROR_COUNTS = [f"rate_{g}" for g in GROUP_SLUGS] + ["rate_flagged", "log_n"]  # "B" in the write-up
S_COMPACT = ["clean", "cost_mean", "spill_mean"]


def base_corpus(corpus: str) -> str:
    """The corpus whose texts an id uses (wi_model -> wi)."""
    return VARIANTS[corpus][0] if corpus in VARIANTS else corpus


def data_dir(corpus: str):
    """data/study3/<corpus id>/, created if missing."""
    path = DATA / corpus
    path.mkdir(parents=True, exist_ok=True)
    return path


def results_dir(corpus: str):
    """results/study3/<corpus id>/ (validation/<id>/ for the corrector check), created if missing."""
    path = RESULTS / "validation" / corpus if corpus in VALIDATION_IDS else RESULTS / corpus
    path.mkdir(parents=True, exist_ok=True)
    return path


def tau() -> float:
    return json.loads((RESULTS / "threshold.json").read_text())["tau_bits"]


def fce_band(score: float):
    for lo, hi in FCE_BANDS:
        if lo <= score <= hi:
            return f"{lo}-{hi}"
    return None


def ellipse_band(score: float) -> str:
    return next(f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}" for lo, hi in ELLIPSE_BANDS if lo <= score <= hi)


def bands(corpus: str) -> list[str]:
    """The level groups of rate_cost, lowest first."""
    base = base_corpus(corpus)
    if base == "wi":
        return ["A", "B", "C"]
    if base == "fce":
        return [f"{lo}-{hi}" for lo, hi in FCE_BANDS]
    return [f"{lo:g}" if lo == hi else f"{lo:g}-{hi:g}" for lo, hi in ELLIPSE_BANDS]
