"""Paths, corpora and analysis units shared by the stages.

Two gold-corrected corpora. W&I+LOCNESS: the unit is the essay and the
criterion is CEFR level (A = 1, B = 2, C = 3; LOCNESS, level N, is described
but has no criterion). FCE: the unit is the script (both answers pooled) and
the criterion is the script score (1-40; the one script scored 0 is left out).
"""

import json

from util.paths import DATA_DIR, FCE_DIR, FCE_DOCBINS_DIR, RESULTS_DIR, WI_DIR, WI_DOCBINS_DIR

S3 = DATA_DIR / "study3"
DATA = S3
RES = RESULTS_DIR / "study3"
# The ELLIPSE token surprisal stored by the appendix's GPU pass (appendix/A0_extract.py)
ELLIPSE_APPX = S3 / "appendix" / "ellipse"
CORPORA = ["wi", "fce"]
DOCBINS = {"wi": WI_DOCBINS_DIR, "fce": FCE_DOCBINS_DIR}
RAW = {"wi": WI_DIR / "raw" / "wi+locness", "fce": FCE_DIR / "raw" / "fce"}
LEVEL_Y = {"A": 1.0, "B": 2.0, "C": 3.0}
FCE_BANDS = [(1, 23), (24, 27), (28, 31), (32, 40)]
SEED = 42
N_BOOT = 2000

# RQ3 feature sets (columns of units_<corpus>.parquet, written by 5_decompose.py)
GROUP_SLUGS = ["spelling_orth", "verb_form_SVA_tense", "determiner", "preposition", "punctuation",
               "noun_number", "word_order", "other_lexical", "other"]
B = [f"rate_{g}" for g in GROUP_SLUGS] + ["rate_flagged", "log_n"]
S_COMPACT = ["clean", "cost_mean", "spill_mean"]


def out(corpus: str):
    d = DATA / corpus
    d.mkdir(parents=True, exist_ok=True)
    return d


def tau() -> float:
    return json.loads((RES / "threshold.json").read_text())["tau_bits"]


def fce_band(score: float):
    for lo, hi in FCE_BANDS:
        if lo <= score <= hi:
            return f"{lo}-{hi}"
    return None
