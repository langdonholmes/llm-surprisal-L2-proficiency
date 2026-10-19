"""Shared pieces of the Shapley-over-context pilot: paths, error groups, players.

Players are the spaCy tokens that own the 7 context subwords of a target's
8-subword window (each subword belongs to the spaCy token in which its first
non-space character lies; a whitespace-only subword therefore joins the token
that follows it). See NOTES.md for the design.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "pilot_corrected"))

from util.paths import DATA_DIR  # noqa: E402

OUT = DATA_DIR / "study3" / "pilot_shapley"
PC = DATA_DIR / "study3" / "pilot_corrected"
WI_RAW = DATA_DIR / "wi_locness" / "raw" / "wi+locness"
TAB = OUT / "tables"
LEVELS = ["A", "B", "C", "N"]
SEED = 42
WIN = 8

# ERRANT category (operation prefix stripped) -> analysis group
GROUP = {
    "SPELL": "spelling/orth", "ORTH": "spelling/orth",
    "VERB:FORM": "verb form/SVA/tense", "VERB:SVA": "verb form/SVA/tense",
    "VERB:TENSE": "verb form/SVA/tense", "VERB:INFL": "verb form/SVA/tense",
    "DET": "determiner",
    "PREP": "preposition",
    "PUNCT": "punctuation",
    "NOUN:NUM": "noun number", "NOUN:INFL": "noun number",
    "WO": "word order",
    "NOUN": "other lexical", "VERB": "other lexical", "ADJ": "other lexical",
    "ADV": "other lexical", "ADJ:FORM": "other lexical", "MORPH": "other lexical",
    "PRON": "other", "CONJ": "other", "PART": "other", "CONTR": "other",
    "NOUN:POSS": "other", "OTHER": "other", "UNK": "other",
}
GROUPS = ["spelling/orth", "verb form/SVA/tense", "determiner", "preposition", "punctuation",
          "noun number", "word order", "other lexical", "other"]


def group_of(errant_type: str) -> str:
    if errant_type in (None, "", "unmapped"):
        return "unmapped"
    cat = errant_type.split(":", 1)[1] if errant_type[:2] in ("M:", "R:", "U:") else errant_type
    return GROUP.get(cat, "other")
