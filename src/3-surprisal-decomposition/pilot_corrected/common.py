"""Shared pieces of the corrected-essay pilot: paths and applying the gold edits.

The W&I+LOCNESS gold edits (first annotator) are [start, end, replacement]
character triples into the original text (``src/essays/ingest.py``). A null
replacement marks a span the annotator flagged without correcting; it is left
as written and recorded as ``flagged``.
"""

import json
from pathlib import Path

import numpy as np

from util.paths import DATA_DIR, RESULTS_DIR

HERE = Path(__file__).resolve().parent
OUT = DATA_DIR / "study3" / "pilot_corrected"
LEVELS = ["A", "B", "C", "N"]
SEED = 42
CLOSE_PUNCT = set(".,;:!?)")


def tau() -> float:
    return json.loads((RESULTS_DIR / "study3" / "threshold.json").read_text())["tau_bits"]


def apply_edits(text: str, edits: list) -> dict:
    """Apply the gold edits to ``text``.

    Returns
      corrected   the corrected text
      cmap        int array, len(text): position of each original character in
                  the corrected text, or -1 if the edit removed or replaced it
      spans       one dict per applied edit: orig [s, e), corrected [cs, ce),
                  kind (sub / ins / del), replacement as applied
      flagged     original [s, e) spans with a null replacement (left as written)
      fixes       counts of the whitespace repairs at edit junctions

    Whitespace at the junctions is repaired locally, because the offsets do not
    always include the separating space: a deletion between two spaces swallows
    one of them (and a space left before closing punctuation), a replacement's
    leading/trailing space is dropped next to existing whitespace, and an
    insertion that would glue two words together gets a space. Nothing else in
    the text is touched, so unedited characters map one to one.
    """
    cmap = np.full(len(text), -1, dtype=np.int64)
    out: list[str] = []          # one character per entry
    src: list[int] = []          # original index of each output char, -1 if inserted
    spans, flagged = [], []
    fixes = {"del_swallow_space": 0, "del_space_before_punct": 0, "strip_lead": 0,
             "strip_trail": 0, "ins_add_space": 0, "skipped_overlap": 0}
    p = 0
    ordered = sorted(edits, key=lambda x: (x[0], x[1]))
    starts = {x[0] for x in ordered if x[2] is not None}
    for s, e, r in ordered:
        if r is None:
            flagged.append((s, e))
            continue
        if s < p:
            fixes["skipped_overlap"] += 1
            continue
        for i in range(p, s):
            out.append(text[i])
            src.append(i)
        left = out[-1] if out else "\n"
        right = text[e] if e < len(text) else "\n"
        r2 = r
        if s == e and r2:
            if left.isalnum() and r2[0].isalnum():
                r2 = " " + r2
                fixes["ins_add_space"] += 1
            if r2[-1].isalnum() and right.isalnum():
                r2 = r2 + " "
                fixes["ins_add_space"] += 1
        if r2 and r2[0] == " " and left in " \n":
            r2 = r2[1:]
            fixes["strip_lead"] += 1
        if r2 and r2[-1] == " " and right in " \n":
            r2 = r2[:-1]
            fixes["strip_trail"] += 1
        e2 = e
        if r2 == "":
            if left in " \n" and right == " " and e not in starts:
                e2 = e + 1  # "a  b" -> "a b"
                fixes["del_swallow_space"] += 1
            elif left == " " and right in CLOSE_PUNCT and out:
                out.pop()   # "the cat ." -> "the cat."
                src.pop()
                fixes["del_space_before_punct"] += 1
        kind = "ins" if s == e else ("del" if r == "" else "sub")
        spans.append({"s": s, "e": e2, "cs": len(out), "ce": len(out) + len(r2), "kind": kind,
                      "orig": text[s:e], "repl": r2})
        for ch in r2:
            out.append(ch)
            src.append(-1)
        p = e2
    for i in range(p, len(text)):
        out.append(text[i])
        src.append(i)
    src_a = np.array(src, dtype=np.int64)
    keep = src_a >= 0
    cmap[src_a[keep]] = np.flatnonzero(keep)
    corrected = "".join(out)
    return {"corrected": corrected, "cmap": cmap, "spans": spans, "flagged": flagged, "fixes": fixes}
