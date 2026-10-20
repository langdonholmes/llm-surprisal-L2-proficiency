"""Applying the gold corrections, and the gold category of each original token.

The gold edits (first annotator, ``src/essays/ingest.py --wi / --fce``) are
[start, end, replacement] character triples into the original text. A null
replacement marks a span the annotator flagged without correcting. It is left
as written.
"""

import numpy as np

CLOSE_PUNCT = set(".,;:!?)")


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
    return {"corrected": "".join(out), "cmap": cmap, "spans": spans, "flagged": flagged, "fixes": fixes}


def gold_for_doc(doc, edits) -> dict:
    """Gold category per spaCy token index.

    ERR_TOK  the token overlaps an edited or flagged span
    ERR_INS  an insertion point lies between the previous token and this one,
             i.e. a word is missing just before it
    POST     the previous token is ERR_TOK
    OK       none of the above (uncorrected)
    """
    spans = [(s, e) for s, e, _ in edits if e > s]
    ins = [s for s, e, _ in edits if e == s]
    cat = {}
    prev_end = 0
    err_prev = False
    for t in doc:
        a, b = t.idx, t.idx + len(t.text)
        if any(s < b and e > a for s, e in spans):
            c = "ERR_TOK"
        elif any(prev_end <= p <= a for p in ins):
            c = "ERR_INS"
        elif err_prev:
            c = "POST"
        else:
            c = "OK"
        cat[t.i] = c
        if not t.is_space:
            err_prev = c == "ERR_TOK"
            prev_end = b
    return cat
