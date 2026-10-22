"""Applying the gold corrections, and the gold category of each original token.

The gold edits (first annotator, ``src/essays/ingest.py --wi / --fce``) are
[start, end, replacement] character triples into the original text. A null
replacement marks a span the annotator flagged without correcting. It is left
as written.
"""

import numpy as np

CLOSE_PUNCT = set(".,;:!?)")


def apply_edits(text: str, edits: list, repair: bool = True) -> dict:
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
    the text is touched, so unedited characters map one to one. ERRANT's edits
    from ``corrector/c5_edits.py`` already carry their whitespace, so they are
    applied with ``repair=False``.
    """
    cmap = np.full(len(text), -1, dtype=np.int64)
    out: list[str] = []          # one character per entry
    source: list[int] = []       # original index of each output char, -1 if inserted
    spans, flagged = [], []
    fixes = {"del_swallow_space": 0, "del_space_before_punct": 0, "strip_lead": 0,
             "strip_trail": 0, "ins_add_space": 0, "skipped_overlap": 0}
    pos = 0  # next original character to copy
    ordered = sorted(edits, key=lambda x: (x[0], x[1]))
    starts = {x[0] for x in ordered if x[2] is not None}
    for s, e, repl in ordered:
        if repl is None:
            flagged.append((s, e))
            continue
        if s < pos:
            fixes["skipped_overlap"] += 1
            continue
        # copy the unedited text up to the edit
        for i in range(pos, s):
            out.append(text[i])
            source.append(i)
        left = out[-1] if out else "\n"
        right = text[e] if e < len(text) else "\n"
        applied = repl
        # whitespace repairs at the two junctions (see the docstring)
        if repair and s == e and applied:
            if left.isalnum() and applied[0].isalnum():
                applied = " " + applied
                fixes["ins_add_space"] += 1
            if applied[-1].isalnum() and right.isalnum():
                applied = applied + " "
                fixes["ins_add_space"] += 1
        if repair and applied and applied[0] == " " and left in " \n":
            applied = applied[1:]
            fixes["strip_lead"] += 1
        if repair and applied and applied[-1] == " " and right in " \n":
            applied = applied[:-1]
            fixes["strip_trail"] += 1
        end = e
        if repair and applied == "":
            if left in " \n" and right == " " and e not in starts:
                end = e + 1  # "a  b" -> "a b"
                fixes["del_swallow_space"] += 1
            elif left == " " and right in CLOSE_PUNCT and out:
                out.pop()   # "the cat ." -> "the cat."
                source.pop()
                fixes["del_space_before_punct"] += 1
        kind = "ins" if s == e else ("del" if repl == "" else "sub")
        spans.append({"s": s, "e": end, "cs": len(out), "ce": len(out) + len(applied), "kind": kind,
                      "orig": text[s:e], "repl": applied})
        for ch in applied:
            out.append(ch)
            source.append(-1)
        pos = end
    for i in range(pos, len(text)):
        out.append(text[i])
        source.append(i)
    # invert source (output -> original) into cmap (original -> output)
    source_arr = np.array(source, dtype=np.int64)
    kept = source_arr >= 0
    cmap[source_arr[kept]] = np.flatnonzero(kept)
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
    insertions = [s for s, e, _ in edits if e == s]
    category = {}
    prev_end = 0
    prev_is_error = False
    for token in doc:
        start, end = token.idx, token.idx + len(token.text)
        if any(s < end and e > start for s, e in spans):
            cat = "ERR_TOK"
        elif any(prev_end <= point <= start for point in insertions):
            cat = "ERR_INS"
        elif prev_is_error:
            cat = "POST"
        else:
            cat = "OK"
        category[token.i] = cat
        if not token.is_space:
            prev_is_error = cat == "ERR_TOK"
            prev_end = end
    return category
