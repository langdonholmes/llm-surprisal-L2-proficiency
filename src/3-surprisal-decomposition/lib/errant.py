"""ERRANT types for the json gold edits, and the groupings used in the analysis.

The BEA-2019 M2 files were made from the json files by ``json_to_m2.py``:
punctuation normalised one character for one, paragraphs split on newlines,
whitespace collapsed, spaCy-tokenised, and the character edits converted to
token edits and typed by ERRANT (Bryant et al. 2017). Here the M2 tokens are
walked back onto the original characters (each token is matched at the next
non-space character of its paragraph), every M2 edit gets a character span,
and each first-annotator json edit takes the type of the M2 edit it overlaps:

  * substitution/deletion (s < e): M2 edits whose span overlaps [s, e), or M2
    insertions whose point lies in [s, e];
  * insertion (s == e): an M2 insertion at the point, else an M2 span
    containing the next non-space character, else one ending at the last
    non-space character before s.

The first match gives the type. A substitution that changes only whitespace
has no M2 edit and is typed R:ORTH. Edits with no match are "unmapped".

Two groupings of the ERRANT category (operation prefix stripped):
  GROUP   nine error groups (D28)
  BUCKET  four kinds of correction (D33): mechanical, function word,
          content-word choice, multi-word / order; UNK and unmapped edits are
          "unclassified"
"""

import json
import re

import pandas as pd

from lib.corpora import RAW_DIRS

NORM = {"’": "'", "´": "'", "‘": "'", "′": "'", "`": "'", '“': '"', '”': '"', '˝': '"',
        '¨': '"', '„': '"', '『': '"', '』': '"', '–': '-', '—': '-', '―': '-', '¬': '-',
        '、': ',', '，': ',', '：': ':', '；': ';', '？': '?', '！': '!', 'ِ': ' ', '​': ' '}
CHAR_NORM = {ord(k): v for k, v in NORM.items()}

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
}
GROUPS = ["spelling/orth", "verb form/SVA/tense", "determiner", "preposition", "punctuation",
          "noun number", "word order", "other lexical", "other"]

BUCKET = {}
for _cat in ["SPELL", "ORTH", "PUNCT", "VERB:FORM", "VERB:SVA", "VERB:INFL", "VERB:TENSE", "NOUN:NUM",
           "NOUN:INFL", "MORPH", "ADJ:FORM", "CONTR", "NOUN:POSS"]:
    BUCKET[_cat] = "mechanical"
for _cat in ["DET", "PREP", "PRON", "CONJ", "PART"]:
    BUCKET[_cat] = "function word"
for _cat in ["NOUN", "VERB", "ADJ", "ADV"]:
    BUCKET[_cat] = "content-word choice"
for _cat in ["OTHER", "WO"]:
    BUCKET[_cat] = "multi-word / order"
BUCKETS = ["mechanical", "function word", "content-word choice", "multi-word / order"]
UNCLASSIFIED = "unclassified"
OPS = {"M": "missing", "R": "replacement", "U": "unnecessary"}

FILES = {"wi": [(f"json/{p}.json", f"m2/{p}.gold.bea19.m2") for p in
                ["A.train", "A.dev", "B.train", "B.dev", "C.train", "C.dev", "N.dev"]],
         "fce": [(f"json/fce.{p}.json", f"m2/fce.{p}.gold.bea19.m2") for p in ["train", "dev", "test"]]}


def category(errant_type: str) -> str:
    """The ERRANT type without its operation prefix (R:VERB:SVA -> VERB:SVA)."""
    t = str(errant_type)
    return t[2:] if t[:2] in ("M:", "R:", "U:") else t


def group_of(errant_type: str) -> str:
    return GROUP.get(category(errant_type), "other")


def bucket_of(errant_type: str) -> str:
    return BUCKET.get(category(errant_type), UNCLASSIFIED)


def op_of(errant_type: str, s: int, e: int, repl: str) -> str:
    """missing / replacement / unnecessary, from the type's prefix, else from the edit."""
    t = str(errant_type)
    if t[1:2] == ":" and t[0] in OPS:
        return OPS[t[0]]
    return "missing" if s == e else ("unnecessary" if repl == "" else "replacement")


def read_m2(path):
    """Yield (tokens, edits) per sentence; edits are (s, e, type) for coder 0."""
    for block in path.read_text().strip("\n").split("\n\n"):
        lines = block.split("\n")
        tokens = lines[0][2:].split(" ") if len(lines[0]) > 2 else []
        edits = []
        for line in lines[1:]:
            fields = line[2:].split("|||")  # "s e", type, correction, required, comment, coder
            s, e = map(int, fields[0].split())
            if s < 0 or fields[1] == "noop" or fields[-1] != "0":
                continue
            edits.append((s, e, fields[1]))
        yield tokens, edits


def essay_m2_spans(text: str, sents):
    """Consume M2 sentences for one essay; return char-span edits and an ok flag."""
    norm = re.sub(r"\s", " ", text.translate(CHAR_NORM))  # same length as text, every whitespace a space
    spans, ok = [], True
    for para in re.finditer(r"[^\n]+", text):
        para_start, para_end = para.start(), para.end()
        if not norm[para_start:para_end].strip():
            continue
        pos = para_start
        while pos < para_end and norm[pos] == " ":
            pos += 1
        # each M2 sentence of the paragraph: match its tokens in order, then place its edits
        while pos < para_end:
            try:
                tokens, edits = next(sents)
            except StopIteration:
                return spans, False
            starts, ends = [], []
            for token in tokens:
                while pos < para_end and norm[pos] == " ":
                    pos += 1
                if norm[pos:pos + len(token)] != token:
                    ok = False
                    starts.append(pos)
                    ends.append(pos)
                    continue
                starts.append(pos)
                ends.append(pos + len(token))
                pos += len(token)
            while pos < para_end and norm[pos] == " ":
                pos += 1
            for s, e, etype in edits:
                if s == e:
                    point = starts[s] if s < len(starts) else ends[-1]
                    spans.append((point, point, etype))
                else:
                    spans.append((starts[s], ends[e - 1], etype))
    return spans, ok


def match(edit, m2, text):
    """The types of the M2 edits that a json edit overlaps, in the order of the module docstring."""
    s, e = edit[0], edit[1]
    hits = []
    if s < e:
        for m2_s, m2_e, etype in m2:
            if (m2_s < m2_e and m2_s < e and m2_e > s) or (m2_s == m2_e and s <= m2_s <= e):
                hits.append(etype)
    else:
        next_char = s  # the next non-space character after the insertion point
        while next_char < len(text) and text[next_char].isspace():
            next_char += 1
        prev_end = s  # just after the last non-space character before it
        while prev_end > 0 and text[prev_end - 1].isspace():
            prev_end -= 1
        tiers = ([etype for m2_s, m2_e, etype in m2 if m2_s == m2_e and m2_s in (next_char, s)],
                 [etype for m2_s, m2_e, etype in m2 if m2_s < m2_e and m2_s <= next_char < m2_e],
                 [etype for m2_s, m2_e, etype in m2 if m2_s < m2_e and m2_e == prev_end])
        hits = [etype for tier in tiers for etype in tier]
    if not hits and s < e and re.sub(r"\s", "", text[s:e]) == re.sub(r"\s", "", edit[2]):
        hits = ["R:ORTH"]  # whitespace-only edit: ERRANT drops it after tokenisation
    return hits


def edit_types(corpus: str) -> pd.DataFrame:
    """One row per first-annotator json edit: text_id, k, s, e, errant, group, bucket, op.

    Flagged (null-replacement) edits get "flagged" in every column.
    """
    rows, n_ok, n_texts = [], 0, 0
    for json_name, m2_name in FILES[corpus]:
        sents = read_m2(RAW_DIRS[corpus] / m2_name)
        for line in open(RAW_DIRS[corpus] / json_name):
            record = json.loads(line)
            text = record["text"]
            m2, ok = essay_m2_spans(text, sents)
            n_ok += ok
            n_texts += 1
            text_id = record["id"] if corpus == "wi" else f"{record['id']}_{record['q']}"
            edits = [x for coder, x in record["edits"] if coder == 0][0]
            for k, edit in enumerate(edits):
                s, e, repl = edit[0], edit[1], edit[2]
                if repl is None:
                    rows.append((text_id, k, s, e, "flagged", "flagged", "flagged", "flagged"))
                    continue
                hits = match(edit, m2, text) if ok else []
                etype = hits[0] if hits else "unmapped"
                rows.append((text_id, k, s, e, etype, group_of(etype), bucket_of(etype), op_of(etype, s, e, repl)))
        left = sum(1 for _ in sents)
        if left:
            print(f"{m2_name}: {left} M2 sentences left over", flush=True)
    print(f"{corpus}: {n_ok}/{n_texts} texts walked onto M2 without a token mismatch", flush=True)
    return pd.DataFrame(rows, columns=["text_id", "k", "s", "e", "errant", "group", "bucket", "op"])
