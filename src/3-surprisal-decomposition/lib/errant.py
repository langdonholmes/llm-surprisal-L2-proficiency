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

from lib.corpora import RAW

NORM = {"’": "'", "´": "'", "‘": "'", "′": "'", "`": "'", '“': '"', '”': '"', '˝': '"',
        '¨': '"', '„': '"', '『': '"', '』': '"', '–': '-', '—': '-', '―': '-', '¬': '-',
        '、': ',', '，': ',', '：': ':', '；': ';', '？': '?', '！': '!', 'ِ': ' ', '​': ' '}
NORM = {ord(k): v for k, v in NORM.items()}

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
for _c in ["SPELL", "ORTH", "PUNCT", "VERB:FORM", "VERB:SVA", "VERB:INFL", "VERB:TENSE", "NOUN:NUM",
           "NOUN:INFL", "MORPH", "ADJ:FORM", "CONTR", "NOUN:POSS"]:
    BUCKET[_c] = "mechanical"
for _c in ["DET", "PREP", "PRON", "CONJ", "PART"]:
    BUCKET[_c] = "function word"
for _c in ["NOUN", "VERB", "ADJ", "ADV"]:
    BUCKET[_c] = "content-word choice"
for _c in ["OTHER", "WO"]:
    BUCKET[_c] = "multi-word / order"
BUCKETS = ["mechanical", "function word", "content-word choice", "multi-word / order"]
UNCL = "unclassified"
OPS = {"M": "missing", "R": "replacement", "U": "unnecessary"}

FILES = {"wi": [(f"json/{p}.json", f"m2/{p}.gold.bea19.m2") for p in
                ["A.train", "A.dev", "B.train", "B.dev", "C.train", "C.dev", "N.dev"]],
         "fce": [(f"json/fce.{p}.json", f"m2/fce.{p}.gold.bea19.m2") for p in ["train", "dev", "test"]]}


def category(errant_type: str) -> str:
    t = str(errant_type)
    return t[2:] if t[:2] in ("M:", "R:", "U:") else t


def group_of(errant_type: str) -> str:
    return GROUP.get(category(errant_type), "other")


def bucket_of(errant_type: str) -> str:
    return BUCKET.get(category(errant_type), UNCL)


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
        toks = lines[0][2:].split(" ") if len(lines[0]) > 2 else []
        eds = []
        for ln in lines[1:]:
            f = ln[2:].split("|||")
            s, e = map(int, f[0].split())
            if s < 0 or f[1] == "noop" or f[-1] != "0":
                continue
            eds.append((s, e, f[1]))
        yield toks, eds


def essay_m2_spans(text: str, sents):
    """Consume M2 sentences for one essay; return char-span edits and an ok flag."""
    t = re.sub(r"\s", " ", text.translate(NORM))
    out, ok = [], True
    for para in re.finditer(r"[^\n]+", text):
        a, b = para.start(), para.end()
        if not t[a:b].strip():
            continue
        p = a
        while p < b and t[p] == " ":
            p += 1
        while p < b:
            try:
                toks, eds = next(sents)
            except StopIteration:
                return out, False
            starts, ends = [], []
            for tk in toks:
                while p < b and t[p] == " ":
                    p += 1
                if t[p:p + len(tk)] != tk:
                    ok = False
                    starts.append(p)
                    ends.append(p)
                    continue
                starts.append(p)
                ends.append(p + len(tk))
                p += len(tk)
            while p < b and t[p] == " ":
                p += 1
            for s, e, typ in eds:
                if s == e:
                    c = starts[s] if s < len(starts) else ends[-1]
                    out.append((c, c, typ))
                else:
                    out.append((starts[s], ends[e - 1], typ))
    return out, ok


def match(edit, m2, text):
    s, e = edit[0], edit[1]
    hits = []
    if s < e:
        for ms, me, typ in m2:
            if (ms < me and ms < e and me > s) or (ms == me and s <= ms <= e):
                hits.append(typ)
    else:
        n = s
        while n < len(text) and text[n].isspace():
            n += 1
        q = s
        while q > 0 and text[q - 1].isspace():
            q -= 1
        tiers = ([t for ms, me, t in m2 if ms == me and ms in (n, s)],
                 [t for ms, me, t in m2 if ms < me and ms <= n < me],
                 [t for ms, me, t in m2 if ms < me and me == q])
        hits = [t for tier in tiers for t in tier]
    if not hits and s < e and re.sub(r"\s", "", text[s:e]) == re.sub(r"\s", "", edit[2]):
        hits = ["R:ORTH"]  # whitespace-only edit: ERRANT drops it after tokenisation
    return hits


def edit_types(corpus: str) -> pd.DataFrame:
    """One row per first-annotator json edit: text_id, k, s, e, errant, group, bucket, op.

    Flagged (null-replacement) edits get "flagged" in every column.
    """
    rows, n_ok, n = [], 0, 0
    for jname, mname in FILES[corpus]:
        sents = read_m2(RAW[corpus] / mname)
        for line in open(RAW[corpus] / jname):
            d = json.loads(line)
            text = d["text"]
            m2, ok = essay_m2_spans(text, sents)
            n_ok += ok
            n += 1
            tid = d["id"] if corpus == "wi" else f"{d['id']}_{d['q']}"
            edits = [x for c, x in d["edits"] if c == 0][0]
            for k, ed in enumerate(edits):
                s, e, r = ed[0], ed[1], ed[2]
                if r is None:
                    rows.append((tid, k, s, e, "flagged", "flagged", "flagged", "flagged"))
                    continue
                h = match(ed, m2, text) if ok else []
                typ = h[0] if h else "unmapped"
                rows.append((tid, k, s, e, typ, group_of(typ), bucket_of(typ), op_of(typ, s, e, r)))
        left = sum(1 for _ in sents)
        if left:
            print(f"{mname}: {left} M2 sentences left over", flush=True)
    print(f"{corpus}: {n_ok}/{n} texts walked onto M2 without a token mismatch", flush=True)
    return pd.DataFrame(rows, columns=["text_id", "k", "s", "e", "errant", "group", "bucket", "op"])
