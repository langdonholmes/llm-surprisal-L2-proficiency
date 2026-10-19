"""Pilot step 1 (CPU) — an ERRANT type for every W&I+LOCNESS json edit.

The BEA-2019 M2 files were made from the json files by ``json_to_m2.py``:
punctuation normalised one character for one, paragraphs split on newlines,
whitespace collapsed, spaCy-tokenised, and the character edits converted to
token edits and minimised by ERRANT. Here the M2 tokens are walked back onto
the original characters (each token is matched at the next non-space
character of its paragraph), every M2 edit gets a character span, and each
first-annotator json edit takes the type of the M2 edit(s) it overlaps.

Matching rule, per json edit [s, e, r] with r not null:
  * substitution/deletion (s < e): M2 edits whose character span overlaps
    [s, e), or M2 insertions whose point lies in [s, e];
  * insertion (s == e): M2 edits whose point (insertions) or span (others)
    contains the first non-space character at or after s, or that end at the
    last non-space character before s.
For insertions the tiers are tried in that order (an M2 insertion at the
point, then a span containing it, then a span ending just before it). The first
match gives the type; ``n_m2`` counts all matches. A substitution that changes
only whitespace has no M2 edit (ERRANT compares tokens) and is typed R:ORTH,
which is ERRANT's category for whitespace and case.
Null-replacement (detection-only) edits get type "flagged".

Output: data/study3/pilot_shapley/edit_types.parquet
    text_id, k (index in the json edit list), s, e, repl, kind, errant, group, n_m2
and tables/errant_coverage.csv.

    python src/3-surprisal-decomposition/pilot_shapley/1_errant.py
"""

import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, TAB, WI_RAW, group_of  # noqa: E402

NORM = {"’": "'", "´": "'", "‘": "'", "′": "'", "`": "'", '“': '"', '”': '"', '˝': '"',
        '¨': '"', '„': '"', '『': '"', '』': '"', '–': '-', '—': '-', '―': '-', '¬': '-',
        '、': ',', '，': ',', '：': ':', '；': ';', '？': '?', '！': '!', 'ِ': ' ', '​': ' '}
NORM = {ord(k): v for k, v in NORM.items()}
FILES = [("A.train", "A.train"), ("A.dev", "A.dev"), ("B.train", "B.train"), ("B.dev", "B.dev"),
         ("C.train", "C.train"), ("C.dev", "C.dev"), ("N.dev", "N.dev")]


def read_m2(path: Path):
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
    """Consume M2 sentences for one essay; return char-span edits and ok flag."""
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


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    rows, essay_ok = [], []
    for jname, mname in FILES:
        sents = read_m2(WI_RAW / "m2" / f"{mname}.gold.bea19.m2")
        for line in open(WI_RAW / "json" / f"{jname}.json"):
            d = json.loads(line)
            text = d["text"]
            m2, ok = essay_m2_spans(text, sents)
            essay_ok.append(ok)
            edits = [x for c, x in d["edits"] if c == 0][0]
            for k, ed in enumerate(edits):
                s, e, r = ed[0], ed[1], ed[2]
                if r is None:
                    typ, n = "flagged", 0
                else:
                    h = match(ed, m2, text) if ok else []
                    typ, n = (h[0] if h else "unmapped"), len(h)
                kind = "flagged" if r is None else ("ins" if s == e else ("del" if r == "" else "sub"))
                rows.append({"text_id": d["id"], "file": jname, "k": k, "s": s, "e": e, "repl": r,
                             "kind": kind, "errant": typ, "n_m2": n})
        rest = sum(1 for _ in sents)
        print(f"{jname}: M2 sentences left over {rest}", flush=True)
    df = pd.DataFrame(rows)
    df["group"] = [("flagged" if t == "flagged" else group_of(t)) for t in df.errant]
    df.to_parquet(OUT / "edit_types.parquet", index=False)
    c = df[df.kind != "flagged"]
    cov = pd.DataFrame({
        "n_edits": c.groupby("kind").size(),
        "mapped": c.groupby("kind").apply(lambda x: (x.errant != "unmapped").mean()),
        "multi_m2": c.groupby("kind").apply(lambda x: (x.n_m2 > 1).mean())})
    cov.loc["all"] = [len(c), (c.errant != "unmapped").mean(), (c.n_m2 > 1).mean()]
    cov.to_csv(TAB / "errant_coverage.csv")
    print(f"essays aligned to M2 without a token mismatch: {sum(essay_ok)}/{len(essay_ok)}")
    print(cov.round(4).to_string())
    print(c.group.value_counts(normalize=True).round(4).to_string())


if __name__ == "__main__":
    main()
