"""Lab, step 1 — build the synthetic error benchmark (CPU).

Ground truth for comparing token representations. Each instance is one dev
ELLIPSE essay with **one** controlled edit, plus the position where the edit
becomes observable to a left-to-right model ("target"):

    DET_DEL   delete a/an/the                         target = the next word
    DET_SUB   a/an <-> the                            target = the determiner
    AGR       subject-verb agreement flipped (is/are, has/have, walk/walks)
    VFORM     verb form: past -> base, base after to/modal -> -ing, VBN -> base
    NUM       noun number flipped (singular <-> plural, regular morphology)
    PREP      preposition swapped within {in on at for to of with about from}
    SPELL     one-character typo in a content word of 5+ letters
    HOMO      homophone/near-homophone swap (their/there, its/it's, then/than ...)
    CLEAN_SURP an unedited token with surprisal >= tau (surprising but not injected)
    MATCHED   for every error instance, an unedited occurrence of the *same
              observed word form* in another host essay (``matched_to`` = the
              error instance, ``matched_type`` = its type). Observed-word
              identity cannot separate an error from its matched control, so
              error-vs-matched AUC is .5 for a purely lexical representation.

Host essays are dev essays with Overall >= 4.0 (CLEAN_SURP: >= 4.5, to keep
natural errors rare). These essays still contain some natural errors, so the
CLEAN_SURP class is noisy, which is noted in every report.

``group`` is the lexeme used for lexeme-disjoint cross-validation: the
lowercased lemma of the *original* word at the target position (so a typo of
"because" is grouped with "because").

    -> data/study3/lab/bench.parquet
"""

import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "1-predictability-benchmark"))
from run_surprisal import load_essays  # noqa: E402

from util.paths import DATA_DIR  # noqa: E402

HERE = Path(__file__).resolve().parent
S3 = DATA_DIR / "study3"
LAB = S3 / "lab"
# LAB_BENCH=conf builds the held-out confirmation benchmark: other host essays
# (dev, Overall == 3.5, natural errors more common), another seed.
BENCH = os.environ.get("LAB_BENCH", "")
SEED = 43 if BENCH == "conf" else 42
N_PER_TYPE = 1200
MAX_PER_ESSAY = 3
PREPS = ["in", "on", "at", "for", "to", "of", "with", "about", "from"]
HOMO = {"their": "there", "there": "their", "they're": "their", "your": "you're",
        "you're": "your", "its": "it's", "it's": "its", "then": "than", "than": "then",
        "to": "too", "too": "to", "were": "where", "where": "were", "know": "no",
        "no": "know", "whether": "weather", "lose": "loose", "affect": "effect",
        "effect": "affect", "accept": "except", "quite": "quiet", "though": "through"}
AGR_IRREG = {"is": "are", "are": "is", "was": "were", "were": "was", "has": "have",
             "have": "has", "does": "do", "do": "does"}


def match_case(new: str, old: str) -> str:
    return new[:1].upper() + new[1:] if old[:1].isupper() else new


def add_s(w: str) -> str:
    if w.endswith(("s", "x", "z", "ch", "sh")):
        return w + "es"
    if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
        return w[:-1] + "ies"
    return w + "s"


def add_ing(w: str) -> str:
    if w.endswith("ie"):
        return w[:-2] + "ying"
    if w.endswith("e") and not w.endswith("ee") and len(w) > 2:
        return w[:-1] + "ing"
    return w + "ing"


def typo(w: str, rng: random.Random) -> str:
    i = rng.randrange(1, len(w) - 1)
    op = rng.choice(["swap", "drop", "double", "vowel"])
    if op == "swap":
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op == "drop":
        return w[:i] + w[i + 1:]
    if op == "double":
        return w[:i] + w[i] + w[i:]
    vow = [j for j, c in enumerate(w) if c in "aeiou" and 0 < j]
    if not vow:
        return w[:i] + w[i + 1:]
    j = rng.choice(vow)
    return w[:j] + rng.choice([v for v in "aeiou" if v != w[j]]) + w[j + 1:]


def candidates(doc, rng: random.Random):
    """Yield (type, token_index, replacement or None for deletion)."""
    for t in doc:
        nxt = doc[t.i + 1] if t.i + 1 < len(doc) else None
        low = t.lower_
        if t.tag_ == "DT" and low in ("a", "an", "the") and nxt is not None and nxt.is_alpha:
            yield "DET_DEL", t.i, None
            yield "DET_SUB", t.i, "the" if low in ("a", "an") else "a"
        if t.pos_ in ("VERB", "AUX") and any(c.dep_ in ("nsubj", "nsubjpass") for c in t.children):
            if low in AGR_IRREG:
                yield "AGR", t.i, AGR_IRREG[low]
            elif t.tag_ == "VBZ" and t.pos_ == "VERB":
                yield "AGR", t.i, t.lemma_.lower()
            elif t.tag_ == "VBP" and t.pos_ == "VERB" and t.lemma_.lower() == low:
                yield "AGR", t.i, add_s(low)
        if t.pos_ == "VERB":
            lem = t.lemma_.lower()
            if t.tag_ == "VBD" and lem != low:
                yield "VFORM", t.i, lem
            elif t.tag_ == "VB" and t.i > 0 and doc[t.i - 1].lower_ in ("to", "can", "will", "would", "should", "could", "must"):
                yield "VFORM", t.i, add_ing(lem)
            elif t.tag_ == "VBN" and lem != low and t.i > 0 and doc[t.i - 1].lemma_ == "have":
                yield "VFORM", t.i, lem
        if t.pos_ == "NOUN" and t.is_alpha:
            lem = t.lemma_.lower()
            if t.tag_ == "NNS" and lem != low and add_s(lem) == low:
                yield "NUM", t.i, lem
            elif t.tag_ == "NN" and lem == low and len(low) > 2:
                yield "NUM", t.i, add_s(low)
        if t.pos_ == "ADP" and low in PREPS and t.dep_ == "prep":
            yield "PREP", t.i, rng.choice([p for p in PREPS if p != low])
        if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and t.is_alpha and len(t) >= 5:
            yield "SPELL", t.i, typo(t.text, rng)
        if low in HOMO:
            yield "HOMO", t.i, HOMO[low]


def make_instance(doc, typ, i, repl, text_id):
    t = doc[i]
    text = doc.text
    if repl is None:  # deletion: remove the word and its trailing whitespace
        new = text[:t.idx] + text[t.idx + len(t.text_with_ws):]
        nxt = doc[i + 1]
        target_char, target_word = t.idx, nxt.text
        group, orig = nxt.lemma_.lower(), t.text
        after = doc[i + 2] if i + 2 < len(doc) else None
    else:
        repl = match_case(repl, t.text)
        new = text[:t.idx] + repl + text[t.idx + len(t.text):]
        target_char, target_word = t.idx, repl
        group, orig = t.lemma_.lower(), t.text
        after = doc[i + 1] if i + 1 < len(doc) else None
    shift = len(new) - len(text)
    return {"text_id": text_id, "type": typ, "orig": orig, "observed": target_word,
            "group": group, "target_char": target_char,
            "next_char": None if after is None else after.idx + shift,
            "next_word": None if after is None else after.text,
            "text": new}


def main():
    LAB.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    split = pd.read_csv(HERE.parent / "split.csv")
    essays = pd.read_parquet(S3 / "ellipse" / "essays.parquet").merge(split)
    dev = essays[essays.split == "dev"]
    if BENCH == "conf":
        host = clean_host = set(dev.text_id[dev.overall == 3.5])
    else:
        host = set(dev.text_id[dev.overall >= 4.0])
        clean_host = set(dev.text_id[dev.overall >= 4.5])
    docs = {tid: doc for tid, doc, _ in load_essays("ellipse", None) if tid in host}

    pool = {}
    for tid, doc in docs.items():
        per = {}
        for typ, i, repl in candidates(doc, rng):
            per.setdefault(typ, []).append((typ, i, repl))
        for typ, c in per.items():
            rng.shuffle(c)
            pool.setdefault(typ, []).extend((tid, x) for x in c[:MAX_PER_ESSAY])
    rows = []
    for typ, c in sorted(pool.items()):
        rng.shuffle(c)
        for tid, (t, i, repl) in c[:N_PER_TYPE]:
            rows.append(make_instance(docs[tid], t, i, repl, tid))

    # clean classes from the stored token table (no edit)
    toks = pd.concat([pd.read_parquet(p) for p in sorted((S3 / "ellipse" / "shards").glob("*.parquet"))])
    tau = pd.read_json(HERE.parents[2] / "results/study3/threshold.json", typ="series")["tau_bits"]
    toks = toks[toks.text_id.isin(host) & toks.is_alpha]
    surp = toks[(toks.surprisal_bits >= tau) & toks.text_id.isin(clean_host)]
    for typ, src in [("CLEAN_SURP", surp)]:
        for r in src.sample(min(N_PER_TYPE, len(src)), random_state=SEED).itertuples():
            doc = docs[r.text_id]
            t = doc[int(r.spacy_idx)]
            after = doc[t.i + 1] if t.i + 1 < len(doc) else None
            rows.append({"text_id": r.text_id, "type": typ, "orig": t.text, "observed": t.text,
                         "group": t.lemma_.lower(), "target_char": t.idx,
                         "next_char": None if after is None else after.idx,
                         "next_word": None if after is None else after.text, "text": doc.text})
    # lexeme-matched clean controls
    occ = {}
    for tid, doc in docs.items():
        for t in doc:
            if t.is_alpha or t.lower_ in HOMO:
                occ.setdefault(t.lower_, []).append((tid, t.i))
    n_err = len(rows) - min(N_PER_TYPE, len(surp))
    matched = []
    for k, r in enumerate(rows[:n_err]):
        opts = [o for o in occ.get(r["observed"].lower(), []) if o[0] != r["text_id"]]
        if not opts:
            continue
        tid, i = opts[rng.randrange(len(opts))]
        doc = docs[tid]
        t = doc[i]
        after = doc[i + 1] if i + 1 < len(doc) else None
        matched.append({"text_id": tid, "type": "MATCHED", "orig": t.text, "observed": t.text,
                        "group": t.lemma_.lower(), "target_char": t.idx,
                        "next_char": None if after is None else after.idx,
                        "next_word": None if after is None else after.text, "text": doc.text,
                        "matched_to": k, "matched_type": r["type"]})
    b = pd.DataFrame(rows + matched)
    b.insert(0, "inst", np.arange(len(b)))
    b.to_parquet(LAB / f"bench{'_' + BENCH if BENCH else ''}.parquet", index=False)
    print(b.type.value_counts())
    print(b.groupby("type").group.nunique())
    print("matched controls by type:", b.matched_type.value_counts().to_dict())
    for typ, g in b.groupby("type"):
        r = g.iloc[0]
        print(typ, "|", r.orig, "->", r.observed, "|", r.text[max(0, r.target_char - 50):r.target_char + 40].replace("\n", " "))


if __name__ == "__main__":
    main()
