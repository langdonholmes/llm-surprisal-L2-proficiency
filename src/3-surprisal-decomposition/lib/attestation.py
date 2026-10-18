"""Attestation of each bench token in the Dolma reference tables built for
Study 2 (data/dolma/frequency_tables/corpus_{a,b}): a = edited prose,
b = unedited prose.

The 3-gram tables count lowercased spaCy tokens with two '#' pads at the start
of each document (min_count 2 after merging), so every occurrence of a token is
counted once as token_2. Hence

    unigram(w)      = sum count over token_2 == w
    bigram(a, b)    = sum count over (token_1, token_2) == (a, b)

both slight undercounts (trigrams seen once are pruned).

Output, row-aligned with the bench (data/study3/repair/attestation_<bench>.parquet):

    found                  1 if the token was located in the spaCy doc
    pos, is_punct, like_num
    {a,b}_log_uni          log(1 + unigram count)
    {a,b}_unseen_uni       the word never occurs in that corpus
    {a,b}_unseen_bi_left   (previous word, w) never occurs
    {a,b}_unseen_bi_right  (w, next word) never occurs

No gold labels are read.
"""

import time

import numpy as np
import pandas as pd
import polars as pl

from util.paths import DATA_DIR, ELLIPSE_DOCBINS_DIR, TOEFL_DOCBINS_DIR, WI_DOCBINS_DIR
from util.process_docs import load_all_docbins

FREQ = DATA_DIR / "dolma" / "frequency_tables"
PAD = "#"
DOCBINS = {"ellipse": ELLIPSE_DOCBINS_DIR, "wi": WI_DOCBINS_DIR, "toefl11": TOEFL_DOCBINS_DIR}


def token_context(bench: pd.DataFrame, docbin_dir) -> pd.DataFrame:
    """The token, its neighbours and a few lexical attributes for each bench row."""
    want = set(bench.text_id)
    docs = {}
    for doc in load_all_docbins(docbin_dir):
        m = doc.user_data.get("meta")
        tid = str(m["text_id"] if isinstance(m, dict) else m)
        if tid in want:
            docs[tid] = doc
    rows = []
    for r in bench.itertuples():
        doc = docs.get(r.text_id)
        t = next((x for x in doc if x.idx == int(r.target_char)), None) if doc is not None else None
        if t is None:
            rows.append({"found": 0})
            continue
        i = t.i
        rows.append({
            "found": 1, "w": t.lower_,
            "p1": doc[i - 1].lower_ if i >= 1 else PAD,
            "n1": doc[i + 1].lower_ if i + 1 < len(doc) else PAD,
            "pos": t.pos_, "is_punct": float(t.is_punct), "like_num": float(t.like_num),
        })
    return pd.DataFrame(rows)


def ngram_counts(ctx: pd.DataFrame, corpus: str) -> pd.DataFrame:
    lf = pl.scan_parquet(str(FREQ / f"corpus_{corpus}" / "3grams.parquet"))
    c = pl.from_pandas(ctx[["w", "p1", "n1"]].fillna(PAD))
    uni = (lf.join(c.select(pl.col("w").alias("u")).unique().lazy(), left_on="token_2", right_on="u", how="semi")
           .group_by("token_2").agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
    bi_keys = pl.concat([c.select(pl.col("p1").alias("a"), pl.col("w").alias("b")),
                         c.select(pl.col("w").alias("a"), pl.col("n1").alias("b"))]).unique()
    bi = (lf.join(bi_keys.lazy(), left_on=["token_1", "token_2"], right_on=["a", "b"], how="semi")
          .group_by(["token_1", "token_2"]).agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
    U = dict(zip(uni["token_2"].to_list(), uni["c"].to_list()))
    B = dict(zip(zip(bi["token_1"].to_list(), bi["token_2"].to_list()), bi["c"].to_list()))
    w, p1, n1 = (ctx[k].fillna(PAD).tolist() for k in ["w", "p1", "n1"])
    cu = np.array([U.get(x, 0) for x in w], float)
    out = pd.DataFrame(index=ctx.index)
    out[f"{corpus}_log_uni"] = np.log1p(cu)
    out[f"{corpus}_unseen_uni"] = (cu == 0).astype(float)
    out[f"{corpus}_unseen_bi_left"] = np.array([B.get(k, 0) == 0 for k in zip(p1, w)], float)
    out[f"{corpus}_unseen_bi_right"] = np.array([B.get(k, 0) == 0 for k in zip(w, n1)], float)
    return out


def build(ctx) -> pd.DataFrame:
    """Attestation table for a bench, computed (CPU) if it is not cached."""
    path = ctx.path("attestation")
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    c = token_context(ctx.tokens, DOCBINS[ctx.name.split("_")[0]])
    print(f"context {time.time() - t0:.0f}s; found {c.found.mean():.3f}", flush=True)
    f = pd.concat([c[["found", "pos", "is_punct", "like_num"]]]
                  + [ngram_counts(c, corpus) for corpus in ["a", "b"]], axis=1)
    f.to_parquet(path, index=False)
    print(f"n-grams {time.time() - t0:.0f}s", flush=True)
    return f
