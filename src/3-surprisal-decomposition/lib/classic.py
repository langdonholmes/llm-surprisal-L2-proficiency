"""Classic phraseological and lexical measures from spaCy parses (D32).

Dolma reference tables (data/dolma/frequency_tables, corpus a = edited prose,
b = unedited):
  bigram MI, ΔP (forward)   adjacent alphabetic token pairs, corpus a, attested only
  share MI ≥ 3              of the attested bigrams
  dependency MI             amod, dobj, advmod (Study 2's MiCalculator: a + b
                            summed, tags collapsed); "dep_mi" is the mean of the three
  content-word frequency    mean log10 relative frequency of attested content-word
                            forms, corpus a
  Study 2 lemma frequency   Study 2's log_mean_token_freq (lemma, a + b, unseen -> 1)

Each measure is a mean over items, so it is computed from item sums, which
lets an FCE script pool its two answers.
"""

import time

import numpy as np
import pandas as pd
import polars as pl

from util.paths import DATA_DIR

FREQ = DATA_DIR / "dolma" / "frequency_tables"
CONTENT = {"NOUN", "VERB", "ADJ", "ADV"}
RELS = ("amod", "dobj", "advmod")
MEASURES = ["bigram_mi", "bigram_dp", "share_mi3", "amod_mi", "dobj_mi", "advmod_mi", "dep_mi",
            "content_freq", "lemma_freq"]


def doc_items(docs):
    """Per-token, per-bigram and per-dependency rows; text_id is the doc's meta text_id."""
    tok, big, dep = [], [], []
    for doc in docs:
        tid = str(doc.user_data["meta"]["text_id"])
        prev = None
        for t in doc:
            tok.append((tid, t.lower_, t.lemma_.lower(), t.pos_, t.is_alpha))
            if t.is_alpha and prev is not None and prev.is_alpha and prev.i == t.i - 1:
                big.append((tid, prev.lower_, t.lower_))
            prev = t
            if t.dep_ in RELS:
                dep.append((tid, t.dep_, t.head.lemma_.lower(), t.lemma_.lower()))
    return (pd.DataFrame(tok, columns=["text_id", "lower", "lemma", "pos", "is_alpha"]),
            pd.DataFrame(big, columns=["text_id", "w1", "w2"]),
            pd.DataFrame(dep, columns=["text_id", "rel", "h", "d"]))


class Ref:
    """Lookups into the Dolma tables for a fixed set of query keys."""

    def __init__(self, uni_keys, bi_keys, dep_keys):
        t0 = time.time()
        self.N, self.uni, self.bi = {}, {}, {}
        for c in ["a", "b"]:
            lf = pl.scan_parquet(str(FREQ / f"corpus_{c}" / "3grams.parquet"))
            self.N[c] = int(lf.select(pl.col("count").sum()).collect().item())
            keys = pl.DataFrame({"u": sorted(set(uni_keys))}).lazy()
            u = (lf.join(keys, left_on="token_2", right_on="u", how="semi").group_by("token_2")
                 .agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
            self.uni[c] = dict(zip(u["token_2"].to_list(), u["c"].to_list()))
            if c == "a":
                bk = pl.DataFrame(list(set(bi_keys)), schema=["a", "b"], orient="row").lazy()
                b = (lf.join(bk, left_on=["token_1", "token_2"], right_on=["a", "b"], how="semi")
                     .group_by(["token_1", "token_2"]).agg(pl.col("count").sum().alias("c"))
                     .collect(engine="streaming"))
                self.bi["a"] = dict(zip(zip(b["token_1"].to_list(), b["token_2"].to_list()), b["c"].to_list()))
        dk = pl.DataFrame(list(set(dep_keys)), schema=["h", "d", "rel"], orient="row")
        hs = pl.DataFrame({"h": dk["h"].unique()}).lazy()
        ds = pl.DataFrame({"d": dk["d"].unique()}).lazy()
        self.dep_total, J, H, D = 0, {}, {}, {}
        for c in ["a", "b"]:
            lf = pl.scan_parquet(str(FREQ / f"corpus_{c}" / "depgrams.parquet"))
            self.dep_total += int(lf.select(pl.col("count").sum()).collect().item())
            j = (lf.join(dk.lazy(), left_on=["head_lemma", "dependent_lemma", "relation"],
                         right_on=["h", "d", "rel"], how="semi")
                 .group_by(["head_lemma", "dependent_lemma", "relation"]).agg(pl.col("count").sum())
                 .collect(engine="streaming"))
            for h, d, r, n in j.iter_rows():
                J[(h, d, r)] = J.get((h, d, r), 0) + n
            for col, key, acc in [("head_lemma", hs, H), ("dependent_lemma", ds, D)]:
                m = (lf.join(key, left_on=col, right_on=key.collect_schema().names()[0], how="semi")
                     .group_by(col).agg(pl.col("count").sum()).collect(engine="streaming"))
                for w, n in m.iter_rows():
                    acc[w] = acc.get(w, 0) + n
        self.J, self.H, self.D = J, H, D
        print(f"Dolma lookups: {time.time() - t0:.0f}s", flush=True)


def item_sums(tok, big, dep, ref: Ref) -> pd.DataFrame:
    """Per text_id: the sum and count behind every measure."""
    Na, Nab = ref.N["a"], ref.N["a"] + ref.N["b"]
    S = {}
    al = tok[tok.is_alpha]
    cnt_a = al.lower.map(ref.uni["a"]).fillna(0).to_numpy(float)
    cnt_lemma = (al.lemma.map(ref.uni["a"]).fillna(0) + al.lemma.map(ref.uni["b"]).fillna(0)).to_numpy(float)
    g = pd.Series(np.log10(np.clip(cnt_lemma, 1, None) / Nab)).groupby(al.text_id.to_numpy())
    S["lemma_freq_s"], S["lemma_freq_n"] = g.sum(), g.size()
    cw = al.pos.isin(CONTENT).to_numpy() & (cnt_a > 0)
    g = pd.Series(np.log10(cnt_a[cw] / Na)).groupby(al.text_id.to_numpy()[cw])
    S["content_freq_s"], S["content_freq_n"] = g.sum(), g.size()

    c12 = np.array([ref.bi["a"].get(k, 0) for k in zip(big.w1, big.w2)], float)
    # the tables prune trigrams seen once, so a derived unigram count can fall
    # below the bigram count; floor the marginals at the joint count
    c1 = np.maximum(big.w1.map(ref.uni["a"]).fillna(0).to_numpy(float), c12)
    c2 = np.maximum(big.w2.map(ref.uni["a"]).fillna(0).to_numpy(float), c12)
    att = c12 > 0
    c12, c1, c2 = c12[att], c1[att], c2[att]
    b = pd.DataFrame({"text_id": big.text_id.to_numpy()[att], "mi": np.log2(c12 * Na / (c1 * c2)),
                      "dp": c12 / c1 - (c2 - c12) / (Na - c1)})
    b["mi3"] = (b.mi >= 3).astype(float)
    g = b.groupby("text_id")
    for m, col in [("bigram_mi", "mi"), ("bigram_dp", "dp"), ("share_mi3", "mi3")]:
        S[f"{m}_s"], S[f"{m}_n"] = g[col].sum(), g.size()

    j = np.array([ref.J.get(k, 0) for k in zip(dep.h, dep.d, dep.rel)], float)
    T = ref.dep_total
    hm = dep.h.map(ref.H).fillna(0).to_numpy(float)
    dm = dep.d.map(ref.D).fillna(0).to_numpy(float)
    ok = j > 0
    d = pd.DataFrame({"text_id": dep.text_id.to_numpy()[ok], "rel": dep.rel.to_numpy()[ok],
                      "mi": np.log2((j[ok] / T) / ((hm[ok] / T) * (dm[ok] / T)))})
    for r in RELS:
        g = d[d.rel == r].groupby("text_id").mi
        S[f"{r}_mi_s"], S[f"{r}_mi_n"] = g.sum(), g.size()
    return pd.DataFrame(S).fillna(0.0)


def measures(sums: pd.DataFrame) -> pd.DataFrame:
    """Measures from summed item sums (one row per unit); undefined with no items."""
    M = pd.DataFrame(index=sums.index)
    for m in MEASURES:
        if m != "dep_mi":
            M[m] = (sums[f"{m}_s"] / sums[f"{m}_n"]).where(sums[f"{m}_n"] > 0)
    M["dep_mi"] = M[[f"{r}_mi" for r in RELS]].mean(axis=1, skipna=True)
    return M[MEASURES]
