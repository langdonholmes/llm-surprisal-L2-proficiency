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
    tokens, bigrams, deps = [], [], []
    for doc in docs:
        text_id = str(doc.user_data["meta"]["text_id"])
        prev = None
        for t in doc:
            tokens.append((text_id, t.lower_, t.lemma_.lower(), t.pos_, t.is_alpha))
            if t.is_alpha and prev is not None and prev.is_alpha and prev.i == t.i - 1:
                bigrams.append((text_id, prev.lower_, t.lower_))
            prev = t
            if t.dep_ in RELS:
                deps.append((text_id, t.dep_, t.head.lemma_.lower(), t.lemma_.lower()))
    return (pd.DataFrame(tokens, columns=["text_id", "lower", "lemma", "pos", "is_alpha"]),
            pd.DataFrame(bigrams, columns=["text_id", "w1", "w2"]),
            pd.DataFrame(deps, columns=["text_id", "rel", "h", "d"]))


class Reference:
    """Lookups into the Dolma tables for a fixed set of query keys."""

    def __init__(self, unigram_keys, bigram_keys, dep_keys):
        t0 = time.time()
        self.n_tokens, self.unigrams, self.bigrams = {}, {}, {}
        for corpus in ["a", "b"]:
            # unigram and bigram counts are summed out of the trigram table
            table = pl.scan_parquet(str(FREQ / f"corpus_{corpus}" / "3grams.parquet"))
            self.n_tokens[corpus] = int(table.select(pl.col("count").sum()).collect().item())
            keys = pl.DataFrame({"u": sorted(set(unigram_keys))}).lazy()
            counts = (table.join(keys, left_on="token_2", right_on="u", how="semi").group_by("token_2")
                      .agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
            self.unigrams[corpus] = dict(zip(counts["token_2"].to_list(), counts["c"].to_list()))
            if corpus == "a":
                keys = pl.DataFrame(list(set(bigram_keys)), schema=["a", "b"], orient="row").lazy()
                counts = (table.join(keys, left_on=["token_1", "token_2"], right_on=["a", "b"], how="semi")
                          .group_by(["token_1", "token_2"]).agg(pl.col("count").sum().alias("c"))
                          .collect(engine="streaming"))
                self.bigrams["a"] = dict(zip(zip(counts["token_1"].to_list(), counts["token_2"].to_list()),
                                             counts["c"].to_list()))
        dep_frame = pl.DataFrame(list(set(dep_keys)), schema=["h", "d", "rel"], orient="row")
        heads = pl.DataFrame({"h": dep_frame["h"].unique()}).lazy()
        dependents = pl.DataFrame({"d": dep_frame["d"].unique()}).lazy()
        # joint (head, dependent, relation) counts and the two marginals, corpora a + b summed
        self.dep_total, joint, head_counts, dep_counts = 0, {}, {}, {}
        for corpus in ["a", "b"]:
            table = pl.scan_parquet(str(FREQ / f"corpus_{corpus}" / "depgrams.parquet"))
            self.dep_total += int(table.select(pl.col("count").sum()).collect().item())
            found = (table.join(dep_frame.lazy(), left_on=["head_lemma", "dependent_lemma", "relation"],
                                right_on=["h", "d", "rel"], how="semi")
                     .group_by(["head_lemma", "dependent_lemma", "relation"]).agg(pl.col("count").sum())
                     .collect(engine="streaming"))
            for head, dep, rel, n in found.iter_rows():
                joint[(head, dep, rel)] = joint.get((head, dep, rel), 0) + n
            for col, keys, acc in [("head_lemma", heads, head_counts), ("dependent_lemma", dependents, dep_counts)]:
                found = (table.join(keys, left_on=col, right_on=keys.collect_schema().names()[0], how="semi")
                         .group_by(col).agg(pl.col("count").sum()).collect(engine="streaming"))
                for word, n in found.iter_rows():
                    acc[word] = acc.get(word, 0) + n
        self.dep_joint, self.dep_head, self.dep_dependent = joint, head_counts, dep_counts
        print(f"Dolma lookups: {time.time() - t0:.0f}s", flush=True)


def item_sums(tokens, bigrams, deps, ref: Reference) -> pd.DataFrame:
    """Per text_id: the sum and count behind every measure."""
    n_a, n_ab = ref.n_tokens["a"], ref.n_tokens["a"] + ref.n_tokens["b"]
    sums = {}
    alpha = tokens[tokens.is_alpha]
    # word frequencies
    count_a = alpha.lower.map(ref.unigrams["a"]).fillna(0).to_numpy(float)
    count_lemma = (alpha.lemma.map(ref.unigrams["a"]).fillna(0)
                   + alpha.lemma.map(ref.unigrams["b"]).fillna(0)).to_numpy(float)
    by_text = pd.Series(np.log10(np.clip(count_lemma, 1, None) / n_ab)).groupby(alpha.text_id.to_numpy())
    sums["lemma_freq_s"], sums["lemma_freq_n"] = by_text.sum(), by_text.size()
    content = alpha.pos.isin(CONTENT).to_numpy() & (count_a > 0)
    by_text = pd.Series(np.log10(count_a[content] / n_a)).groupby(alpha.text_id.to_numpy()[content])
    sums["content_freq_s"], sums["content_freq_n"] = by_text.sum(), by_text.size()

    # bigram MI and ΔP
    c12 = np.array([ref.bigrams["a"].get(k, 0) for k in zip(bigrams.w1, bigrams.w2)], float)
    # the tables prune trigrams seen once, so a derived unigram count can fall
    # below the bigram count; floor the marginals at the joint count
    c1 = np.maximum(bigrams.w1.map(ref.unigrams["a"]).fillna(0).to_numpy(float), c12)
    c2 = np.maximum(bigrams.w2.map(ref.unigrams["a"]).fillna(0).to_numpy(float), c12)
    attested = c12 > 0
    c12, c1, c2 = c12[attested], c1[attested], c2[attested]
    bi = pd.DataFrame({"text_id": bigrams.text_id.to_numpy()[attested], "mi": np.log2(c12 * n_a / (c1 * c2)),
                       "dp": c12 / c1 - (c2 - c12) / (n_a - c1)})
    bi["mi3"] = (bi.mi >= 3).astype(float)
    by_text = bi.groupby("text_id")
    for m, col in [("bigram_mi", "mi"), ("bigram_dp", "dp"), ("share_mi3", "mi3")]:
        sums[f"{m}_s"], sums[f"{m}_n"] = by_text[col].sum(), by_text.size()

    # dependency MI
    joint = np.array([ref.dep_joint.get(k, 0) for k in zip(deps.h, deps.d, deps.rel)], float)
    total = ref.dep_total
    head = deps.h.map(ref.dep_head).fillna(0).to_numpy(float)
    dependent = deps.d.map(ref.dep_dependent).fillna(0).to_numpy(float)
    attested = joint > 0
    dep_mi = pd.DataFrame({"text_id": deps.text_id.to_numpy()[attested], "rel": deps.rel.to_numpy()[attested],
                           "mi": np.log2((joint[attested] / total)
                                         / ((head[attested] / total) * (dependent[attested] / total)))})
    for rel in RELS:
        by_text = dep_mi[dep_mi.rel == rel].groupby("text_id").mi
        sums[f"{rel}_mi_s"], sums[f"{rel}_mi_n"] = by_text.sum(), by_text.size()
    return pd.DataFrame(sums).fillna(0.0)


def measures(sums: pd.DataFrame) -> pd.DataFrame:
    """Measures from summed item sums (one row per unit); undefined with no items."""
    out = pd.DataFrame(index=sums.index)
    for m in MEASURES:
        if m != "dep_mi":
            out[m] = (sums[f"{m}_s"] / sums[f"{m}_n"]).where(sums[f"{m}_n"] > 0)
    out["dep_mi"] = out[[f"{rel}_mi" for rel in RELS]].mean(axis=1, skipna=True)
    return out[MEASURES]
