"""Attestation lab — build per-token attestation and lexis features for a real bench.

For every bench row, the token (spaCy doc token whose idx == target_char) and its
neighbours are looked up in the Dolma reference tables built for Study 2
(data/dolma/frequency_tables/corpus_{a,b}): a = edited prose, b = unedited.

The 3-gram tables count lowercased spaCy tokens with two '#' pads at the start
of each document (min_count 2 after merging), so every occurrence of a token is
counted once as token_2. Hence

    unigram(w)      = sum count over token_2 == w
    bigram(a, b)    = sum count over (token_1, token_2) == (a, b)
    trigram(a,b,c)  = count of (a, b, c)

all slight undercounts (trigrams seen once are pruned). Depgrams are aggregated
over tags to (head_lemma, relation, dep_lemma), excluding ROOT and punct.

Output (row-aligned with the bench, SUFFIX-keyed):
    data/study3/lab/att/feats{SUFFIX}.parquet   named numeric features
No gold labels, levels or splits are read.

    LAB_BENCH=real_wi OMP_NUM_THREADS=8 python src/3-vector-transformations/lab/att_build.py
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lab.harness import LAB, SUFFIX  # noqa: E402
from util.paths import DATA_DIR, ELLIPSE_DOCBINS_DIR, TOEFL_DOCBINS_DIR, WI_DOCBINS_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

ATT = LAB / "att"
FREQ = DATA_DIR / "dolma" / "frequency_tables"
PAD = "#"
POS = ["NOUN", "VERB", "ADJ", "ADV", "PROPN", "NUM", "PRON", "DET", "ADP", "AUX", "CCONJ",
       "SCONJ", "PART", "PUNCT", "INTJ", "SYM", "X"]
DOCBINS = {"real_wi": WI_DOCBINS_DIR}


def docbin_dir():
    b = os.environ.get("LAB_BENCH", "")
    if b.startswith("real_wi"):
        return WI_DOCBINS_DIR
    if b.startswith("real_ellipse"):
        return ELLIPSE_DOCBINS_DIR
    if b.startswith("real_toefl"):
        return TOEFL_DOCBINS_DIR
    raise SystemExit(f"att_build supports real benches only (LAB_BENCH={b!r})")


def token_context(bench: pd.DataFrame) -> pd.DataFrame:
    """Neighbours, lemma, arcs and lexical attributes for each bench row."""
    want = set(bench.text_id)
    docs = {}
    for doc in load_all_docbins(docbin_dir()):
        m = doc.user_data.get("meta")
        tid = str(m["text_id"] if isinstance(m, dict) else m)
        if tid in want:
            docs[tid] = doc
    import spacy
    vec_nlp = spacy.load("en_core_web_lg", exclude=["tagger", "parser", "ner", "lemmatizer",
                                                    "attribute_ruler", "senter", "tok2vec"])
    vocab = vec_nlp.vocab
    essay_vec = {}

    def vec(w):
        lex = vocab[w]
        return lex.vector if lex.has_vector else None

    rows = []
    for r in bench.itertuples():
        doc = docs.get(r.text_id)
        t = None
        if doc is not None:
            t = next((x for x in doc if x.idx == int(r.target_char)), None)
        if t is None:
            rows.append({"found": 0})
            continue
        i = t.i
        tx = [x.text.lower() for x in doc]
        get = lambda j: tx[j] if 0 <= j < len(tx) else PAD  # noqa: E731
        if r.text_id not in essay_vec:
            vs = [vocab[x.lower_].vector for x in doc
                  if x.pos_ in ("NOUN", "VERB", "ADJ", "ADV") and vocab[x.lower_].has_vector]
            essay_vec[r.text_id] = (np.sum(vs, 0), len(vs)) if vs else (None, 0)
        v = vec(t.lower_)
        sim_essay = sim_sent = np.nan
        if v is not None and np.linalg.norm(v) > 0:
            tot, n = essay_vec[r.text_id]
            if tot is not None and n > 1:
                m = tot - (v if t.pos_ in ("NOUN", "VERB", "ADJ", "ADV") else 0)
                sim_essay = float(v @ m / (np.linalg.norm(v) * np.linalg.norm(m) + 1e-9))
            sv = [vocab[x.lower_].vector for x in t.sent if x.i != i and x.is_alpha and vocab[x.lower_].has_vector]
            if sv:
                m = np.mean(sv, 0)
                sim_sent = float(v @ m / (np.linalg.norm(v) * np.linalg.norm(m) + 1e-9))
        children = [(t.lemma_.lower(), c.dep_, c.lemma_.lower()) for c in t.children
                    if c.dep_ not in ("punct",)]
        rows.append({
            "found": 1, "w": get(i), "p1": get(i - 1) if i >= 1 else PAD, "p2": get(i - 2) if i >= 2 else PAD,
            "n1": get(i + 1), "n2": get(i + 2),
            "lemma": t.lemma_.lower(), "pos": t.pos_,
            "arc_head": t.head.lemma_.lower(), "arc_rel": t.dep_,
            "children": children,
            "is_title_mid": float(t.is_title and not t.is_sent_start),
            "is_sent_start": float(t.is_sent_start), "like_num": float(t.like_num),
            "is_alpha": float(t.is_alpha), "is_punct": float(t.is_punct), "length": float(len(t.text)),
            "has_vector": float(v is not None), "sim_essay": sim_essay, "sim_sent": sim_sent,
            "word_count_in_essay": float(sum(1 for x in doc if x.lower_ == t.lower_)),
        })
    return pd.DataFrame(rows)


def ngram_counts(ctx: pd.DataFrame, corpus: str) -> pd.DataFrame:
    path = str(FREQ / f"corpus_{corpus}" / "3grams.parquet")
    lf = pl.scan_parquet(path)
    c = pl.from_pandas(ctx[["w", "p1", "p2", "n1", "n2"]].fillna(PAD))
    uni_keys = pl.concat([c.select(pl.col(k).alias("u")) for k in ["w", "p1", "n1"]]).unique()
    uni = (lf.join(uni_keys.lazy(), left_on="token_2", right_on="u", how="semi")
           .group_by("token_2").agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
    bi_keys = pl.concat([c.select(pl.col("p1").alias("a"), pl.col("w").alias("b")),
                         c.select(pl.col("w").alias("a"), pl.col("n1").alias("b"))]).unique()
    bi = (lf.join(bi_keys.lazy(), left_on=["token_1", "token_2"], right_on=["a", "b"], how="semi")
          .group_by(["token_1", "token_2"]).agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
    tri_keys = pl.concat([c.select(pl.col("p2").alias("a"), pl.col("p1").alias("b"), pl.col("w").alias("cc")),
                          c.select(pl.col("p1").alias("a"), pl.col("w").alias("b"), pl.col("n1").alias("cc")),
                          c.select(pl.col("w").alias("a"), pl.col("n1").alias("b"), pl.col("n2").alias("cc"))]).unique()
    tri = (lf.join(tri_keys.lazy(), left_on=["token_0", "token_1", "token_2"], right_on=["a", "b", "cc"], how="semi")
           .select(["token_0", "token_1", "token_2", "count"]).collect(engine="streaming"))
    total = lf.select(pl.col("count").sum()).collect().item()
    U = dict(zip(uni["token_2"].to_list(), uni["c"].to_list()))
    B = dict(zip(zip(bi["token_1"].to_list(), bi["token_2"].to_list()), bi["c"].to_list()))
    T = dict(zip(zip(tri["token_0"].to_list(), tri["token_1"].to_list(), tri["token_2"].to_list()),
                 tri["count"].to_list()))
    out = pd.DataFrame(index=ctx.index)
    w, p1, p2, n1, n2 = (ctx[k].fillna(PAD).tolist() for k in ["w", "p1", "p2", "n1", "n2"])
    cu = np.array([U.get(x, 0) for x in w], float)
    cp = np.array([U.get(x, 0) for x in p1], float)
    cn = np.array([U.get(x, 0) for x in n1], float)
    bl = np.array([B.get((a, b), 0) for a, b in zip(p1, w)], float)
    br = np.array([B.get((a, b), 0) for a, b in zip(w, n1)], float)
    out[f"{corpus}_log_uni"] = np.log1p(cu)
    out[f"{corpus}_log_bi_left"] = np.log1p(bl)
    out[f"{corpus}_log_bi_right"] = np.log1p(br)
    out[f"{corpus}_log_tri_left"] = np.log1p([T.get(k, 0) for k in zip(p2, p1, w)])
    out[f"{corpus}_log_tri_mid"] = np.log1p([T.get(k, 0) for k in zip(p1, w, n1)])
    out[f"{corpus}_log_tri_right"] = np.log1p([T.get(k, 0) for k in zip(w, n1, n2)])
    # PMI of the two bigrams (add-0.5 smoothing); conditional log p(w | prev), log p(next | w)
    out[f"{corpus}_pmi_left"] = np.log2((bl + .5) * total / ((cp + 1) * (cu + 1)))
    out[f"{corpus}_pmi_right"] = np.log2((br + .5) * total / ((cu + 1) * (cn + 1)))
    out[f"{corpus}_lp_w_given_prev"] = np.log2((bl + .5) / (cp + 1))
    out[f"{corpus}_lp_next_given_w"] = np.log2((br + .5) / (cu + 1))
    out[f"{corpus}_unseen_uni"] = (cu == 0).astype(float)
    out[f"{corpus}_unseen_bi_left"] = (bl == 0).astype(float)
    out[f"{corpus}_unseen_bi_right"] = (br == 0).astype(float)
    return out


def dep_counts(ctx: pd.DataFrame, corpus: str) -> pd.DataFrame:
    path = str(FREQ / f"corpus_{corpus}" / "depgrams.parquet")
    lf = pl.scan_parquet(path).filter(~pl.col("relation").is_in(["ROOT", "punct"]))
    arcs = [(h, r, d) for h, r, d in zip(ctx.arc_head.fillna(""), ctx.arc_rel.fillna(""), ctx.lemma.fillna(""))]
    kids = [k for ks in ctx.children for k in (ks if isinstance(ks, (list, np.ndarray)) else [])]
    keys = pl.DataFrame([tuple(x) for x in set(arcs) | set(map(tuple, kids))],
                        schema=["h", "r", "d"], orient="row")
    j = (lf.join(keys.lazy(), left_on=["head_lemma", "relation", "dependent_lemma"], right_on=["h", "r", "d"], how="semi")
         .group_by(["head_lemma", "relation", "dependent_lemma"]).agg(pl.col("count").sum().alias("c"))
         .collect(engine="streaming"))
    lem = set(ctx.lemma.dropna()) | set(ctx.arc_head.dropna()) | {k[2] for k in kids}
    lem_df = pl.DataFrame({"l": list(lem)})
    hm = (lf.join(lem_df.lazy(), left_on="head_lemma", right_on="l", how="semi")
          .group_by("head_lemma").agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
    dm = (lf.join(lem_df.lazy(), left_on="dependent_lemma", right_on="l", how="semi")
          .group_by("dependent_lemma").agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
    total = lf.select(pl.col("count").sum()).collect().item()
    J = dict(zip(zip(j["head_lemma"].to_list(), j["relation"].to_list(), j["dependent_lemma"].to_list()),
                 j["c"].to_list()))
    H = dict(zip(hm["head_lemma"].to_list(), hm["c"].to_list()))
    D = dict(zip(dm["dependent_lemma"].to_list(), dm["c"].to_list()))

    def mi(h, r, d):
        c = J.get((h, r, d), 0)
        return np.log2((c + .5) * total / ((H.get(h, 0) + 1) * (D.get(d, 0) + 1))), c

    out = pd.DataFrame(index=ctx.index)
    own = [mi(*a) if a[1] not in ("ROOT", "punct", "") else (np.nan, np.nan) for a in arcs]
    out[f"{corpus}_arc_mi"] = [o[0] for o in own]
    out[f"{corpus}_arc_logc"] = np.log1p([o[1] for o in own])
    kid_mi = []
    for ks in ctx.children:
        ks = list(ks) if isinstance(ks, (list, np.ndarray)) else []
        v = [mi(*tuple(k)) for k in ks]
        kid_mi.append((min((x[0] for x in v), default=np.nan), float(np.mean([x[1] == 0 for x in v])) if v else np.nan,
                       len(v)))
    out[f"{corpus}_kid_min_mi"] = [k[0] for k in kid_mi]
    out[f"{corpus}_kid_unseen_share"] = [k[1] for k in kid_mi]
    out["n_children"] = [k[2] for k in kid_mi]
    return out


def build(bench: pd.DataFrame) -> pd.DataFrame:
    ATT.mkdir(parents=True, exist_ok=True)
    path = ATT / f"feats{SUFFIX}.parquet"
    if path.exists():
        return pd.read_parquet(path)
    t0 = time.time()
    ctx = token_context(bench)
    print(f"context {time.time() - t0:.0f}s; found {ctx.found.mean():.3f}", flush=True)
    parts = [ctx.drop(columns=["w", "p1", "p2", "n1", "n2", "lemma", "pos", "arc_head", "arc_rel", "children"],
                      errors="ignore")]
    for k in POS:
        parts.append(pd.DataFrame({f"pos_{k}": (ctx.pos == k).astype(float)}))
    for corpus in ["a", "b"]:
        parts.append(ngram_counts(ctx, corpus))
        print(f"ngrams {corpus} {time.time() - t0:.0f}s", flush=True)
        parts.append(dep_counts(ctx, corpus))
        print(f"deps {corpus} {time.time() - t0:.0f}s", flush=True)
    f = pd.concat(parts, axis=1)
    f = f.loc[:, ~f.columns.duplicated()]
    f["seconds_total"] = time.time() - t0
    f.to_parquet(path, index=False)
    return f


if __name__ == "__main__":
    b = pd.read_parquet(LAB / f"bench{SUFFIX}.parquet")
    f = build(b)
    print(f.describe().T.round(2).to_string())
