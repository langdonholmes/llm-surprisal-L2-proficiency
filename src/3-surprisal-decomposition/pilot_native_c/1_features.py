"""Native-vs-C pilot, step 1 (CPU) — token and essay features for the controls.

No surprisal is compared between levels here. Reads the corrected-essay pilot
(data/study3/pilot_corrected/) and writes, to data/study3/pilot_native_c/:

  tokens.parquet  one row per scored, non-space post-correction token:
      text_id, level, spacy_idx, pos_in_essay, text, low, lemma_low, pos, tag,
      surprisal_bits, own (aligned writer-own token, the D22 `corr_own` set),
      propn_num, earlier (lemma used earlier in the essay), topic_essay,
      topic_cluster, sent_init, para_init, freq_pm (edited Dolma, word form)
  essays.parquet  per essay: level, n_tokens, cluster, genre flags,
      register features, topic similarity to LOCNESS, register propensity,
      and the subset flags of control 6 (c6_genre, c6_topic, c6_register)

    python src/3-surprisal-decomposition/pilot_native_c/1_features.py
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, normalize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from util.paths import DATA_DIR  # noqa: E402
from util.process_docs import load_all_docbins  # noqa: E402

SRC = DATA_DIR / "study3" / "pilot_corrected"
OUT = DATA_DIR / "study3" / "pilot_native_c"
FREQ = DATA_DIR / "dolma" / "frequency_tables" / "corpus_a" / "3grams.parquet"
SEED = 42
CONTENT = {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}
K_CLUSTERS = 30
KEY_RATIO = 10.0       # topic word: essay or cluster rate >= 10x the Dolma rate
CLUSTER_DF = 0.10      # ... and (cluster rule) in >= 10% of the cluster's essays, >= 3 essays
NN_TOPIC = 5           # control 6b: the 5 nearest C essays of each N essay
LETTER_OPEN = re.compile(r"^\W*(dear|hi|hello|hey|to:|good (morning|afternoon|evening))\b", re.I)
LETTER_CLOSE = re.compile(r"(yours sincerely|yours faithfully|best wishes|kind regards|best regards|"
                          r"warm regards|love,|see you soon|write back soon|hope to hear from you)", re.I)
ARG_MARKERS = re.compile(r"\b(in my opinion|i (strongly )?(believe|think|agree|disagree)|in conclusion|"
                         r"on the other hand|advantages?|disadvantages?|however|therefore|argue[ds]?|"
                         r"to sum up|furthermore|moreover|should)\b", re.I)


def dolma_unigrams(words: list[str]) -> tuple[dict, int]:
    lf = pl.scan_parquet(str(FREQ))
    total = lf.select(pl.col("count").sum()).collect().item()
    u = pl.DataFrame({"u": sorted(set(words))})
    uni = (lf.join(u.lazy(), left_on="token_2", right_on="u", how="semi")
           .group_by("token_2").agg(pl.col("count").sum().alias("c")).collect(engine="streaming"))
    return dict(zip(uni["token_2"].to_list(), uni["c"].to_list())), total


def doc_flags(docs: dict) -> pd.DataFrame:
    rows = []
    for tid, doc in docs.items():
        text = doc.text
        seen = set()
        k = 0
        for t in doc:
            if t.is_space:
                continue
            before = text[:t.idx].rstrip(" \t ")
            lem = t.lemma_.lower()
            wordlike = any(ch.isalpha() for ch in t.text)
            rows.append((tid, t.i, k, bool(t.is_sent_start), before == "" or before.endswith("\n"),
                         wordlike and lem in seen))
            if wordlike:
                seen.add(lem)
            k += 1
    return pd.DataFrame(rows, columns=["text_id", "spacy_idx", "pos_in_essay", "sent_init", "para_init", "earlier"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    essays = pd.read_parquet(SRC / "essays.parquet", columns=["text_id", "level", "corrected"])
    level = dict(zip(essays.text_id, essays.level))
    corr = pd.read_parquet(SRC / "tokens_corr.parquet")
    corr = corr[~corr.is_space].copy()
    al = pd.read_parquet(SRC / "aligned.parquet", columns=["text_id", "aligned", "c_spacy_idx"])
    own = set(zip(al.text_id[al.aligned], al.c_spacy_idx[al.aligned]))
    corr["own"] = [(t, i) in own for t, i in zip(corr.text_id, corr.spacy_idx)]
    corr["level"] = corr.text_id.map(level)
    corr["low"] = corr.text.str.lower()
    corr["lemma_low"] = corr.lemma.str.lower()
    corr["propn_num"] = corr.pos.isin(["PROPN", "NUM"]) | corr.tag.isin(["NNP", "NNPS", "CD"]) | corr.like_num \
        | corr.text.str.contains(r"\d")

    docs = {d.user_data["meta"]["text_id"]: d for d in load_all_docbins(SRC / "docbins_corrected")}
    fl = doc_flags(docs)
    corr = corr.merge(fl, on=["text_id", "spacy_idx"], how="left", validate="1:1")
    assert corr.sent_init.notna().all()

    U, total = dolma_unigrams(corr.low.tolist())
    corr["dolma_count"] = corr.low.map(U).fillna(0)
    corr["freq_pm"] = corr.dolma_count / total * 1e6
    p_dolma = (corr.dolma_count + 0.5) / total     # floor for unseen words

    # ---- topic clusters (content lemmas, TF-IDF -> SVD -> k-means) ----------
    cont = corr[corr.pos.isin(CONTENT) & corr.text.str.contains(r"[A-Za-z]")]
    bags = cont.groupby("text_id").lemma_low.apply(lambda s: " ".join(x.replace(" ", "_") for x in s))
    ids = essays.text_id[essays.text_id.isin(bags.index)].tolist()
    vec = TfidfVectorizer(min_df=3, sublinear_tf=True, token_pattern=r"\S+")
    X = vec.fit_transform(bags.loc[ids])
    Z = normalize(TruncatedSVD(100, random_state=SEED).fit_transform(X))
    km = KMeans(K_CLUSTERS, n_init=10, random_state=SEED).fit(Z)
    E = pd.DataFrame({"text_id": ids, "cluster": km.labels_}).set_index("text_id")
    E["level"] = E.index.map(level)
    corr["cluster"] = corr.text_id.map(E.cluster)

    # ---- control 3: topic words ---------------------------------------------
    is_cont = corr.pos.isin(CONTENT) & corr.text.str.contains(r"[A-Za-z]")
    n_ess = corr.groupby("text_id").size()
    c_ew = corr[is_cont].groupby(["text_id", "low"]).size().rename("c_ew").reset_index()
    c_ew["rate"] = c_ew.c_ew / c_ew.text_id.map(n_ess)
    c_ew = c_ew.merge(corr[["low"]].assign(p=p_dolma).drop_duplicates("low"), on="low")
    c_ew["key_essay"] = (c_ew.c_ew >= 2) & (c_ew.rate >= KEY_RATIO * c_ew.p)
    c_ew["cluster"] = c_ew.text_id.map(E.cluster)
    n_cl_tok = corr.groupby("cluster").size()
    n_cl_ess = E.groupby("cluster").size()
    cw = c_ew.groupby(["cluster", "low"]).agg(c=("c_ew", "sum"), df=("text_id", "nunique"), p=("p", "first")).reset_index()
    cw["key_cluster"] = (cw.df >= np.maximum(3, CLUSTER_DF * cw.cluster.map(n_cl_ess))) & \
        (cw.c / cw.cluster.map(n_cl_tok) >= KEY_RATIO * cw.p)
    ke = set(zip(c_ew.text_id[c_ew.key_essay], c_ew.low[c_ew.key_essay]))
    kc = set(zip(cw.cluster[cw.key_cluster], cw.low[cw.key_cluster]))
    corr["topic_essay"] = is_cont & pd.Series([(t, w) in ke for t, w in zip(corr.text_id, corr.low)], index=corr.index)
    corr["topic_cluster"] = is_cont & pd.Series([(c, w) in kc for c, w in zip(corr.cluster, corr.low)], index=corr.index)
    cw[cw.key_cluster].sort_values(["cluster", "c"], ascending=[True, False]).to_csv(OUT / "cluster_topic_words.csv", index=False)

    # ---- control 6: genre, topic similarity, register propensity -------------
    txt = essays.set_index("text_id").corrected
    E["letter"] = [bool(LETTER_OPEN.search(txt[t][:60]) or LETTER_CLOSE.search(txt[t])) for t in E.index]
    verbs = corr[corr.tag.str.startswith("VB") | (corr.tag == "MD")]
    E["past_share"] = verbs.groupby("text_id").tag.apply(lambda s: s.isin(["VBD"]).mean())
    E["arg_markers_pk"] = [len(ARG_MARKERS.findall(txt[t])) for t in E.index] / n_ess.reindex(E.index) * 1000
    E["second_person"] = corr.assign(p2=corr.low.isin(["you", "your"])).groupby("text_id").p2.mean()
    # expository/argumentative: not a letter, not narrative, not addressed to a reader
    E["genre_arg"] = ~E.letter & (E.past_share < 0.30) & (E.second_person < 0.01)

    zN = Z[(E.level == "N").to_numpy()]
    sims = Z @ zN.T                                   # cosine (rows are unit length)
    E["topic_sim_N"] = sims.mean(1)
    cmask = (E.level == "C").to_numpy()
    nn = set()
    for j in range(zN.shape[0]):
        order = np.argsort(-sims[cmask, j])[:NN_TOPIC]
        nn.update(E.index[cmask][order])
    E["c6_topic"] = E.index.isin(nn)

    # register features: fine-tag shares, person, sentence length (no lexical frequency)
    tag_sh = pd.crosstab(corr.text_id, corr.tag, normalize="index")
    tag_sh = tag_sh.loc[:, tag_sh.columns[(corr.tag.value_counts().reindex(tag_sh.columns) >= 500).to_numpy()]]
    pron = corr.assign(p1=corr.low.isin(["i", "me", "my", "we", "us", "our"]),
                       p2=corr.low.isin(["you", "your"])).groupby("text_id")[["p1", "p2"]].mean()
    sl = corr.groupby("text_id").sent_init.mean().rename("sent_per_tok")
    R = tag_sh.join(pron).join(sl).reindex(E.index).fillna(0)
    lc = E.level.isin(["N", "C"]).to_numpy()
    Xr = StandardScaler().fit_transform(R[lc])
    lr = LogisticRegression(C=0.05, max_iter=5000).fit(Xr, (E.level[lc] == "N").to_numpy())
    E.loc[lc, "register_ps"] = lr.predict_proba(Xr)[:, 1]
    q = E.loc[E.level == "C", "register_ps"].quantile(0.75)
    E["c6_register"] = E.level.isin(["C", "N"]) & (E.register_ps >= q)
    E.loc[E.level == "N", "c6_register"] = True
    E["c6_genre"] = E.genre_arg
    E.loc[E.level == "N", "c6_topic"] = True
    E["n_tokens"] = n_ess.reindex(E.index)
    E.reset_index().to_parquet(OUT / "essays.parquet", index=False)

    keep = ["text_id", "level", "spacy_idx", "pos_in_essay", "idx", "text", "low", "lemma_low", "pos", "tag",
            "surprisal_bits", "own", "propn_num", "earlier", "topic_essay", "topic_cluster", "sent_init",
            "para_init", "freq_pm", "cluster"]
    corr[keep].to_parquet(OUT / "tokens.parquet", index=False)

    # descriptive summaries (no surprisal)
    print(pd.crosstab(E.cluster, E.level).to_string())
    print(E.groupby("level")[["letter", "genre_arg", "past_share", "arg_markers_pk", "topic_sim_N", "c6_topic",
                              "c6_register", "register_ps"]].mean().round(3).to_string())
    print(corr.groupby("level")[["propn_num", "earlier", "topic_essay", "topic_cluster", "sent_init",
                                 "para_init", "own"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
